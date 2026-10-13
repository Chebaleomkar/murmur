"""
Murmur — local, offline push-to-dictate for Windows, using the Whistle STT
model (Cactus Compute) running fully offline via the needle engine.

Hotkeys:
  Ctrl+Win   toggle recording: press once to start, press again to stop.
             While recording, speech is cut into chunks (at pauses, or
             every HARD_CUT_SECONDS if you never pause) and transcribed in
             the background as you keep talking — so there is no practical
             limit on how long you can dictate for. The joined transcript
             is pasted once, when you stop.
  Alt+Z      re-paste the last dictation (no re-recording).

A small always-on-top window shows recording/transcribing status. Every
recording is kept under history/ with its transcript in
history/transcripts.jsonl, for building a personal dataset over time.
"""

import collections
import datetime as dt
import json
import queue
import subprocess
import threading
import time
import tkinter as tk
import wave
from pathlib import Path

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd

NEEDLE_BIN = Path(r"E:\windows-x86_64\needle.exe")
MODEL_PATH = Path(r"E:\whistle.cact")
SAMPLE_RATE = 16000

BLOCK_SECONDS = 0.1
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SECONDS)

MIN_CHUNK_SECONDS = 10.0  # never cut before this much is buffered
HARD_CUT_SECONDS = 25.0  # cut regardless of silence; stays under Whistle's 30s cap
QUIET_BLOCKS_NEEDED = 3  # ~300ms of quiet confirms a real pause, not a plosive dip
# Fixed, not adaptive: an earlier version estimated the "floor" from a
# percentile of recent blocks, which breaks down during a long stretch of
# continuous speech with no actual pause yet — the estimate drifts up toward
# the speech level itself and starts misfiring. A plain fixed threshold,
# tuned against real captured levels (observed mic noise floor well under
# 0.004, real speech rms 0.01-0.1+), is simpler and doesn't have that failure
# mode. If cuts land wrong for your mic, this is the first number to adjust.
QUIET_RMS_THRESHOLD = 0.004

HISTORY_DIR = Path(__file__).resolve().parent / "history"
TRANSCRIPTS_LOG = HISTORY_DIR / "transcripts.jsonl"

state_lock = threading.Lock()
is_recording = False
last_dictation = ""
overlay = None
live_rms = collections.deque(maxlen=50)  # read by the (future) waveform UI

TOGGLE_COOLDOWN = 0.75  # seconds; the Windows key doesn't send clean key-up
# events, so `keyboard` can fire a hotkey callback more than once per actual
# press. Without this, one press could start-then-immediately-stop a
# recording, or two presses could collapse into a no-op.
_last_toggle_time = 0.0


def log_history(entry: dict) -> None:
    """Append one line per recorded chunk so every dictation + its audio is
    kept for building a personal dataset later. Never raises — a logging
    failure shouldn't break dictation itself."""
    try:
        HISTORY_DIR.mkdir(exist_ok=True)
        with open(TRANSCRIPTS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[dictate] history logging failed (non-fatal): {e}")


class Overlay:
    """Tiny always-on-top indicator window. Runs on its own Tk mainloop
    thread so it never blocks hotkey handling or transcription."""

    def __init__(self):
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.label = tk.Label(
            self.root,
            text="",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#1e1e1e",
            padx=16,
            pady=8,
        )
        self.label.pack()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"+{screen_w // 2 - 100}+{screen_h - 120}")
        self._poll()
        self.root.mainloop()

    def _poll(self):
        try:
            while True:
                text = self._queue.get_nowait()
                if text is None:
                    self.root.withdraw()
                else:
                    self.label.config(text=text)
                    self.root.deiconify()
                    self.root.lift()
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def show(self, text: str):
        self._queue.put(text)

    def hide(self):
        self._queue.put(None)


def normalize_audio(samples: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Boost quiet recordings toward full scale. A ~0.04 peak (quiet mic/room
    distance) wastes most of the 16-bit range the model expects and visibly
    hurts accuracy; capping the gain keeps a near-silent clip from being
    amplified into pure noise."""
    peak = float(np.abs(samples).max()) if len(samples) else 0.0
    if peak < 1e-4:
        return samples
    gain = min(target_peak / peak, 20.0)
    return samples * gain


def save_wav(path: Path, samples: np.ndarray) -> None:
    samples = normalize_audio(samples)
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(int_samples.tobytes())


def transcribe(wav_path: Path) -> str:
    result = subprocess.run(
        [str(NEEDLE_BIN), "--model", str(MODEL_PATH), "--audio", str(wav_path)],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    data = json.loads(result.stdout.decode("utf-8", "replace"))
    return data["text"]


def paste(text: str) -> None:
    if not text:
        return
    pyperclip.copy(text)
    keyboard.send("ctrl+v")


class ChunkWorker:
    """Transcribes chunks strictly in submission order on one background
    thread, so `needle.exe` is never run concurrently (it reloads the whole
    model per call — two in flight would just fight for CPU) and results
    come back in the order they were recorded."""

    def __init__(self):
        self._queue: "queue.Queue[tuple | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, session: "RecordingSession", index: int, samples: np.ndarray):
        with session.lock:
            session.pending_count += 1
        self._queue.put((session, index, samples))

    def _run(self):
        while True:
            session, index, samples = self._queue.get()
            duration = len(samples) / SAMPLE_RATE
            peak = float(np.abs(samples).max()) if len(samples) else 0.0
            rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
            wav_path = session.dir / f"chunk_{index:03d}.wav"
            text = ""
            try:
                save_wav(wav_path, samples)
                text = transcribe(wav_path)
                print(f"[dictate] chunk {index} ({duration:.1f}s): {text!r}")
            except Exception as e:
                print(f"[dictate] chunk {index} ERROR: {e}")
            log_history({
                "ts": dt.datetime.now().isoformat(timespec="seconds"),
                "session": session.id,
                "chunk": index,
                "wav": str(wav_path.relative_to(HISTORY_DIR)),
                "duration_s": round(duration, 2),
                "peak": round(peak, 4),
                "rms": round(rms, 4),
                "text": text,
            })
            with session.lock:
                session.results[index] = text
                session.pending_count -= 1
                done = session.finished and session.pending_count == 0
            if done:
                session.all_done.set()


chunk_worker: ChunkWorker | None = None


class RecordingSession:
    """State for one Ctrl+Win-to-Ctrl+Win recording: the audio not yet cut
    into a chunk, the noise floor estimate used to detect pauses, and the
    (possibly still in-flight) transcribed text per chunk index."""

    def __init__(self):
        self.id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.dir = HISTORY_DIR / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.pending_blocks: list[np.ndarray] = []
        self.pending_seconds = 0.0
        self.quiet_run = 0
        self.chunk_index = 0
        self.results: dict[int, str] = {}
        self.pending_count = 0
        self.finished = False
        self.all_done = threading.Event()

    def add_block(self, block: np.ndarray):
        """Called from the mic callback thread for every ~100ms block.
        Returns a chunk's samples to transcribe if a cut point was found,
        else None."""
        block_rms = float(np.sqrt(np.mean(block**2))) if len(block) else 0.0
        live_rms.append(block_rms)

        with self.lock:
            self.pending_blocks.append(block)
            self.pending_seconds += len(block) / SAMPLE_RATE

            is_quiet = block_rms < QUIET_RMS_THRESHOLD
            self.quiet_run = self.quiet_run + 1 if is_quiet else 0

            long_enough = self.pending_seconds >= MIN_CHUNK_SECONDS
            found_pause = long_enough and self.quiet_run >= QUIET_BLOCKS_NEEDED
            hit_hard_cap = self.pending_seconds >= HARD_CUT_SECONDS

            if not (found_pause or hit_hard_cap):
                return None

            chunk_samples = np.concatenate(self.pending_blocks, axis=0).reshape(-1)
            self.pending_blocks = []
            self.pending_seconds = 0.0
            self.quiet_run = 0
            index = self.chunk_index
            self.chunk_index += 1
            return index, chunk_samples

    def flush_final(self):
        """Whatever's left when recording stops, however short."""
        with self.lock:
            if not self.pending_blocks:
                return None
            chunk_samples = np.concatenate(self.pending_blocks, axis=0).reshape(-1)
            self.pending_blocks = []
            index = self.chunk_index
            self.chunk_index += 1
            return index, chunk_samples

    def joined_text(self) -> str:
        return " ".join(
            self.results[i] for i in sorted(self.results) if self.results[i]
        )


current_session: RecordingSession | None = None


def start_recording():
    global is_recording, current_session
    with state_lock:
        if is_recording:
            return
        is_recording = True

    current_session = RecordingSession()
    session = current_session
    overlay.show("🎙  Listening…")

    def callback(indata, frames, time_info, status):
        result = session.add_block(indata.copy().reshape(-1))
        if result is not None:
            index, chunk_samples = result
            chunk_worker.submit(session, index, chunk_samples)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=callback,
    )
    stream.start()
    start_recording.active_stream = stream


def stop_recording():
    global is_recording, last_dictation
    with state_lock:
        if not is_recording:
            return
        is_recording = False

    session = current_session
    stream = getattr(start_recording, "active_stream", None)
    if stream is not None:
        stream.stop()
        stream.close()

    overlay.show("⏳  Transcribing…")

    final = session.flush_final()
    if final is not None:
        index, chunk_samples = final
        chunk_worker.submit(session, index, chunk_samples)

    with session.lock:
        session.finished = True
        already_done = session.pending_count == 0
    if already_done:
        session.all_done.set()

    def wait_and_paste():
        session.all_done.wait(timeout=60)
        text = session.joined_text()
        if text:
            print(f"[dictate] full dictation: {text!r}")
            last_dictation = text
            paste(text)
        overlay.hide()

    threading.Thread(target=wait_and_paste, daemon=True).start()


def toggle_recording():
    global _last_toggle_time
    now = time.monotonic()
    if now - _last_toggle_time < TOGGLE_COOLDOWN:
        return
    _last_toggle_time = now

    if is_recording:
        stop_recording()
    else:
        start_recording()


def repaste_last():
    if last_dictation:
        print(f"[dictate] re-pasting: {last_dictation!r}")
        paste(last_dictation)
    else:
        print("[dictate] no previous dictation yet")


def main():
    global overlay, chunk_worker
    if not NEEDLE_BIN.is_file():
        raise SystemExit(f"needle binary not found at {NEEDLE_BIN}")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"whistle.cact not found at {MODEL_PATH}")

    overlay = Overlay()
    chunk_worker = ChunkWorker()

    keyboard.add_hotkey("ctrl+windows", toggle_recording)
    keyboard.add_hotkey("alt+z", repaste_last)

    print("[dictate] ready — Ctrl+Win to dictate, Alt+Z to re-paste last result")
    print("[dictate] press Ctrl+C in this window to quit")
    try:
        keyboard.wait()
    except KeyboardInterrupt:
        print("[dictate] bye")


if __name__ == "__main__":
    main()

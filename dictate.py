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
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pyperclip
import sounddevice as sd

import hotkeys
from overlay import Overlay

WINDOWS = sys.platform == "win32"
# A PyInstaller build runs from a temp folder that is deleted on exit, so
# history lives next to the executable instead.
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def _find(*candidates: Path) -> Path:
    """First candidate that exists, else the first one (main() reports it)."""
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


NEEDLE_BIN = _find(
    APP_DIR / ("needle.exe" if WINDOWS else "needle"),
    *([Path(r"E:\windows-x86_64\needle.exe")] if WINDOWS else []),
)
MODEL_PATH = _find(
    APP_DIR / "whistle.cact",
    *([Path(r"E:\whistle.cact")] if WINDOWS else []),
)
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

# Whistle (like most STT models) hallucinates a stock phrase ("Thank you.",
# "Bye.", "you") on audio with no real speech instead of returning empty.
# A whole-chunk peak threshold does not catch this: a single transient
# block anywhere in the clip (a tap, a cough, the mic bumping) passes any
# peak threshold exactly as easily as real speech does, since peak only
# ever looks at the single loudest sample in the whole chunk. What actually
# distinguishes speech from a transient is duration: speech holds energy
# across many consecutive blocks, a transient does not. Count blocks whose
# RMS clears this threshold and only transcribe once enough of them do.
VOICED_RMS_THRESHOLD = 0.01
MIN_VOICED_BLOCKS = 3  # ~300ms of sustained energy, not a single spike
# Clips under a second were noise in practice (taps, breaths) and the model
# turned them into "Thank you." or other junk. Tradeoff: a short final word
# after a pause can be dropped too.
MIN_SPEECH_SECONDS = 1.0


def count_voiced_blocks(samples: np.ndarray) -> int:
    usable_len = (len(samples) // BLOCK_SIZE) * BLOCK_SIZE
    if usable_len == 0:
        return 0
    blocks = samples[:usable_len].reshape(-1, BLOCK_SIZE)
    block_rms = np.sqrt(np.mean(blocks**2, axis=1))
    return int(np.sum(block_rms > VOICED_RMS_THRESHOLD))
HISTORY_DIR = APP_DIR / "history"
TRANSCRIPTS_LOG = HISTORY_DIR / "transcripts.jsonl"

state_lock = threading.Lock()
is_recording = False
last_dictation = ""
overlay = None
live_rms = collections.deque(maxlen=50)  # read by the overlay waveform

DICTATE_HOTKEYS = ("ctrl+windows", "f9")  # hold to talk; change here to rebind
# Keys whose release ends a hold. Releasing any of them stops recording.
DICTATE_RELEASE_KEYS = ("ctrl", "windows", "f9")
REPASTE_HOTKEY = "ctrl+alt+z"


MURMUR_LOG = HISTORY_DIR / "murmur.log"


def log(message: str) -> None:
    """Under pythonw there is no stdout, so print() silently does nothing.
    Also append to a file so failures leave a record."""
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
    print(line)
    try:
        HISTORY_DIR.mkdir(exist_ok=True)
        with open(MURMUR_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def log_history(entry: dict) -> None:
    """Append one line per recorded chunk so every dictation + its audio is
    kept for building a personal dataset later. Never raises — a logging
    failure shouldn't break dictation itself."""
    try:
        HISTORY_DIR.mkdir(exist_ok=True)
        with open(TRANSCRIPTS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        log(f"[dictate] history logging failed (non-fatal): {e}")


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
        # Under pythonw there is no parent console, so Windows would pop a
        # new console window for every needle.exe run (once per chunk).
        creationflags=subprocess.CREATE_NO_WINDOW if WINDOWS else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    data = json.loads(result.stdout.decode("utf-8", "replace"))
    return data["text"]


def paste(text: str) -> None:
    if not text:
        return
    pyperclip.copy(text)
    hotkeys.send_paste()


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
            # Check speech on the boosted audio: quiet real speech otherwise
            # fails the block count, and save_wav boosts it anyway.
            boosted = normalize_audio(samples)
            voiced_blocks = count_voiced_blocks(boosted)
            if duration < MIN_SPEECH_SECONDS or voiced_blocks < MIN_VOICED_BLOCKS:
                log(f"[dictate] chunk {index} ({duration:.1f}s): skipped, "
                      f"{voiced_blocks} voiced block(s)")
                save_wav(wav_path, samples)  # keep in history for debugging
            else:
                try:
                    save_wav(wav_path, samples)
                    text = transcribe(wav_path)
                    log(f"[dictate] chunk {index} ({duration:.1f}s): {text!r}")
                except Exception as e:
                    log(f"[dictate] chunk {index} ERROR: {e}")
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
        self.dropped_startup_block = False

    def add_block(self, block: np.ndarray):
        """Called from the mic callback thread for every ~100ms block.
        Returns a chunk's samples to transcribe if a cut point was found,
        else None.

        The very first block of a fresh stream carries a startup transient
        on this hardware (measured peak 0.18-0.62, confirmed across
        repeated trials) that dwarfs the real noise floor (peak 0.003-0.02)
        for the rest of the recording. Left in, it would count as one
        voiced block toward MIN_VOICED_BLOCKS on every single recording,
        for free, regardless of whether anyone said anything. Drop it
        outright rather than trying to filter it later."""
        if not self.dropped_startup_block:
            self.dropped_startup_block = True
            return None

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
    overlay.show_listening()

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

    overlay.show_transcribing()

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
        global last_dictation
        session.all_done.wait(timeout=60)
        text = session.joined_text()
        if text:
            log(f"[dictate] full dictation: {text!r}")
            last_dictation = text
            paste(text)
            overlay.hide()
        else:
            log("[dictate] nothing transcribed for this recording")
            overlay.show_error("⚠  nothing transcribed")
            time.sleep(1.5)
            overlay.hide()

    threading.Thread(target=wait_and_paste, daemon=True).start()


class HotkeyController:
    """DICTATE_HOTKEYS: hold to talk, release to stop. Windows auto-repeats a
    held key, so repeated presses while held are ignored."""

    def __init__(self, start, stop):
        self._start, self._stop = start, stop
        self._held = False

    def press(self):
        if self._held:
            return
        self._held = True
        self._start()

    def release(self):
        if not self._held:
            return
        self._held = False
        self._stop()


def dictate_latest_recording():
    """Transcribe and paste the most recent saved recording from history."""
    sessions = sorted(p for p in HISTORY_DIR.glob("*") if p.is_dir())
    if not sessions:
        log("[dictate] no saved recordings yet")
        return
    chunks = sorted(sessions[-1].glob("chunk_*.wav"))
    threading.Thread(target=_transcribe_and_paste, args=(chunks,), daemon=True).start()


def _transcribe_and_paste(chunks):
    global last_dictation
    overlay.show_transcribing()
    texts = []
    for chunk in chunks:
        try:
            texts.append(transcribe(chunk))
        except Exception as e:
            log(f"[dictate] re-transcribe {chunk.name} ERROR: {e}")
    text = " ".join(t for t in texts if t)
    if text:
        log(f"[dictate] re-transcribed latest: {text!r}")
        last_dictation = text
        paste(text)
        overlay.hide()
    else:
        log("[dictate] nothing transcribed from latest recording")
        overlay.show_error("nothing transcribed")
        time.sleep(1.5)
        overlay.hide()


def main():
    global overlay, chunk_worker
    if not NEEDLE_BIN.is_file():
        raise SystemExit(f"needle binary not found at {NEEDLE_BIN}")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"whistle.cact not found at {MODEL_PATH}")

    overlay = Overlay(lambda: list(live_rms), QUIET_RMS_THRESHOLD)
    chunk_worker = ChunkWorker()

    hotkey = HotkeyController(start_recording, stop_recording)
    hotkeys.start(
        DICTATE_HOTKEYS, DICTATE_RELEASE_KEYS, REPASTE_HOTKEY,
        on_press=hotkey.press, on_release=hotkey.release,
        repaste=dictate_latest_recording,
    )

    log(f"[dictate] ready: hold {' or '.join(DICTATE_HOTKEYS)} to talk; {REPASTE_HOTKEY} dictates the latest recording")
    try:
        overlay.exec()  # Qt needs the main thread; hotkeys run in their own
    except KeyboardInterrupt:
        pass
    print("[dictate] bye")


if __name__ == "__main__":
    main()

"""
Quick local dictation for Windows, using the Whistle STT model (Cactus Compute)
running fully offline via the needle engine.

Hotkeys:
  Ctrl+Win   toggle recording: press once to start, press again to stop,
             transcribe, and paste into whatever window currently has focus.
  Alt+Z      re-paste the last dictation (no re-recording).

A small always-on-top window appears while recording so you can see the mic
is actually listening — that's the only acknowledgment; no sounds.
"""

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
MAX_RECORD_SECONDS = 28  # Whistle's engine caps at 30s; stop a little early.

state_lock = threading.Lock()
is_recording = False
last_dictation = ""
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
overlay = None

TOGGLE_COOLDOWN = 0.75  # seconds; the Windows key doesn't send clean key-up
# events, so `keyboard` can fire a hotkey callback more than once per actual
# press. Without this, one press could start-then-immediately-stop a
# recording, or two presses could collapse into a no-op.
_last_toggle_time = 0.0


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


def start_recording():
    global is_recording
    with state_lock:
        if is_recording:
            return
        is_recording = True

    while not audio_queue.empty():
        audio_queue.get_nowait()

    overlay.show("🎙  Listening…")

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    )
    stream.start()

    def auto_stop_guard():
        start = time.monotonic()
        while True:
            with state_lock:
                if not is_recording:
                    return
            if time.monotonic() - start > MAX_RECORD_SECONDS:
                print(f"[dictate] hit {MAX_RECORD_SECONDS}s cap, auto-stopping")
                stop_recording(stream)
                return
            time.sleep(0.2)

    threading.Thread(target=auto_stop_guard, daemon=True).start()
    # Stash the stream where the hotkey handler can find it to stop cleanly.
    start_recording.active_stream = stream


def stop_recording(stream=None):
    global is_recording, last_dictation
    with state_lock:
        if not is_recording:
            return
        is_recording = False

    stream = stream or getattr(start_recording, "active_stream", None)
    if stream is not None:
        stream.stop()
        stream.close()

    overlay.show("⏳  Transcribing…")

    chunks = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get_nowait())

    if not chunks:
        overlay.hide()
        return

    samples = np.concatenate(chunks, axis=0).reshape(-1)
    duration = len(samples) / SAMPLE_RATE
    peak = float(np.abs(samples).max()) if len(samples) else 0.0
    rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
    print(f"[dictate] recorded {duration:.1f}s  peak={peak:.4f}  rms={rms:.4f}")
    if peak < 0.01:
        print("[dictate] WARNING: captured audio is near-silent — check mic "
              "device/mute/privacy settings, this is likely why transcription "
              "is wrong")

    # Kept (not deleted) on purpose right now so a bad capture can be
    # inspected after the fact instead of guessing from the transcript alone.
    tmp_wav = Path.cwd() / "last_recording.wav"
    try:
        save_wav(tmp_wav, samples)
        text = transcribe(tmp_wav)
        print(f"[dictate] text: {text!r}")
        last_dictation = text
        paste(text)
        overlay.hide()
    except Exception as e:
        print(f"[dictate] ERROR: {e}")
        overlay.show("⚠  dictation failed, see console")
        time.sleep(1.5)
        overlay.hide()


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
    global overlay
    if not NEEDLE_BIN.is_file():
        raise SystemExit(f"needle binary not found at {NEEDLE_BIN}")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"whistle.cact not found at {MODEL_PATH}")

    overlay = Overlay()

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

# Murmur

Local, offline push-to-dictate for Windows. Runs entirely on-device using
[Whistle](https://huggingface.co/Cactus-Compute/whistle) (Cactus Compute)
via its `needle` engine — no cloud, no API keys, no network calls at runtime.

## Hotkeys

- `Ctrl+Win` — toggle recording: press once to start, press again to stop,
  transcribe, and paste into whatever window currently has focus.
- `Alt+Z` — re-paste the last dictation.

## Requirements

- The `needle` engine binary and `whistle.cact` weights (see
  [Cactus's needle project](https://github.com/cactus-compute/needle) for how
  to fetch them: `needle download windows-x86_64` then `needle download whistle`).
  Update `NEEDLE_BIN` / `MODEL_PATH` at the top of `dictate.py` to wherever
  you place them.
- Python 3.11+ with `keyboard`, `sounddevice`, `pyperclip`, `numpy` installed.

## Running

```
python dictate.py
```

Keep the console window open — closing it stops the hotkeys.

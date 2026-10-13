# Murmur

Local, offline dictation for Windows. Runs entirely on-device using
[Whistle](https://huggingface.co/Cactus-Compute/whistle) (Cactus Compute),
a 16.9 MB speech-to-text model, via its `needle` engine. No cloud, no API
keys, no network calls at runtime.

## Hotkeys

- `Ctrl+Win` — toggle recording: press once to start, press again to stop.
  While recording, speech is cut into chunks (at natural pauses, or every
  25 seconds if you never pause) and transcribed in the background as you
  keep talking, so there's no practical limit on how long you can dictate
  for. The joined transcript is pasted once, into whatever window has
  focus, when you stop.
- `Alt+Z` — re-paste the last dictation.

A small pill-shaped indicator shows recording/transcribing status, with a
waveform that reacts to your actual mic input.

## Dataset

Every recorded chunk and its transcript is saved under `history/`, with one
line per chunk appended to `history/transcripts.jsonl` (timestamp, session,
duration, peak/rms, text). Useful for reviewing what got transcribed, or for
building a personal voice dataset over time. This directory is gitignored;
it's local data, not part of the repo.

## Requirements

- The `needle` engine binary and `whistle.cact` weights. Install
  `cactus-needle` from PyPI, then:
  ```
  needle download windows-x86_64
  needle download whistle
  ```
  Update `NEEDLE_BIN` / `MODEL_PATH` at the top of `dictate.py` to wherever
  the resulting `needle.exe` and `whistle.cact` end up.
- Python 3.11+ with the packages in `requirements.txt`:
  ```
  pip install -r requirements.txt
  ```

## Running

```
python dictate.py
```

Keep the console window open — closing it stops the hotkeys.

## Notes

- Tested on Windows with a single microphone input device.
- Whistle's engine caps a single inference call at 30 seconds of audio;
  Murmur stays under that per chunk internally, there's no need to manage
  this yourself.
- If `Ctrl+Win` doesn't register, try running from a terminal opened as
  Administrator — the `keyboard` library's handling of the Windows key can
  need elevated privileges on some setups.

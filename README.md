# Murmur

Local, offline dictation for Windows. Runs entirely on-device using
[Whistle](https://huggingface.co/Cactus-Compute/whistle) (Cactus Compute),
a 16.9 MB speech-to-text model, via its `needle` engine. No cloud, no API
keys, no network calls at runtime.

## Hotkeys

- `Ctrl+Win` or `F9` — hold to talk, release to stop. While you hold the key,
  speech is cut into chunks (at natural pauses, or every 25 seconds if you
  never pause) and transcribed in the background, so there's no practical
  limit on how long you can dictate for. The joined transcript is pasted
  once, into whatever window has focus, when you release.
- `Ctrl+Alt+Z` — transcribe the most recent saved recording and paste it.

The key you hold is the `DICTATE_HOTKEYS` setting at the top of `dictate.py`.
Releasing `Ctrl`, `Win` or `F9` ends a recording. The Fn key on most laptops
is handled by firmware and can't be bound from Windows.

A small pill-shaped indicator (`overlay.py`) shows listening, transcribing,
and error states, with a waveform that reacts to your microphone input. It
never takes focus, so the paste still goes to the window you were typing in.

## Audio handling

Quiet recordings are boosted before transcription. Each chunk is checked for
sustained speech after the boost, and clips shorter than one second are
skipped, since those were mostly noise. A very short final word after a pause
can be dropped as a result.

## Dataset and logs

Every recorded chunk and its transcript is saved under `history/`, with one
line per chunk appended to `history/transcripts.jsonl` (timestamp, session,
duration, peak/rms, text). Chunk results, skipped clips and errors are also
appended to `history/murmur.log`, since the app running under `pythonw` has
no console to print to. This directory is gitignored; it's local data, not
part of the repo.

## Requirements

- The `needle` engine binary and `whistle.cact` weights. Install
  `cactus-needle` from PyPI, then:
  ```
  needle download windows-x86_64
  needle download whistle
  ```
  Update `NEEDLE_BIN` / `MODEL_PATH` at the top of `dictate.py` to wherever
  the resulting `needle.exe` and `whistle.cact` end up.
- Python 3.11+ with the packages in `requirements.txt`, ideally in a virtual
  environment:
  ```
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  ```

## Running

```
python dictate.py
```

Keep that console window open; closing it stops the hotkeys. To run without a
console window, use `pythonw dictate.py` instead. Murmur then logs to
`history/murmur.log` rather than the console.

To start it at login, place a shortcut to `pythonw.exe dictate.py` (with the
working directory set to this folder) in your Startup folder.

## Notes

- Tested on Windows with a single microphone input device.
- Whistle's engine caps a single inference call at 30 seconds of audio;
  Murmur stays under that per chunk internally, so you don't need to manage
  this yourself.
- If a hotkey doesn't register, try running from a terminal opened as
  Administrator. The `keyboard` library's handling of the Windows key can
  need elevated privileges on some setups.

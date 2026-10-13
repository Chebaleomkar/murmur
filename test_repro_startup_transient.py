"""Regression repro for the startup-transient hallucination bug.

A fresh audio stream's first ~100ms block carries a hardware transient
(measured peak 0.18-0.62 across repeated trials on the dev machine) that
dwarfs the real noise floor (peak 0.003-0.02) for the rest of a silent
recording. Before the fix, RecordingSession.add_block() let that block
pollute the whole chunk's peak, so MIN_SIGNAL_PEAK never caught a short
accidental recording with no real speech, and Whistle hallucinated a stock
phrase ("Thank you.") on it.

Run directly: produces real needle.exe output, not a mock.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
import dictate

rng = np.random.default_rng(42)


def synthetic_accidental_recording():
    """First block: startup transient shape actually measured on this
    machine. Rest: genuine quiet floor. No real speech anywhere."""
    transient = rng.standard_normal(dictate.BLOCK_SIZE).astype(np.float32) * 0.3
    silence_blocks = [
        rng.standard_normal(dictate.BLOCK_SIZE).astype(np.float32) * 0.003
        for _ in range(8)  # ~0.8s, matching the reported 0.7-1.9s durations
    ]
    return [transient] + silence_blocks


def run_session(blocks) -> str:
    dictate.chunk_worker = dictate.ChunkWorker()
    session = dictate.RecordingSession()
    for b in blocks:
        result = session.add_block(b)
        if result is not None:
            index, samples = result
            dictate.chunk_worker.submit(session, index, samples)
    final = session.flush_final()
    if final is not None:
        index, samples = final
        dictate.chunk_worker.submit(session, index, samples)
    with session.lock:
        session.finished = True
        done = session.pending_count == 0
    if done:
        session.all_done.set()
    session.all_done.wait(timeout=30)
    return session.joined_text()


if __name__ == "__main__":
    text = run_session(synthetic_accidental_recording())
    print(f"result: {text!r}")
    if text:
        print("FAIL: accidental silent recording produced text (hallucination bug present)")
        sys.exit(1)
    print("PASS: accidental silent recording produced no text")

# Project guidance

Minimal Windows cmd-to-OBS Swedish subtitle tool. Keep it small: no Node.js, React, dashboard, frontend build or cloud inference.

- start.cmd sets up a local Python environment once and runs run.py. Default microphone capture, one local Python process, green-screen HTML at http://localhost:8000 for OBS Browser source + Chroma Key.
- Native microphone capture must use a bounded queue and timestamps. Reject stale (>1s) audio; never silently build a backlog. Stream resampling uses soxr at the device native rate.
- Backend: Pianissimo ONNX int8 CPU, default4threads. Models cached under models/. No weights or recordings in Git.
- Single producer, passive OBS clients. Interim/final text shares utterance id. Stop flushes tail.
- Hard3s queue+inference job budget. Expired output never publishes. Native inference retains exclusive model ownership until it finishes even after timeout.
- Overlay contains only green background and bottom subtitles. No control/status chrome on-air, local fonts, immediate clear on disconnect.
- Validate backend/tests, CLI list-devices, and real prerecorded PCM through websocket without opening a microphone. Do not claim prerecorded tests measure actual microphone-to-OBS latency.
- Preserve model and upstream attribution in NOTICE.md.

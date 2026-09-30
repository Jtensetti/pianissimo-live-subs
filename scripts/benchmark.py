"""Measure local Pianissimo CPU inference on a WAV file, without microphone access.

Run from the repository root:
    python scripts/benchmark.py path/to/swedish.wav --threads 4

This measures recognition after audio is already buffered. It is not a measure of
microphone-to-subtitle delay. The first run downloads the official model (~660 MB).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time
import wave


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="Mono 16-bit PCM WAV file")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--seconds", type=float, nargs="+", default=[2, 4, 0],
        help="Lengths to test from the file start; 0 means the full recording",
    )
    parser.add_argument("--offset", type=float, default=0, help="Skip this many seconds")
    parser.add_argument("--offline", action="store_true", help="Use only cached model files")
    parser.add_argument("--json", type=Path, help="Optional detailed JSON results path")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or args.offset < 0 or any(s < 0 for s in args.seconds):
        parser.error("Threads/repeats must be positive; offset and lengths cannot be negative")

    # Hugging Face reads these settings during import, so configure them first.
    repository = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(repository / "models"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import numpy as np
    import onnx_asr
    import onnxruntime as ort

    with wave.open(str(args.wav), "rb") as audio_file:
        if audio_file.getnchannels() != 1 or audio_file.getsampwidth() != 2:
            parser.error("Provide a mono 16-bit PCM WAV file")
        sample_rate = audio_file.getframerate()
        samples = np.frombuffer(audio_file.readframes(audio_file.getnframes()), dtype="<i2")
    audio = samples.astype(np.float32) / 32768.0
    audio = audio[round(args.offset * sample_rate):]
    if len(audio) < round(sample_rate * 0.1):
        parser.error("At least 0.1 seconds of audio must remain after the offset")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    model_id = "KlangAI/pianissimo-sv-onnx"
    print("Loading Pianissimo int8 on CPU...", flush=True)
    started = time.perf_counter()
    model = onnx_asr.load_model(
        model_id, quantization="int8", providers=["CPUExecutionProvider"],
        sess_options=options,
    )
    load_seconds = time.perf_counter() - started
    report = {
        "model": model_id,
        "quantization": "int8",
        "provider": "CPUExecutionProvider",
        "threads": args.threads,
        "repeats": args.repeats,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "onnx_asr": importlib.metadata.version("onnx-asr"),
        "onnxruntime": ort.__version__,
        "wav": str(args.wav.resolve()),
        "sample_rate": sample_rate,
        "offset_seconds": args.offset,
        "hf_home": os.environ["HF_HOME"],
        "model_load_seconds": load_seconds,
        "measurement": "Warmed CPU recognition only; excludes capture, buffering, network and display",
        "results": [],
    }

    print(f"Model ready after {load_seconds:.2f} s. Warming each requested length before measurement.", flush=True)
    for requested in args.seconds:
        segment = audio if requested == 0 else audio[:round(requested * sample_rate)]
        duration = len(segment) / sample_rate
        model.recognize(segment, sample_rate=sample_rate)
        timings = []
        texts = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            texts.append(model.recognize(segment, sample_rate=sample_rate))
            timings.append(time.perf_counter() - started)
        median = statistics.median(timings)
        result = {
            "requested_seconds": requested,
            "audio_seconds": duration,
            "recognition_seconds": timings,
            "median_seconds": median,
            "rtf": median / duration,
            "realtime_multiple": duration / median,
            "text": texts[-1],
            "consistent_text": len(set(texts)) == 1,
        }
        report["results"].append(result)
        print(f"{duration:.3f} s audio: {median:.3f} s median, RTF {median / duration:.3f}, {duration / median:.1f}x", flush=True)
        print(f"  {texts[-1]}", flush=True)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Detailed results: {args.json.resolve()}")
    print("These times exclude audio buffering and OBS; they are not live subtitle latency.")


if __name__ == "__main__":
    main()

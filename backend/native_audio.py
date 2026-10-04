"""Native microphone capture with a small, explicitly bounded audio queue."""

import asyncio
from dataclasses import dataclass
import json
import queue
import time

import numpy as np


class CaptureError(RuntimeError):
    pass


@dataclass
class AudioPacket:
    samples: np.ndarray
    captured_at: float


class CaptureBuffer:
    """PortAudio callback producer; asyncio consumes and resamples outside it."""

    def __init__(self, sample_rate, *, max_chunks=5, max_age=1.0, clock=time.monotonic):
        self.sample_rate = sample_rate
        self.max_age = max_age
        self.clock = clock
        self.queue = queue.Queue(maxsize=max_chunks)
        self.error = None

    def push(self, indata, frames, status=None):
        if self.error:
            return
        if status:
            self.error = f"Ljudinmatningen tappade data: {status}. Inspelningen har stoppats."
            return
        packet = AudioPacket(indata[:, 0].copy(), self.clock() - frames / self.sample_rate)
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.error = "Ljudkön är full. Inspelningen har stoppats för att undvika fördröjda undertexter."

    def pop(self):
        if self.error:
            raise CaptureError(self.error)
        try:
            packet = self.queue.get_nowait()
        except queue.Empty:
            return None
        if self.clock() - packet.captured_at > self.max_age:
            raise CaptureError("Mikrofonljudet är mer än en sekund försenat. Inspelningen har stoppats.")
        return packet.samples


async def capture_microphone(device, stop_event, clear_overlay, *, sensitivity=4, diagnostics=False):
    """Capture until Ctrl+C or an explicit error; never automatically restart."""
    import sounddevice as sd
    import soxr
    from websockets.asyncio.client import connect

    info = sd.query_devices(device, kind="input")
    rate = float(info["default_samplerate"])
    sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=rate)
    buffer = CaptureBuffer(rate)
    resampler = soxr.ResampleStream(rate, 16000, 1, dtype="float32", quality="HQ")

    def callback(indata, frames, _time_info, status):
        buffer.push(indata, frames, status)
        if buffer.error:
            raise sd.CallbackAbort

    stream = sd.InputStream(device=device, samplerate=rate, channels=1, dtype="float32",
                            blocksize=round(rate * .1), callback=callback)
    sender = receiver = None
    try:
        async with connect(f"ws://127.0.0.1:8000/ws/transcribe?model=pianissimo&instant=true&vad={sensitivity}",
                           compression=None, max_queue=8, write_limit=16000, close_timeout=1) as ws:
            ready = json.loads(await ws.recv())
            if ready.get("type") != "ready":
                raise CaptureError(ready.get("message", "Transkriberingen kunde inte starta."))
            stream.start()
            print(f"Mikrofon: {info['name']} ({rate:g} Hz -> 16000 Hz)", flush=True)

            async def send_samples(samples):
                if not np.isfinite(samples).all():
                    raise CaptureError("Mikrofonen gav ogiltiga ljudvärden. Inspelningen har stoppats.")
                # A resampling filter may overshoot slightly at loud transients.
                samples = np.clip(samples, -1, 1)
                # Streaming soxr can return no output while filling its filter.
                for offset in range(0, len(samples), 16000):
                    payload = samples[offset:offset + 16000].astype("<f4", copy=False).tobytes()
                    try:
                        await asyncio.wait_for(ws.send(payload), timeout=.5)
                    except TimeoutError as exc:
                        raise CaptureError("Ljudöverföringen tog för lång tid. Inspelningen har stoppats.") from exc

            async def send_audio():
                level_report = time.monotonic()
                peak_rms = 0.0
                while not stop_event.is_set():
                    audio = buffer.pop()
                    if audio is not None:
                        if diagnostics:
                            peak_rms = max(peak_rms, float(np.sqrt(np.mean(audio * audio))))
                            if time.monotonic() - level_report >= 5:
                                dbfs = 20 * np.log10(max(peak_rms, 1e-8))
                                print(f"Ljudnivå: {dbfs:.0f} dBFS (högsta RMS senaste 5 s); känslighet {sensitivity}/5.", flush=True)
                                peak_rms = 0.0
                                level_report = time.monotonic()
                        await send_samples(resampler.resample_chunk(audio))
                    elif not stream.active:
                        raise CaptureError("Mikrofonen har stannat oväntat.")
                    else:
                        await asyncio.sleep(.01)
                # Stop callbacks first, then send every remaining fresh sample and
                # the resampler tail before asking the backend to finalize speech.
                stream.stop()
                while (audio := buffer.pop()) is not None:
                    await send_samples(resampler.resample_chunk(audio))
                await send_samples(resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True))
                await ws.send(json.dumps({"type": "stop"}))

            async def receive_results():
                async for payload in ws:
                    event = json.loads(payload)
                    if event.get("type") == "error":
                        raise CaptureError(event.get("message", "Transkriberingen har stoppats."))
                    if event.get("type") == "stopped":
                        return
                    if diagnostics and event.get("type") == "transcription" and event.get("mode") == "final":
                        metrics = event.get("metrics", {})
                        print(f"Textning: {metrics.get('audio_seconds', 0):g} s ljud; "
                              f"beräkning {metrics.get('inference_ms', 0):.0f} ms; "
                              f"kö {metrics.get('queue_ms', 0):.0f} ms.", flush=True)
                raise CaptureError("Anslutningen till transkriberingen stängdes.")

            sender = asyncio.create_task(send_audio())
            receiver = asyncio.create_task(receive_results())
            try:
                done, _ = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
                if receiver in done:
                    await receiver
                    await sender
                else:
                    await sender
                    await asyncio.wait_for(receiver, timeout=4)
            except BaseException:
                stream.abort()
                clear_overlay()
                raise
            finally:
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
    finally:
        stream.close()

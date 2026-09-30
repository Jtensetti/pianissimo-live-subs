import asyncio
from contextlib import asynccontextmanager
import json

import numpy as np
import pytest

from backend.native_audio import CaptureBuffer, CaptureError, capture_microphone


def test_capture_callback_copies_portaudio_memory_and_timestamps_first_sample():
    buffer = CaptureBuffer(48000, clock=lambda: 100)
    audio = np.full((4800, 1), .5, dtype=np.float32)
    buffer.push(audio, len(audio))
    audio[:] = 0  # PortAudio reuses this memory after the callback returns.
    packet = buffer.queue.get_nowait()
    assert packet.captured_at == pytest.approx(99.9)
    np.testing.assert_array_equal(packet.samples, np.full(4800, .5, dtype=np.float32))


def test_capture_queue_overflow_is_fatal_instead_of_silently_dropping_audio():
    buffer = CaptureBuffer(16000)
    audio = np.ones((1600, 1), dtype=np.float32)
    for _ in range(6):
        buffer.push(audio, len(audio))
    assert buffer.queue.qsize() == 5
    with pytest.raises(CaptureError, match="full"):
        buffer.pop()


def test_capture_rejects_audio_older_than_one_second():
    now = [100.0]
    buffer = CaptureBuffer(16000, clock=lambda: now[0])
    buffer.push(np.zeros((1600, 1), dtype=np.float32), 1600)
    now[0] = 101
    with pytest.raises(CaptureError, match="en sekund"):
        buffer.pop()


def test_portaudio_overflow_status_is_propagated_to_consumer():
    buffer = CaptureBuffer(16000)
    buffer.push(np.zeros((1600, 1), dtype=np.float32), 1600, "input overflow")
    with pytest.raises(CaptureError, match="input overflow"):
        buffer.pop()


@pytest.mark.parametrize("sample_rate", [44100, 48000])
def test_native_capture_resamples_and_flushes_all_samples_before_stop_without_hardware(monkeypatch, sample_rate):
    import sounddevice as sd
    from websockets.asyncio import client

    async def scenario():
        stop = asyncio.Event()
        sent = []
        incoming = asyncio.Queue()
        streams = []
        class FakeStream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
                self.active = False
                self.closed = False
                streams.append(self)
            def start(self):
                self.active = True
                # A full-scale square wave verifies resampler overshoot is clipped.
                frames = round(sample_rate * .1)
                audio = np.ones((frames, 1), dtype=np.float32)
                audio[frames // 2:] = -1
                for _ in range(2):
                    self.callback(audio, frames, None, False)
                stop.set()
            def stop(self):
                self.active = False
            abort = stop
            def close(self):
                self.closed = True
        class FakeWebSocket:
            async def recv(self):
                return json.dumps({"type": "ready"})
            async def send(self, payload):
                sent.append(payload)
                if isinstance(payload, str):
                    assert json.loads(payload) == {"type": "stop"}
                    await incoming.put(json.dumps({"type": "stopped"}))
            def __aiter__(self):
                return self
            async def __anext__(self):
                return await incoming.get()
        @asynccontextmanager
        async def fake_connect(*_args, **_kwargs):
            yield FakeWebSocket()
        monkeypatch.setattr(sd, "query_devices", lambda *_args, **_kwargs: {"name": "Synthetic microphone", "default_samplerate": sample_rate})
        monkeypatch.setattr(sd, "check_input_settings", lambda **_kwargs: None)
        monkeypatch.setattr(sd, "InputStream", FakeStream)
        monkeypatch.setattr(client, "connect", fake_connect)
        clear_events = []
        await capture_microphone(None, stop, lambda: clear_events.append("clear"))
        assert isinstance(sent[-1], str)
        audio = np.concatenate([np.frombuffer(payload, dtype="<f4") for payload in sent if isinstance(payload, bytes)])
        assert len(audio) == 3200  # 200 ms at16kHz, including the resampler tail.
        assert np.max(np.abs(audio)) <= 1
        assert streams[0].closed and not streams[0].active
        assert clear_events == []
    asyncio.run(scenario())


def test_backend_error_stops_native_microphone_and_clears_overlay_without_auto_restart(monkeypatch):
    import sounddevice as sd
    from websockets.asyncio import client

    async def scenario():
        stop = asyncio.Event()
        streams = []
        class FakeStream:
            def __init__(self, **_kwargs):
                self.active = False
                self.closed = False
                self.starts = 0
                streams.append(self)
            def start(self):
                self.starts += 1
                self.active = True
            def abort(self):
                self.active = False
            def close(self):
                self.closed = True
        class FakeWebSocket:
            async def recv(self):
                return json.dumps({"type": "ready"})
            def __aiter__(self):
                return self
            async def __anext__(self):
                return json.dumps({"type": "error", "message": "3-second deadline exceeded"})
        @asynccontextmanager
        async def fake_connect(*_args, **_kwargs):
            yield FakeWebSocket()
        monkeypatch.setattr(sd, "query_devices", lambda *_args, **_kwargs: {"name": "Synthetic microphone", "default_samplerate": 48000})
        monkeypatch.setattr(sd, "check_input_settings", lambda **_kwargs: None)
        monkeypatch.setattr(sd, "InputStream", FakeStream)
        monkeypatch.setattr(client, "connect", fake_connect)
        clear_events = []
        with pytest.raises(CaptureError, match="deadline"):
            await capture_microphone(None, stop, lambda: clear_events.append("clear"))
        assert streams[0].closed and not streams[0].active
        assert streams[0].starts == 1 and clear_events == ["clear"]
    asyncio.run(scenario())

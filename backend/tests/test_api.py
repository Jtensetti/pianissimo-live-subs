import asyncio
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.main import ModelService, SubtitleHub, create_app


class FakeModel:
    def __init__(self):
        self.calls = []

    def recognize(self, audio, sample_rate):
        assert sample_rate == 16000 and audio.dtype == np.float32 and audio.ndim == 1
        self.calls.append(audio.copy())
        return f"Svensk text {len(self.calls)}"


def ready_service(model=None):
    service = ModelService()
    service.model = model or FakeModel()
    service.status = "ready"
    return service


def sound(seconds, amplitude=.1):
    return np.full(round(seconds * 16000), amplitude, dtype="<f4").tobytes()


def send_sound(ws, seconds, amplitude=.1):
    payload = sound(seconds, amplitude)
    for offset in range(0, len(payload), 64000):
        ws.send_bytes(payload[offset:offset + 64000])


def until_type(ws, kind):
    for _ in range(30):
        event = ws.receive_json()
        if event["type"] == kind:
            return event
    pytest.fail(f"No {kind} event received")


def wait_status(client, expected):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = client.get("/model-status").json()
        if result["status"] == expected:
            return result
        time.sleep(.01)
    pytest.fail(f"Expected status {expected}")


def test_import_and_observer_do_not_load_model_or_claim_microphone():
    calls = []
    service = ModelService(lambda: calls.append(1))
    app = create_app(service)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/model-status").json()["is_loaded"] is False
        with client.websocket_connect("/ws/subtitles") as observer:
            assert observer.receive_json() == {"type": "ready", "role": "observer"}
            assert app.state.producer is None
        with client.websocket_connect("/ws/transcribe") as producer:
            assert producer.receive_json()["type"] == "error"
        assert calls == []


def test_root_serves_obs_overlay_without_loading_a_model():
    service = ModelService(lambda: pytest.fail("Overlay must not load model"))
    with TestClient(create_app(service)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "/ws/subtitles" in response.text
        assert service.model is None


def test_model_loading_is_background_idempotent_and_error_visible():
    release = threading.Event()
    calls = []
    model = FakeModel()
    def loader():
        calls.append(1)
        release.wait(3)
        return model
    service = ModelService(loader)
    try:
        with TestClient(create_app(service)) as client:
            assert client.post("/load-model").json()["status"] == "loading"
            assert client.post("/load-model").json()["status"] == "loading"
            assert client.get("/health").status_code == 200
            release.set()
            assert wait_status(client, "ready")["is_loaded"]
            assert client.post("/load-model").json()["is_loaded"]
            assert calls == [1]
    finally:
        release.set()
    def broken():
        raise RuntimeError("Download unavailable offline")
    service = ModelService(broken)
    with TestClient(create_app(service)) as client:
        client.post("/load-model")
        assert "offline" in wait_status(client, "error")["error"]
        assert client.get("/model-status").json()["is_downloading"] is False


def test_partial_replacement_final_and_observer_receive_same_events():
    service = ready_service()
    with TestClient(create_app(service)) as client:
        with client.websocket_connect("/ws/subtitles") as observer:
            observer.receive_json()
            with client.websocket_connect("/ws/transcribe?instant=true") as producer:
                assert producer.receive_json()["type"] == "ready"
                assert observer.receive_json()["type"] == "clear"
                send_sound(producer, 1.3)
                partial = producer.receive_json()
                assert partial["mode"] == "instant"
                assert observer.receive_json() == partial
                producer.send_bytes(sound(.6, 0))
                final = producer.receive_json()
                assert final["mode"] == "final"
                assert final["data"]["id"] == partial["data"]["id"]
                assert final["data"]["text"] != partial["data"]["text"]
                assert observer.receive_json() == final
                producer.send_json({"type": "stop"})
                assert producer.receive_json()["type"] == "stopped"
            with client.websocket_connect("/ws/subtitles") as reconnect:
                assert reconnect.receive_json()["type"] == "ready"
                snapshot = reconnect.receive_json()
                assert snapshot["data"] == final["data"]
                assert 0 < snapshot["expires_in_ms"] <= 8000


def test_stop_flushes_final_short_tail_and_releases_producer():
    app = create_app(ready_service())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.receive_json()
            ws.send_bytes(sound(.213))
            ws.send_json({"type": "stop"})
            result = ws.receive_json()
            assert result["mode"] == "final"
            assert result["metrics"]["audio_seconds"] == .213
            assert ws.receive_json()["type"] == "stopped"
        assert app.state.producer is None


def test_second_producer_rejected_but_passive_observers_allowed():
    with TestClient(create_app(ready_service())) as client:
        with client.websocket_connect("/ws/transcribe") as first:
            first.receive_json()
            with client.websocket_connect("/ws/transcribe") as second:
                assert "annan flik" in second.receive_json()["message"]
            with client.websocket_connect("/ws/subtitles") as observer:
                assert observer.receive_json()["type"] == "ready"
            first.send_json({"type": "stop"})
            assert first.receive_json()["type"] == "stopped"


@pytest.mark.parametrize("payload", [b"123", bytes(64004), np.array([np.nan], dtype="<f4").tobytes()], ids=["odd-bytes", "oversized", "nan"])
def test_malformed_packets_return_error_and_release_microphone(payload):
    app = create_app(ready_service())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.receive_json()
            ws.send_bytes(payload)
            assert ws.receive_json()["type"] == "error"
        assert app.state.producer is None


@pytest.mark.parametrize("control", ["broken json", "[]", '{"type":"unknown"}'])
def test_malformed_control_visible(control):
    with TestClient(create_app(ready_service())) as client:
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.receive_json()
            ws.send_text(control)
            assert ws.receive_json()["type"] == "error"


def test_model_inference_error_visible_and_ownership_released():
    class BrokenModel:
        def recognize(self, *_args, **_kwargs):
            raise RuntimeError("inference failed")
    app = create_app(ready_service(BrokenModel()))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.receive_json()
            send_sound(ws, 1.2)
            assert "inference failed" in ws.receive_json()["message"]
        assert app.state.producer is None


def test_overload_is_visible_while_capture_and_health_remain_responsive():
    entered, release = threading.Event(), threading.Event()
    class SlowModel:
        def recognize(self, *_args, **_kwargs):
            entered.set()
            release.wait(5)
            return "text"
    app = create_app(ready_service(SlowModel()))
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe") as ws:
                ws.receive_json()
                send_sound(ws, 1.2)
                assert entered.wait(2)
                assert client.get("/health").json()["status"] == "ok"
                for _ in range(5):
                    ws.send_bytes(sound(.2))
                    ws.send_bytes(sound(.6, 0))
                error = ws.receive_json()
                assert error["type"] == "error" and "hinner inte" in error["message"]
                release.set()
    finally:
        release.set()


def test_clear_during_inference_suppresses_stale_partial():
    entered, release = threading.Event(), threading.Event()
    class SlowFirstModel(FakeModel):
        def recognize(self, audio, sample_rate):
            if not self.calls:
                entered.set()
                release.wait(5)
            return super().recognize(audio, sample_rate)
    try:
        with TestClient(create_app(ready_service(SlowFirstModel()))) as client:
            with client.websocket_connect("/ws/transcribe") as ws:
                ws.receive_json()
                send_sound(ws, 1.2)
                assert entered.wait(2)
                ws.send_json({"type": "clear"})
                assert ws.receive_json()["type"] == "clear"
                release.set()
                ws.send_bytes(sound(.3))
                ws.send_json({"type": "stop"})
                final = ws.receive_json()
                assert final["mode"] == "final" and final["data"]["id"].endswith("-2")
                assert ws.receive_json()["type"] == "stopped"
    finally:
        release.set()


def test_caption_snapshots_expire_and_retain_remaining_lifetime(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("backend.main.time.monotonic", lambda: clock[0])
    hub = SubtitleHub()
    hub.publish({"type": "transcription", "mode": "final", "data": {"id": "x", "text": "Hej"}})
    clock[0] = 103
    assert hub.snapshot()[0]["expires_in_ms"] == 5000
    clock[0] = 109
    assert hub.snapshot() == []


def test_clear_removes_captions_waiting_in_slow_observer_queue():
    hub = SubtitleHub()
    queue = asyncio.Queue(maxsize=32)
    hub.listeners.add(queue)
    hub.publish({"type": "transcription", "mode": "instant", "data": {"id": "x", "text": "Old"}})
    hub.publish({"type": "clear"})
    assert queue.get_nowait() == {"type": "clear"}
    assert queue.empty()


def test_unknown_models_and_nonlocal_origins_rejected():
    with TestClient(create_app(ready_service())) as client:
        assert client.get("/model-status?model=whisper").status_code == 400
        assert client.post("/load-model", headers={"Origin": "https://evil.example"}).status_code == 403
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/transcribe", headers={"Origin": "https://evil.example"}):
                pass
        with client.websocket_connect("/ws/transcribe", headers={"Origin": "http://127.0.0.1:3000"}) as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "stop"})
            assert ws.receive_json()["type"] == "stopped"


def test_inference_deadline_errors_before_native_thread_finishes_and_never_publishes_late_text():
    entered, release = threading.Event(), threading.Event()
    class SlowFirstModel:
        calls = 0
        def recognize(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                release.wait(5)
                return "OLD result must never appear"
            return "NEW current result"
    model = SlowFirstModel()
    app = create_app(ready_service(model), max_inference_age=.05)
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/subtitles") as observer:
                assert observer.receive_json()["type"] == "ready"
                with client.websocket_connect("/ws/transcribe") as ws:
                    assert ws.receive_json()["type"] == "ready"
                    assert observer.receive_json()["type"] == "clear"
                    send_sound(ws, 1.2)
                    assert entered.wait(2)
                    error = ws.receive_json()
                    assert error["type"] == "error" and "tidsgränsen" in error["message"]
                    assert not release.is_set()  # Error arrives before native inference completes.
                    with pytest.raises(WebSocketDisconnect) as closed:
                        ws.receive_json()
                    assert closed.value.code == 1013
                    assert observer.receive_json()["type"] == "clear"
                    assert client.get("/health").json()["status"] == "ok"
                    status = client.get("/model-status").json()
                    assert status["is_busy"] and status["is_draining"]
                    assert app.state.hub.snapshot() == []
                    with client.websocket_connect("/ws/transcribe") as rejected:
                        assert "avslutar" in rejected.receive_json()["message"]
                    assert model.calls == 1
                    release.set()
                    deadline = time.monotonic() + 2
                    while client.get("/model-status").json()["is_busy"] and time.monotonic() < deadline:
                        time.sleep(.01)
                    assert client.get("/model-status").json()["is_busy"] is False
                    assert app.state.hub.snapshot() == []
                with client.websocket_connect("/ws/transcribe") as fresh:
                    assert fresh.receive_json()["type"] == "ready"
                    # Any late publication from the timed-out inference would appear here.
                    assert observer.receive_json()["type"] == "clear"
                    fresh.send_bytes(sound(.3))
                    fresh.send_json({"type": "stop"})
                    final = fresh.receive_json()
                    assert final["data"]["text"] == "NEW current result"
                    assert observer.receive_json() == final
                    assert fresh.receive_json()["type"] == "stopped"
    finally:
        release.set()


def test_expired_queued_job_is_rejected_before_inference(monkeypatch):
    from backend.audio import Segmenter
    class AlreadyDelayedSegmenter(Segmenter):
        def feed(self, audio):
            jobs = super().feed(audio)
            for job in jobs:
                job.created_at -= 10
            return jobs
    monkeypatch.setattr("backend.main.Segmenter", AlreadyDelayedSegmenter)
    model = FakeModel()
    with TestClient(create_app(ready_service(model))) as client:
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.receive_json()
            send_sound(ws, 1.2)
            error = ws.receive_json()
            assert error["type"] == "error" and "3 sekunder" in error["message"]
            assert model.calls == []


def test_inference_deadline_includes_time_already_spent_in_queue(monkeypatch):
    from backend.audio import Segmenter
    entered, release = threading.Event(), threading.Event()
    class QueueDelaySegmenter(Segmenter):
        def feed(self, audio):
            jobs = super().feed(audio)
            for job in jobs:
                job.created_at -= .5
            return jobs
    class BlockedModel:
        def recognize(self, *_args, **_kwargs):
            entered.set()
            release.wait(5)
            return "This result is late"
    monkeypatch.setattr("backend.main.Segmenter", QueueDelaySegmenter)
    try:
        with TestClient(create_app(ready_service(BlockedModel()), max_inference_age=.6)) as client:
            with client.websocket_connect("/ws/transcribe") as ws:
                ws.receive_json()
                started = time.monotonic()
                send_sound(ws, 1.2)
                assert entered.wait(2)
                assert "tidsgränsen" in ws.receive_json()["message"]
                # Only ~100ms remains; giving inference another full600ms would fail.
                assert time.monotonic() - started < .5
                release.set()
    finally:
        release.set()

"""Local Pianissimo subtitles. Importing this module never loads/downloads a model."""

import asyncio
from collections import OrderedDict
from contextlib import suppress
import json
import logging
import os
from pathlib import Path
import time

# Configure before lazily importing Hugging Face / ONNX Runtime.
PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("PIANISSIMO_CACHE_DIR", str(PROJECT_DIR / "models"))).resolve()
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
CPU_THREADS = max(1, min(16, int(os.environ.get("PIANISSIMO_THREADS", "4"))))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

try:
    from .audio import AudioProtocolError, JobBuffer, Overloaded, Segmenter, decode_audio, trim_overlap
except ImportError:  # python backend/main.py
    from audio import AudioProtocolError, JobBuffer, Overloaded, Segmenter, decode_audio, trim_overlap

logger = logging.getLogger(__name__)
MODEL_ID = "KlangAI/pianissimo-sv-onnx"
ORIGINS = {"http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000", "http://127.0.0.1:3000"}
CAPTION_TTL = 8.0
MAX_INFERENCE_AGE = 3.0


class ModelService:
    def __init__(self, loader=None):
        self.model = None
        self.status = "not_loaded"
        self.error = None
        self.load_task = None
        self.loader = loader or self._load

    @staticmethod
    def _load():
        import onnx_asr
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = CPU_THREADS
        session_options.inter_op_num_threads = 1
        return onnx_asr.load_model(
            MODEL_ID, quantization="int8", providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )

    def snapshot(self):
        return {
            "model": "pianissimo", "model_id": MODEL_ID, "is_loaded": self.model is not None,
            # Loading can download on first use; exact byte progress is unavailable.
            "is_downloading": self.status == "loading", "status": self.status,
            "error": self.error, "device": "cpu", "quantization": "int8",
            "cpu_threads": CPU_THREADS, "cache_dir": os.environ["HF_HUB_CACHE"],
            "offline": os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"},
        }

    def start_load(self):
        if self.model is not None or (self.load_task and not self.load_task.done()):
            return
        self.status, self.error = "loading", None
        self.load_task = asyncio.create_task(self._load_background())

    async def _load_background(self):
        try:
            loaded = await asyncio.to_thread(self.loader)
            self.model = loaded
            self.status = "ready"
        except Exception as exc:
            logger.exception("Pianissimo model loading failed")
            self.error = str(exc)[:500]
            self.status = "error"

    def recognize(self, audio):
        if self.model is None:
            raise RuntimeError("Pianissimo är inte laddad.")
        return self.model.recognize(audio, sample_rate=16000).strip()


class SubtitleHub:
    def __init__(self):
        self.listeners = set()
        self.latest = OrderedDict()

    def publish(self, event):
        if event["type"] == "clear":
            self.latest.clear()
        elif event["type"] == "transcription":
            self.latest[event["data"]["id"]] = (time.monotonic(), event)
            while len(self.latest) > 2:
                self.latest.popitem(last=False)
        for queue in tuple(self.listeners):
            if event["type"] == "clear":
                # A deadline/explicit clear must supersede captions still waiting
                # in a slow observer's queue, not sit behind obsolete results.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(event)
                continue
            # Slow OBS observers cannot delay capture. Reset and snapshot on overflow.
            if queue.full():
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({"type": "clear"})
                for current in self.snapshot():
                    queue.put_nowait(current)
            else:
                queue.put_nowait(event)

    def snapshot(self):
        now = time.monotonic()
        return [dict(event, expires_in_ms=max(0, int((CAPTION_TTL - (now - created)) * 1000)))
                for created, event in self.latest.values() if now - created < CAPTION_TTL]


def create_app(service=None, *, max_inference_age=MAX_INFERENCE_AGE):
    if max_inference_age <= 0:
        raise ValueError("max_inference_age must be positive")
    app = FastAPI(title="Local Pianissimo subtitles")
    app.state.service = service or ModelService()
    app.state.hub = SubtitleHub()
    app.state.producer = None
    app.state.draining = False
    app.add_middleware(CORSMiddleware, allow_origins=sorted(ORIGINS),
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def local_mutations(request: Request, call_next):
        if request.method == "POST" and request.headers.get("origin") not in ORIGINS | {None}:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Endast den lokala appen får ladda modellen."}, status_code=403)
        return await call_next(request)

    def validate_model(model):
        if model != "pianissimo":
            raise HTTPException(400, "Den här versionen stöder modellen pianissimo.")

    @app.get("/")
    async def overlay():
        return HTMLResponse(Path(__file__).with_name("overlay.html").read_text(encoding="utf-8"))

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "pianissimo", "sample_rate": 16000}

    @app.get("/model-status")
    async def model_status(model: str = "pianissimo"):
        validate_model(model)
        return dict(app.state.service.snapshot(), is_busy=app.state.producer is not None,
                    is_draining=app.state.draining, max_inference_age_seconds=max_inference_age)

    @app.post("/load-model")
    async def load_model(model: str = "pianissimo"):
        validate_model(model)
        app.state.service.start_load()
        return app.state.service.snapshot()

    async def accept_local(ws):
        if ws.headers.get("origin") not in ORIGINS | {None}:
            await ws.close(code=1008)
            return False
        await ws.accept()
        return True

    @app.websocket("/ws/subtitles")
    async def subtitles(ws: WebSocket):
        if not await accept_local(ws):
            return
        queue = asyncio.Queue(maxsize=32)
        app.state.hub.listeners.add(queue)
        try:
            await ws.send_json({"type": "ready", "role": "observer"})
            for event in app.state.hub.snapshot():
                await ws.send_json(event)

            async def sender():
                while True:
                    await ws.send_json(await queue.get())

            async def receiver():
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        return

            tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            app.state.hub.listeners.discard(queue)

    @app.websocket("/ws/transcribe")
    async def transcribe(ws: WebSocket, model: str = "pianissimo", vad: int = Query(3), instant: bool = True):
        if not await accept_local(ws):
            return

        async def fail(message, code=1008):
            with suppress(WebSocketDisconnect, RuntimeError):
                await ws.send_json({"type": "error", "message": message})
                await ws.close(code=code)

        if model != "pianissimo" or not app.state.service.snapshot()["is_loaded"]:
            await fail("Ladda Pianissimo innan du startar mikrofonen.")
            return
        if app.state.producer is not None:
            if app.state.draining:
                await fail("Modellen avslutar en för lång beräkning. Inspelningen är stoppad; vänta tills modellen blir ledig innan du startar igen.")
            else:
                await fail("Mikrofonen används redan av en annan flik eller ljudkälla. OBS ska använda http://localhost:8000.")
            return
        try:
            segmenter = Segmenter(vad, instant)
        except AudioProtocolError as exc:
            await fail(str(exc))
            return
        app.state.producer = ws
        app.state.draining = False
        jobs = JobBuffer()
        disconnected = False
        inference_task = None

        def latency_error():
            return Overloaded(f"Transkriberingen överskred tidsgränsen {max_inference_age:g} sekunder för kö och beräkning. Inspelningen har stoppats för att undvika fördröjda undertexter.")

        async def worker():
            nonlocal inference_task
            previous_final = ""
            while (job := await jobs.get()) is not None:
                remaining = job.created_at + max_inference_age - time.monotonic()
                if remaining <= 0:
                    raise latency_error()
                started = time.perf_counter()
                # Shield the native inference task: a timeout must notify the user
                # immediately, but Python cannot cancel ONNX's running native thread.
                # The endpoint retains ownership and awaits it only after error/close.
                inference_task = asyncio.create_task(asyncio.to_thread(app.state.service.recognize, job.audio))
                try:
                    text = await asyncio.wait_for(asyncio.shield(inference_task), timeout=remaining)
                except TimeoutError as exc:
                    raise latency_error() from exc
                if time.monotonic() - job.created_at >= max_inference_age:
                    raise latency_error()
                if disconnected or job.generation != segmenter.generation:
                    continue
                raw_text = text
                if job.continuation:
                    text = trim_overlap(previous_final, text)
                if job.final:
                    previous_final = raw_text
                event = {
                    "type": "transcription", "mode": "final" if job.final else "instant",
                    "data": {"id": job.id, "text": text, "start": round(job.start, 3), "end": round(job.end, 3)},
                    "metrics": {"inference_ms": round((time.perf_counter() - started) * 1000, 1),
                                "audio_seconds": round(len(job.audio) / 16000, 3)},
                    "expires_in_ms": int(CAPTION_TTL * 1000),
                }
                # Empty finals retract an earlier partial with the same ID.
                if text or job.final:
                    app.state.hub.publish(event)
                    await ws.send_json(event)

        async def receiver():
            nonlocal disconnected
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    disconnected = True
                    jobs.close(discard=True)
                    return "disconnect"
                if message.get("bytes") is not None:
                    for job in segmenter.feed(decode_audio(message["bytes"])):
                        jobs.submit(job)
                elif message.get("text") is not None:
                    try:
                        control = json.loads(message["text"])
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise AudioProtocolError("Ogiltigt kontrollmeddelande.") from exc
                    if not isinstance(control, dict):
                        raise AudioProtocolError("Kontrollmeddelandet måste vara ett JSON-objekt.")
                    if control.get("type") == "stop":
                        for job in segmenter.flush():
                            jobs.submit(job)
                        jobs.close()
                        return "stop"
                    if control.get("type") == "clear":
                        segmenter.clear()
                        jobs.clear()
                        event = {"type": "clear"}
                        app.state.hub.publish(event)
                        await ws.send_json(event)
                    else:
                        raise AudioProtocolError("Okänt kontrollmeddelande; använd stop eller clear.")

        receive_task = worker_task = None
        try:
            app.state.hub.publish({"type": "clear"})
            await ws.send_json({"type": "ready", "model": "pianissimo", "sample_rate": 16000})
            receive_task = asyncio.create_task(receiver())
            worker_task = asyncio.create_task(worker())
            done, _ = await asyncio.wait([receive_task, worker_task], return_when=asyncio.FIRST_COMPLETED)
            if worker_task in done:
                await worker_task
            result = await receive_task
            await worker_task
            if result == "stop":
                await ws.send_json({"type": "stopped"})
                await ws.close()
        except (AudioProtocolError, Overloaded) as exc:
            disconnected = True
            jobs.close(discard=True)
            if isinstance(exc, Overloaded):
                app.state.draining = inference_task is not None and not inference_task.done()
                app.state.hub.publish({"type": "clear"})
            await fail(str(exc), code=1003 if isinstance(exc, AudioProtocolError) else 1013)
        except WebSocketDisconnect:
            disconnected = True
            jobs.close(discard=True)
        except Exception as exc:
            disconnected = True
            jobs.close(discard=True)
            logger.exception("Transcription session failed")
            await fail(f"Transkriberingen misslyckades: {str(exc)[:300]}", code=1011)
        finally:
            jobs.close(discard=True)
            if receive_task and not receive_task.done():
                receive_task.cancel()
            if receive_task:
                await asyncio.gather(receive_task, return_exceptions=True)
            if worker_task:
                await asyncio.gather(worker_task, return_exceptions=True)
            # Never release the model while its native thread is still running.
            # Its discarded result cannot publish captions or start another job.
            if inference_task:
                await asyncio.gather(inference_task, return_exceptions=True)
            app.state.draining = False
            app.state.producer = None

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000, ws_max_size=64000)

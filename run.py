"""One-process launcher: native microphone → local Pianissimo → green OBS page."""

import argparse
import asyncio
from contextlib import contextmanager
import os
import signal
import sys
import webbrowser


def arguments():
    parser = argparse.ArgumentParser(description="Pianissimo: svensk mikrofontext till OBS.")
    parser.add_argument("--list-devices", action="store_true", help="Lista mikrofoner utan att ladda modellen.")
    parser.add_argument("--device", help="Mikrofonens nummer eller del av namnet (standard: systemets mikrofon).")
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 17), metavar="1–16", help="CPU-trådar, standard 4.")
    parser.add_argument("--sensitivity", type=int, default=4, choices=range(1, 6), metavar="1–5", help="Mikrofonkänslighet: 5 fångar svagare tal, 1 filtrerar mer. Standard 4.")
    parser.add_argument("--precision", choices=("int8", "fp16", "fp32"), default="int8", help="Modellprecision: int8 är snabbast; fp16/fp32 kräver större nedladdning.")
    parser.add_argument("--diagnostics", action="store_true", help="Visa ljudnivå och beräkningstid i cmd, utan att spara ljud eller text.")
    parser.add_argument("--no-microphone", action="store_true", help="Starta server och modell utan mikrofon (för test).")
    parser.add_argument("--no-browser", action="store_true", help="Öppna inte webbläsaren (om du bara använder OBS).")
    return parser.parse_args()


async def open_overlay():
    try:
        opened = await asyncio.to_thread(webbrowser.open, "http://localhost:8000", new=2)
    except Exception:
        opened = False
    if not opened:
        print("Öppna http://localhost:8000 i webbläsaren för att se texten.", flush=True)


async def run(args):
    import uvicorn
    from backend.main import ModelService, create_app
    from backend.native_audio import capture_microphone

    class LocalServer(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            # Our SIGINT handler flushes microphone audio before server shutdown.
            yield

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous_handlers = {}
    interrupt_count = 0

    def request_stop(*_args):
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count > 1:
            # A native model call/download cannot be safely cancelled by Python.
            # A second explicit interrupt terminates only this launcher process.
            os._exit(130)
        print("Avslutar ... Ctrl+C igen avbryter direkt.", flush=True)
        loop.call_soon_threadsafe(stop.set)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    service = ModelService()
    app = create_app(service)
    server = LocalServer(uvicorn.Config(app, host="127.0.0.1", port=8000,
                                       log_level="warning", ws_max_size=64000))
    server_task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("Servern kunde inte starta på port 8000.")
            await asyncio.sleep(.025)
        print("OBS-webbläsarkälla: http://localhost:8000", flush=True)
        print(f"Laddar Pianissimo ({args.precision}, CPU, {args.threads} trådar). Första starten hämtar modellen; därefter används lokal cache.", flush=True)
        service.start_load()
        await service.load_task
        if not service.model:
            raise RuntimeError(service.error or "Modellen kunde inte laddas.")
        print("Pianissimo är klar. Tryck Ctrl+C för att avsluta.", flush=True)
        if args.no_microphone:
            await stop.wait()
        elif not stop.is_set():
            if not args.no_browser:
                await open_overlay()
            if stop.is_set():
                return
            device = int(args.device) if args.device and args.device.isdecimal() else args.device
            try:
                await capture_microphone(device, stop, lambda: app.state.hub.publish({"type": "clear"}),
                                         sensitivity=args.sensitivity, diagnostics=args.diagnostics)
            except Exception as exc:
                app.state.hub.publish({"type": "clear"})
                print(f"\nInspelningen stoppades: {exc}", file=sys.stderr, flush=True)
                print("Ingen automatisk omstart. Tryck Ctrl+C och starta programmet igen när problemet är löst.", flush=True)
                await stop.wait()
    finally:
        server.should_exit = True
        await server_task
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    args = arguments()
    os.environ["PIANISSIMO_THREADS"] = str(args.threads)
    os.environ["PIANISSIMO_PRECISION"] = args.precision
    if args.list_devices:
        import sounddevice as sd
        default_input = sd.default.device[0]
        for index, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                marker = " (standard)" if index == default_input else ""
                print(f"{index}: {device['name']}{marker}")
        return 0
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Fel: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

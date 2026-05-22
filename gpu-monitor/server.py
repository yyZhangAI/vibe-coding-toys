import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from gpu_collector import GPUCollector, MockGPUCollector, load_config

MOCK = "--mock" in sys.argv


class MonitorState:
    def __init__(self):
        self.collectors: list = []
        self.clients: set[WebSocket] = set()
        self._task: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(max_workers=8)

    async def broadcast(self, data: dict):
        payload = json.dumps(data)
        stale = set()
        for ws in self.clients:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.add(ws)
        self.clients -= stale

    def _poll_all(self):
        results = list(self._executor.map(lambda c: c.poll(), self.collectors))
        all_gpus = []
        all_procs = []
        errors = []
        for r in results:
            if "error" in r:
                errors.append(f"{r['server']}: {r['error']}")
            all_gpus.extend(r.get("gpus", []))
            all_procs.extend(r.get("processes", []))
        return {
            "gpus": all_gpus,
            "processes": all_procs,
            "servers": [c.name for c in self.collectors],
            "connected": [c.name for c in self.collectors if c.connected],
            "errors": errors,
            "timestamp": time.time(),
        }

    async def poll_loop(self):
        while True:
            data = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._poll_all
            )
            await self.broadcast(data)
            await asyncio.sleep(2)


state = MonitorState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MOCK:
        print("[mock] using mock GPU data for 2 servers")
        state.collectors = [
            MockGPUCollector("server1"),
            MockGPUCollector("server2"),
        ]
    else:
        servers = load_config()
        for cfg in servers:
            c = GPUCollector(
                name=cfg["name"],
                host=cfg["host"],
                port=cfg["port"],
                user=cfg["user"],
                key_file=cfg["key_file"],
            )
            c.connect()
            state.collectors.append(c)
            print(f"[{c.name}] {'connected' if c.connected else 'failed to connect'}")

    state._task = asyncio.create_task(state.poll_loop())
    yield
    if state._task:
        state._task.cancel()
    for c in state.collectors:
        c.disconnect()
    state._executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse(
        "static/index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    state.clients.add(ws)
    data = await asyncio.get_event_loop().run_in_executor(
        state._executor, state._poll_all
    )
    await ws.send_text(json.dumps(data))
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "refresh":
                data = await asyncio.get_event_loop().run_in_executor(
                    state._executor, state._poll_all
                )
                await ws.send_text(json.dumps(data))
    except WebSocketDisconnect:
        pass
    finally:
        state.clients.discard(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
    print("Dashboard: http://127.0.0.1:8000")

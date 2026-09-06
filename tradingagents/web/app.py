from __future__ import annotations

import json
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .schemas import CreateRunRequest
from .service import RunManager

load_dotenv()

app = FastAPI(title="TradingAgents RL Console", version="0.1.0")
app.state.run_manager = RunManager()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
def create_run(payload: CreateRunRequest, request: Request) -> dict:
    try:
        return request.app.state.run_manager.create(payload).snapshot()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    try:
        return request.app.state.run_manager.get(run_id).snapshot()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/events")
def stream_events(run_id: str, request: Request) -> StreamingResponse:
    try:
        record = request.app.state.run_manager.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    last_event_id = request.headers.get("last-event-id")
    try:
        initial_cursor = int(last_event_id) if last_event_id is not None else -1
    except ValueError:
        initial_cursor = -1

    def generate():
        cursor = initial_cursor
        while True:
            events = record.wait_after(cursor)
            if not events:
                yield ": keep-alive\n\n"
            for event in events:
                cursor = event["id"]
                payload = json.dumps(event["data"], ensure_ascii=False)
                yield f"id: {cursor}\nevent: {event['type']}\ndata: {payload}\n\n"
            if record.status in {"completed", "failed"} and cursor == len(record.events) - 1:
                return

    return StreamingResponse(generate(), media_type="text/event-stream")


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "web" / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:

    @app.get("/")
    def frontend_missing() -> JSONResponse:
        return JSONResponse(
            {
                "message": "Frontend is not built. Run npm install && npm run build in web/frontend."
            }
        )


def main() -> None:
    uvicorn.run("tradingagents.web.app:app", host="127.0.0.1", port=8000)

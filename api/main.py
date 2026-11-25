"""FastAPI Production Server for EdgeLogic-AOI Industrial Inspection Workbench."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.engine_service import EdgeInspectionService

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="EdgeLogic-AOI Production Inspection API",
    description="Industrial Edge AOI Runtime with Camera Drift Gate, Intel OpenVINO NNCF/QAMR, Conformal 3-Way Triage, and PLC Interface.",
    version="1.0.0",
)

service = EdgeInspectionService()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_workbench() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Workbench UI not found.")
    return FileResponse(str(index_path))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready",
        "device": service.device,
        "plc_mode": service.plc.mode,
        "models_loaded": len(service.models_catalog),
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return {"models": service.list_models()}


@app.post("/api/inspect")
async def inspect_frame(
    file: UploadFile = File(...),
    model_id: str | None = Form(default=None),
    source: str = Form(default="Image"),
    simulate_blur: bool = Form(default=False),
) -> dict[str, Any]:
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image payload.")
    return service.inspect_bgr(
        img_bgr=img_bgr,
        model_id=model_id,
        source_name=source,
        simulate_blur=simulate_blur,
    )


@app.post("/api/conveyor/next")
def conveyor_next(
    model_id: str | None = Form(default=None),
    simulate_blur: bool = Form(default=False),
) -> dict[str, Any]:
    try:
        return service.inspect_next_conveyor(model_id=model_id, simulate_blur=simulate_blur)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/benchmarks")
def benchmarks() -> dict[str, Any]:
    return service.get_benchmarks()


@app.websocket("/ws/inspect")
async def ws_inspect_stream(websocket: WebSocket) -> None:
    """High-throughput binary WebSocket for live Video / Camera frame inspection."""
    await websocket.accept()
    params = websocket.query_params
    model_id = params.get("model_id") or None
    source = params.get("source") or "Live Stream"
    simulate_blur = params.get("simulate_blur", "false").lower() == "true"

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                # Allow live runtime config updates without reconnecting socket
                try:
                    import json
                    cfg = json.loads(msg["text"])
                    if "model_id" in cfg:
                        model_id = cfg["model_id"] or None
                    if "source" in cfg:
                        source = cfg["source"] or source
                    if "simulate_blur" in cfg:
                        simulate_blur = bool(cfg["simulate_blur"])
                except Exception:
                    pass
                continue

            raw = msg.get("bytes")
            if not raw:
                continue
            arr = np.frombuffer(raw, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            result = await asyncio.to_thread(
                service.inspect_bgr,
                img_bgr,
                model_id,
                source,
                simulate_blur,
                None,
                False,
                True,
            )
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return

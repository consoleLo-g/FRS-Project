# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as frs_router
from app.database.mongo_gallery import connect, init_db, close
from app.models.model_frs import fa
from app.core.config import settings
import asyncio
from typing import Set

app = FastAPI(title="FRS CCTV (MongoDB-backed)")

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")] if settings.CORS_ORIGINS else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple WebSocket alerts manager
class AlertsManager:
    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message: str):
        async with self._lock:
            clients = list(self._clients)
        for c in clients:
            try:
                await c.send_text(message)
            except Exception:
                await self.disconnect(c)

alerts_manager = AlertsManager()

@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await alerts_manager.connect(ws)
    try:
        while True:
            # keep connection alive, we don't expect client messages
            await ws.receive_text()
    except WebSocketDisconnect:
        await alerts_manager.disconnect(ws)
    except Exception:
        await alerts_manager.disconnect(ws)
        raise

@app.on_event("startup")
async def startup_event():
    await connect()
    try:
        await init_db()
    except Exception as e:
        print("Warning: init_db error:", e)
    print("✅ Startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    await close()
    print("✅ Shutdown complete.")

app.include_router(frs_router, prefix="/frs", tags=["Face Recognition"])

# import router for webrtc endpoint and include it
from app.utils.webrtc import router as webrtc_router  # noqa: E402
app.include_router(webrtc_router, prefix="/webrtc", tags=["webrtc"])

@app.get("/")
async def root():
    return {"message": "FRS CCTV service (MongoDB). Use /frs/* endpoints and /webrtc/offer for camera"}

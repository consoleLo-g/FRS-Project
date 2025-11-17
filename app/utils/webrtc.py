# app/webrtc.py
import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import APIRouter, Request, HTTPException
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
import av
import numpy as np
from app.utils.utils import detect_faces_from_bgr_np
from app.database.mongo_gallery import load_gallery_cache
from app.core.config import settings
from app.main import alerts_manager  # import broadcast manager

router = APIRouter(prefix="/webrtc", tags=["webrtc"])
logger = logging.getLogger("uvicorn")

# Keep references so RTCPeerConnections are not GC'd
pcs = set()

async def process_video_track(track, pc_id: str, camera_id: str = None):
    """
    Consume a video track, run detection at ~1 fps (throttled), and broadcast events.
    """
    last_process_ts = 0
    # throttle settings
    process_hz = 1.0  # process 1 frame per second by default
    min_interval = 1.0 / process_hz

    while True:
        try:
            frame = await track.recv()
            # frame is av.VideoFrame
            now = asyncio.get_event_loop().time()
            if now - last_process_ts < min_interval:
                continue
            last_process_ts = now

            # convert to numpy BGR
            img = frame.to_ndarray(format="bgr24")
            faces = await detect_faces_from_bgr_np(img)
            if not faces:
                continue

            # load gallery (cached)
            gallery = await load_gallery_cache()
            gallery_embs = [r["embedding"] for r in gallery if r.get("embedding") is not None]

            results = []
            for f in faces:
                q = f["embedding"]
                if q is None or len(gallery_embs) == 0:
                    continue
                sims = np.array([float(np.dot(q, emb)) for emb in gallery_embs])
                idxs = np.argsort(-sims)[:settings.TOP_K]
                candidates = []
                for idx in idxs:
                    candidates.append({
                        "identity_id": gallery[idx]["identity_id"],
                        "name": gallery[idx]["name"],
                        "score": float(sims[idx])
                    })
                results.append({"bbox": f["bbox"], "score": f["score"], "candidates": candidates})

            # Broadcast result to WebSocket alert clients
            if results:
                event = {
                    "camera_id": camera_id or pc_id,
                    "timestamp": asyncio.get_event_loop().time(),
                    "results": results
                }
                await alerts_manager.broadcast(json.dumps(event))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("Error processing track: %s", e)
            await asyncio.sleep(0.1)

@router.post("/offer")
async def offer(request: Request):
    """
    Accept an SDP offer from a publisher (Android/browser). Returns an SDP answer.
    Request JSON:
    {
      "sdp": "<offer sdp>",
      "type": "offer",
      "camera_id": "optional_camera_id"
    }
    """
    data = await request.json()
    if "sdp" not in data or "type" not in data:
        raise HTTPException(status_code=400, detail="Invalid SDP payload")
    camera_id = data.get("camera_id")

    pc = RTCPeerConnection()
    pcs.add(pc)
    pc_id = f"pc-{id(pc)}"
    logger.info("Created PeerConnection %s for camera=%s", pc_id, camera_id)

    # blackhole local media (we don't send anything back)
    media_blackhole = MediaBlackhole()

    @pc.on("track")
    def on_track(track):
        logger.info("Track %s received kind=%s", track, track.kind)
        if track.kind == "video":
            asyncio.ensure_future(process_video_track(track, pc_id, camera_id))
        # if audio tracks exist, consume to avoid backpressure
        if track.kind == "audio":
            asyncio.ensure_future(media_blackhole.consume(track))

    # set remote description
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    await pc.setRemoteDescription(offer)
    # create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Return the answer
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

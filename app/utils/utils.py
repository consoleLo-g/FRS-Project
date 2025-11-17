# app/utils.py
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any
from app.models.model_frs import fa
from app.core.config import settings
from starlette.concurrency import run_in_threadpool
import logging

logger = logging.getLogger("uvicorn")

def pil_to_bgr_np(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def bgr_np_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _detect_blocking(img_bgr: np.ndarray):
    try:
        return fa.get(img_bgr)
    except Exception as e:
        logger.exception("FaceAnalysis.get failed: %s", e)
        return []

async def detect_faces_pil(img_pil: Image.Image, max_faces: int = None) -> List[Dict[str, Any]]:
    img_bgr = pil_to_bgr_np(img_pil)
    faces = await run_in_threadpool(_detect_blocking, img_bgr)
    if max_faces:
        faces = faces[:max_faces]

    results = []
    for f in faces:
        bbox = [int(float(x)) for x in f.bbox]
        score = float(getattr(f, "det_score", getattr(f, "score", 0.0)))
        if score < settings.MIN_SCORE:
            continue
        x1,y1,x2,y2 = bbox
        w, h = max(0, x2-x1), max(0, y2-y1)
        if w < settings.MIN_FACE_SIZE or h < settings.MIN_FACE_SIZE:
            continue

        embedding = None
        if hasattr(f, "embedding") and f.embedding is not None:
            emb = np.array(f.embedding, dtype=np.float32)
            n = np.linalg.norm(emb)
            if n > 0:
                embedding = (emb / n).astype(np.float32)
        kps = f.kps.tolist() if hasattr(f, "kps") and f.kps is not None else None

        results.append({
            "bbox": bbox,
            "score": score,
            "kps": kps,
            "embedding": embedding
        })
    return results

async def detect_faces_from_bgr_np(img_bgr: np.ndarray, max_faces: int = None) -> List[Dict[str, Any]]:
    pil = bgr_np_to_pil(img_bgr)
    return await detect_faces_pil(pil, max_faces=max_faces)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))

def topk_bruteforce(query_emb: np.ndarray, gallery_embs: List[np.ndarray], top_k: int = 3):
    if query_emb is None or len(gallery_embs) == 0:
        return []
    embs = np.stack(gallery_embs)
    sims = np.dot(embs, query_emb)
    idxs = np.argsort(-sims)[:top_k]
    return [{"index": int(i), "score": float(sims[i])} for i in idxs]

# app/utils.py
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any
from app.models.model_frs import fa
from app.core.config import settings
from starlette.concurrency import run_in_threadpool

def pil_to_bgr_np(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB -> OpenCV BGR ndarray"""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _detect_blocking(img_bgr: np.ndarray):
    """
    Blocking call to insightface.FaceAnalysis.get
    Returns list of Face objects
    """
    return fa.get(img_bgr)

async def detect_faces_pil(img_pil: Image.Image) -> List[Dict[str, Any]]:
    """
    Async wrapper: returns list of dicts:
    {bbox: [x1,y1,x2,y2], score: float, kps: [[x,y],...], embedding: np.ndarray (L2-normalized)}
    """
    img_bgr = pil_to_bgr_np(img_pil)
    faces = await run_in_threadpool(_detect_blocking, img_bgr)

    results = []
    for f in faces:
        # bounding box & score
        bbox = [int(float(x)) for x in f.bbox]  # x1,y1,x2,y2
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

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))

def topk_bruteforce(query_emb: np.ndarray, gallery_embs: List[np.ndarray], top_k: int = 3):
    if query_emb is None or len(gallery_embs) == 0:
        return []
    embs = np.stack(gallery_embs)  # (N, D)
    sims = np.dot(embs, query_emb)  # (N,)
    idxs = np.argsort(-sims)[:top_k]
    return [{"index": int(i), "score": float(sims[i])} for i in idxs]

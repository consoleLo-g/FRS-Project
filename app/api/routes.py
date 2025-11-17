# app/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np
from typing import List
from app.utils.utils import detect_faces_pil, cosine_sim
from app.database.mongo_gallery import connect, close, init_db, add_identity, list_identities, load_gallery, load_gallery_cache
from app.core.config import settings

router = APIRouter(prefix="/frs", tags=["Face Recognition"])

@router.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    faces = await detect_faces_pil(img)
    boxes = [{"bbox": f["bbox"], "score": f["score"]} for f in faces]
    return {"num_faces": len(boxes), "faces": boxes}

@router.post("/compare")
async def compare_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...), threshold: float = settings.SIM_THRESHOLD):
    b1 = await file1.read(); b2 = await file2.read()
    img1 = Image.open(io.BytesIO(b1)).convert("RGB")
    img2 = Image.open(io.BytesIO(b2)).convert("RGB")
    faces1 = await detect_faces_pil(img1)
    faces2 = await detect_faces_pil(img2)
    if len(faces1) == 0 or len(faces2) == 0:
        raise HTTPException(status_code=400, detail="No faces detected in one or both images")
    results = []
    for i,f1 in enumerate(faces1):
        for j,f2 in enumerate(faces2):
            s = cosine_sim(f1["embedding"], f2["embedding"])
            results.append({"face1_index": i, "face2_index": j, "similarity": round(s, 4), "match": s >= threshold})
    best = max(results, key=lambda x: x["similarity"])
    return {"num_faces_image1": len(faces1), "num_faces_image2": len(faces2), "threshold": threshold, "results": results, "best_match": best}

@router.post("/add_identity")
async def add_identity_endpoint(name: str, file: UploadFile = File(...), metadata: str = None):
    if not name:
        raise HTTPException(status_code=400, detail="Provide name query param")
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    faces = await detect_faces_pil(img)
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected")
    embs = [f["embedding"] for f in faces if f["embedding"] is not None]
    if len(embs) == 0:
        raise HTTPException(status_code=400, detail="Embeddings not available")
    inserted_id = await add_identity(name, embs, metadata)
    return {"identity_id": inserted_id, "stored_embeddings": len(embs)}

@router.get("/list_identities")
async def list_identities_endpoint():
    ids = await list_identities()
    return {"count": len(ids), "identities": ids}

@router.post("/recognize")
async def recognize_endpoint(file: UploadFile = File(...), top_k: int = settings.TOP_K):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    faces = await detect_faces_pil(img)
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No faces detected")
    gallery = await load_gallery_cache()
    if len(gallery) == 0:
        raise HTTPException(status_code=400, detail="Gallery empty. Add identities first.")
    gallery_embs = [r["embedding"] for r in gallery if r.get("embedding") is not None]
    if len(gallery_embs) == 0:
        raise HTTPException(status_code=500, detail="Gallery embeddings unavailable.")
    out = []
    for i, f in enumerate(faces):
        q = f["embedding"]
        sims = np.array([float(np.dot(q, emb)) for emb in gallery_embs])
        idxs = np.argsort(-sims)[:top_k]
        candidates = []
        for idx in idxs:
            candidates.append({
                "identity_id": gallery[idx]["identity_id"],
                "name": gallery[idx]["name"],
                "score": float(sims[idx])
            })
        out.append({"face_index": i, "bbox": f["bbox"], "candidates": candidates})
    return {"num_faces": len(faces), "results": out}

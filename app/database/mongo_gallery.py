# app/database/mongo_gallery.py
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import numpy as np
from typing import List, Dict, Any
import asyncio

client: AsyncIOMotorClient = None
db = None

_gallery_cache: List[Dict[str, Any]] = []
_gallery_lock = asyncio.Lock()

def get_collection():
    global client, db
    if client is None:
        raise RuntimeError("Mongo client not initialized. Call connect() first.")
    return db[settings.MONGO_COLLECTION_IDENTITIES]

async def connect(uri: str = None):
    global client, db
    uri = uri or settings.MONGODB_URL
    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    db = client[settings.MONGO_DB_NAME]

async def close():
    global client
    if client:
        client.close()

async def init_db():
    coll = get_collection()
    await coll.create_index("name")
    await refresh_gallery_cache()

async def add_identity(name: str, embeddings: List[np.ndarray], metadata: Dict[str, Any] = None) -> str:
    coll = get_collection()
    emb_list = [emb.astype(float).tolist() for emb in embeddings]
    doc = {"name": name, "embeddings": emb_list, "metadata": metadata or {}}
    res = await coll.insert_one(doc)
    await refresh_gallery_cache()
    return str(res.inserted_id)

async def list_identities() -> List[Dict[str, Any]]:
    coll = get_collection()
    cursor = coll.find({}, {"name": 1})
    out = []
    async for doc in cursor:
        out.append({"id": str(doc["_id"]), "name": doc["name"]})
    return out

async def load_gallery() -> List[Dict[str, Any]]:
    coll = get_collection()
    cursor = coll.find({})
    records = []
    async for doc in cursor:
        _id = str(doc["_id"])
        name = doc.get("name")
        for emb in doc.get("embeddings", []):
            arr = np.array(emb, dtype=np.float32)
            n = np.linalg.norm(arr)
            if n > 0:
                arr = arr / n
            records.append({"identity_id": _id, "name": name, "embedding": arr})
    return records

async def refresh_gallery_cache():
    global _gallery_cache
    coll = get_collection()
    cursor = coll.find({})
    tmp = []
    async for doc in cursor:
        _id = str(doc["_id"])
        name = doc.get("name")
        for emb in doc.get("embeddings", []):
            arr = np.array(emb, dtype=np.float32)
            n = np.linalg.norm(arr)
            if n > 0:
                arr = arr / n
            tmp.append({"identity_id": _id, "name": name, "embedding": arr})
    async with _gallery_lock:
        _gallery_cache = tmp

async def load_gallery_cache() -> List[Dict[str, Any]]:
    global _gallery_cache
    if _gallery_cache:
        return _gallery_cache
    await refresh_gallery_cache()
    return _gallery_cache

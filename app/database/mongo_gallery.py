# app/mongo_gallery.py
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import numpy as np
from typing import List, Dict, Any

client: AsyncIOMotorClient = None
db = None

def get_collection():
    global client, db
    if client is None:
        raise RuntimeError("Mongo client not initialized. Call connect() first.")
    return db[settings.MONGO_COLLECTION_IDENTITIES]

async def connect(uri: str = None):
    global client, db
    uri = uri or settings.MONGODB_URL
    client = AsyncIOMotorClient(uri)
    db = client[settings.MONGO_DB_NAME]

async def close():
    global client
    if client:
        client.close()

async def init_db():
    coll = get_collection()
    # ensure indexes (e.g., on name)
    await coll.create_index("name")

async def add_identity(name: str, embeddings: List[np.ndarray], metadata: Dict[str, Any] = None) -> str:
    """
    Add identity document. Stores multiple embeddings for the same identity.
    Document shape:
    {
      name: str,
      embeddings: [ [float,..], [float,..], ... ],
      metadata: {...}
    }
    Returns inserted_id (str)
    """
    coll = get_collection()
    emb_list = [emb.astype(float).tolist() for emb in embeddings]
    doc = {"name": name, "embeddings": emb_list, "metadata": metadata or {}}
    res = await coll.insert_one(doc)
    return str(res.inserted_id)

async def list_identities() -> List[Dict[str, Any]]:
    coll = get_collection()
    cursor = coll.find({}, {"name": 1})
    out = []
    async for doc in cursor:
        out.append({"id": str(doc["_id"]), "name": doc["name"]})
    return out

async def load_gallery() -> List[Dict[str, Any]]:
    """
    Load gallery as list of records:
    { _id (str), name, embedding (np.ndarray) } 
    We return one record per stored embedding (so multiple records might share same identity id).
    """
    coll = get_collection()
    cursor = coll.find({})
    records = []
    async for doc in cursor:
        _id = str(doc["_id"])
        name = doc.get("name")
        for emb in doc.get("embeddings", []):
            arr = np.array(emb, dtype=np.float32)
            # ensure normalized
            n = np.linalg.norm(arr)
            if n > 0:
                arr = arr / n
            records.append({"identity_id": _id, "name": name, "embedding": arr})
    return records

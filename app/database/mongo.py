from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

mongo = MongoDB()

async def connect_to_mongo():
    mongo.client = AsyncIOMotorClient(settings.MONGODB_URL)
    mongo.db = mongo.client[settings.MONGO_DB_NAME]
    print("✅ Connected to MongoDB")

async def close_mongo_connection():
    mongo.client.close()
    print("🛑 MongoDB connection closed")

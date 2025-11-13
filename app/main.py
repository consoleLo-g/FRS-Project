from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database import mongo
from app.api.routes_frs import router as frs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    await mongo.connect_to_mongo()
    print("🚀 App starting... MongoDB connected.")

    collections = await mongo.db.list_collection_names()
    print(f"📂 Available collections: {collections}")

    yield  # <-- app runs while here

    # --- Shutdown ---
    await mongo.close_mongo_connection()
    print("🛑 MongoDB connection closed.")


# Initialize FastAPI with lifespan handler
app = FastAPI(title="Facial Recognition System", lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Facial Recognition API 🚀"}


# Register routes
app.include_router(frs_router, prefix="/frs", tags=["Face Recognition"])

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database.mongo import connect_to_mongo, close_mongo_connection
from app.api.routes_frs import router as frs_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    print("🚀 App starting... MongoDB connected.")
    yield
    await close_mongo_connection()

app = FastAPI(title="Facial Recognition System", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Facial Recognition API 🚀"}

app.include_router(frs_router, prefix="/frs", tags=["Face Recognition"])

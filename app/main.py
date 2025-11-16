# app/main.py
from fastapi import FastAPI
from app.api.routes import router as frs_router
from app.database.mongo_gallery import connect, init_db, close
from app.models.model_frs import fa  # keep import so model is loaded at startup

app = FastAPI(title="FRS CCTV (MongoDB-backed)")

@app.on_event("startup")
async def startup_event():
    # connect to MongoDB then ensure indexes
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

@app.get("/")
async def root():
    return {"message": "FRS CCTV service (MongoDB). Use /frs/* endpoints."}

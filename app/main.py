from fastapi import FastAPI
from app.api.routes_frs import router as frs_router

app = FastAPI(title="Facial Recognition System")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Facial Recognition API 🚀"}

app.include_router(frs_router, prefix="/frs", tags=["Face Recognition"])

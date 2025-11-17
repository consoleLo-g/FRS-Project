# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # InsightFace detector name
    DETECTOR_NAME: str = "buffalo_l"
    DET_SIZE: int = 640

    # Face quality thresholds
    MIN_FACE_SIZE: int = 40
    MIN_SCORE: float = 0.6

    # Recognition params
    TOP_K: int = 3
    SIM_THRESHOLD: float = 0.35

    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "frs_db"
    MONGO_COLLECTION_IDENTITIES: str = "identities"

    # Threadpool guidance (not strictly enforced)
    THREADPOOL_WORKERS: int = 4

    # InsightFace providers: e.g. "CUDAExecutionProvider,CPUExecutionProvider" or "CPUExecutionProvider"
    INSIGHTFACE_PROVIDERS: str = "CPUExecutionProvider"
    INSIGHTFACE_CTX_ID: int = -1

    # CORS origins (comma-separated) - set specific origins in production
    CORS_ORIGINS: str = "*"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

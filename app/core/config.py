# app/config.py
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
    SIM_THRESHOLD: float = 0.35  # ArcFace CCTV start point - tune on your val set
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "frs_db"
    MONGO_COLLECTION_IDENTITIES: str = "identities"
    # Threadpool safety: number of workers (optional, tune per CPU)
    THREADPOOL_WORKERS: int = 4

settings = Settings()

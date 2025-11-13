import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present

class Settings(BaseModel):
    PROJECT_NAME: str = "Facial Recognition System"
    MONGODB_URL: str = Field(default=os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    MONGO_DB_NAME: str = Field(default=os.getenv("MONGO_DB_NAME", "frs_db"))

settings = Settings()

import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Facial Recognition System"

    # MongoDB Connection String
    # Example: mongodb://localhost:27017 or mongodb+srv://<user>:<password>@cluster.mongodb.net/<dbname>
    MONGODB_URL: str = os.getenv(
        "MONGODB_URL", 
        "mongodb://localhost:27017"
    )

    # Database Name
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "frs_db")

settings = Settings()

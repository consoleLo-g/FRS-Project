# app/models_frs.py
import numpy as np
from insightface.app import FaceAnalysis
from app.core.config import settings

print("➡️ Initializing FaceAnalysis. Models may be downloaded on first run (allow internet).")
# FaceAnalysis defaults to good detector+recognizer combos for names like 'buffalo_l'
fa = FaceAnalysis(name=settings.DETECTOR_NAME, providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=0, det_size=(settings.DET_SIZE, settings.DET_SIZE))
print("✅ FaceAnalysis ready.")

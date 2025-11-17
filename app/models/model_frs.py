# app/models_frs.py
import logging
from insightface.app import FaceAnalysis
from app.core.config import settings

logger = logging.getLogger("uvicorn")

logger.info("➡️ Initializing FaceAnalysis. Models may be downloaded on first run (allow internet).")

_providers = [p.strip() for p in settings.INSIGHTFACE_PROVIDERS.split(",") if p.strip()]
if not _providers:
    _providers = ["CPUExecutionProvider"]

_ctx_id = settings.INSIGHTFACE_CTX_ID

try:
    fa = FaceAnalysis(name=settings.DETECTOR_NAME, providers=_providers)
    fa.prepare(ctx_id=_ctx_id, det_size=(settings.DET_SIZE, settings.DET_SIZE))
    logger.info("✅ FaceAnalysis ready. Providers=%s ctx_id=%s", _providers, _ctx_id)
except Exception as e:
    logger.exception("❌ Failed to initialize FaceAnalysis: %s", e)
    raise

from fastapi import APIRouter, UploadFile, File
from PIL import Image
from io import BytesIO
from app.utils.image_utils import compare_faces

router = APIRouter(prefix="/frs", tags=["Face Recognition"])

@router.post("/compare-faces/")
async def compare_faces_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = Image.open(BytesIO(await file1.read())).convert("RGB")
    img2 = Image.open(BytesIO(await file2.read())).convert("RGB")
    result = compare_faces(img1, img2)
    return result

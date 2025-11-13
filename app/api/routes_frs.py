from fastapi import APIRouter, UploadFile, File
from app.models.frs_model import compare_faces

router = APIRouter()

@router.post("/compare-faces/")
async def compare_faces_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1_bytes = await file1.read()
    img2_bytes = await file2.read()

    result = compare_faces(img1_bytes, img2_bytes)
    return result

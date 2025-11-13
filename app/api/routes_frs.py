from fastapi import APIRouter, UploadFile, File
from pathlib import Path
from app.utils.image_utils import save_upload_file
from app.models.frs_model import compare_faces

router = APIRouter()

@router.post("/compare-faces/")
async def compare_faces_api(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1_path = save_upload_file(file1, Path(f"app/data/uploads/{file1.filename}"))
    img2_path = save_upload_file(file2, Path(f"app/data/uploads/{file2.filename}"))

    score = compare_faces(str(img1_path), str(img2_path))
    return {"match_score": score}

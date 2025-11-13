from fastapi import UploadFile
from PIL import Image
import io
import traceback
from app.utils.image_utils import extract_faces, get_embedding, cosine_similarity

async def compare_faces(file1: UploadFile, file2: UploadFile):
    """Compare faces between two images (single or group)."""
    try:
        # Read both images
        img1_bytes = await file1.read()
        img2_bytes = await file2.read()

        img1 = Image.open(io.BytesIO(img1_bytes)).convert("RGB")
        img2 = Image.open(io.BytesIO(img2_bytes)).convert("RGB")

        # Extract faces
        faces1 = extract_faces(img1)
        faces2 = extract_faces(img2)

        if not faces1 or not faces2:
            return {"error": "No faces detected in one or both images."}

        # Get embeddings
        embeddings1 = [get_embedding(face) for face in faces1]
        embeddings2 = [get_embedding(face) for face in faces2]

        # Compare every face in img1 with every face in img2
        results = []
        for i, emb1 in enumerate(embeddings1):
            for j, emb2 in enumerate(embeddings2):
                sim = cosine_similarity(emb1, emb2)
                results.append({
                    "face1_index": i,
                    "face2_index": j,
                    "similarity": round(sim, 3),
                    "match": sim > 0.6
                })

        return {
            "num_faces_image1": len(faces1),
            "num_faces_image2": len(faces2),
            "results": results
        }

    except Exception as e:
        print("🔥 ERROR in compare_faces():", e)
        print(traceback.format_exc())
        return {"error": str(e)}

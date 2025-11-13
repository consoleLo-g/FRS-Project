from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# ============================
# Initialization (global)
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector & aligner
mtcnn = MTCNN(image_size=160, margin=0, post_process=True, device=device)

# Pretrained face embedding model
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# ============================
# Helper Functions
# ============================
def extract_faces(img: Image.Image):
    """
    Detect all faces in an image and return aligned face tensors.
    Returns: list[torch.Tensor] each of shape [3, 160, 160]
    """
    faces = mtcnn(img, return_prob=False)
    if faces is None:
        return []
    
    if isinstance(faces, torch.Tensor):
        faces = [faces]

    # Normalize to correct shape
    cleaned_faces = []
    for f in faces:
        if f.dim() == 4 and f.shape[0] == 1:
            f = f.squeeze(0)
        cleaned_faces.append(f)
    
    return cleaned_faces


def get_embedding(face_tensor: torch.Tensor):
    """
    Generate 512-D embedding vector for a single face tensor.
    """
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(face_tensor)
    return embedding.squeeze(0).cpu().numpy()


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two 512-D embeddings.
    Range: [-1, 1], where 1 means identical faces.
    """
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


# ============================
# Main Function
# ============================
def compare_faces(img1: Image.Image, img2: Image.Image, threshold=0.6):
    """
    Compare *all* faces in img1 vs *all* faces in img2.
    Returns similarity results for each pair and best matches.
    """
    faces1 = extract_faces(img1)
    faces2 = extract_faces(img2)

    if not faces1:
        return {"error": "No face detected in image1"}
    if not faces2:
        return {"error": "No face detected in image2"}

    embeddings1 = [get_embedding(f) for f in faces1]
    embeddings2 = [get_embedding(f) for f in faces2]

    results = []
    for i, emb1 in enumerate(embeddings1):
        comparisons = []
        for j, emb2 in enumerate(embeddings2):
            sim = cosine_similarity(emb1, emb2)
            comparisons.append({
                "face1_index": i,
                "face2_index": j,
                "similarity": round(sim, 3),
                "match": sim > threshold
            })
        # Find the best match for this face in image1
        best = max(comparisons, key=lambda x: x["similarity"])
        results.extend(comparisons)
    
    # Find the overall best match across all comparisons
    best_match = max(results, key=lambda x: x["similarity"])

    return {
        "num_faces_image1": len(faces1),
        "num_faces_image2": len(faces2),
        "threshold": threshold,
        "results": results,
        "best_match": best_match
    }


# ============================
# Example Usage (manual test)
# ============================
if __name__ == "__main__":
    img1 = Image.open("BillGates.jpg")
    img2 = Image.open("BillGatesFamily.webp")

    result = compare_faces(img1, img2)
    print(result)

import torch
import numpy as np
from PIL import Image
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN

# ============================
# Initialization
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detect multiple faces
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# ============================
# Helpers
# ============================
def preprocess_image(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def extract_faces(image_bytes):
    """
    Detect all faces and return aligned face tensors.
    """
    img = preprocess_image(image_bytes)
    faces = mtcnn(img)

    if faces is None:
        return []

    # Ensure we return list of faces
    if isinstance(faces, torch.Tensor) and faces.ndim == 3:
        faces = [faces]

    return faces


def get_embedding(face_tensor: torch.Tensor):
    """
    Generate 512-D embedding vector for one face.
    """
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(face_tensor)
    return embedding.squeeze(0).cpu().numpy()


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two 512-D vectors.
    """
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


# ============================
# Main Compare Function
# ============================
def compare_faces(img1_bytes, img2_bytes, threshold=0.6):
    """
    Compare all faces in image1 vs all faces in image2.
    Returns all similarity results and best match.
    """
    faces1 = extract_faces(img1_bytes)
    faces2 = extract_faces(img2_bytes)

    if not faces1:
        return {"error": "No face detected in image1", "match": False}
    if not faces2:
        return {"error": "No face detected in image2", "match": False}

    embeddings1 = [get_embedding(f) for f in faces1]
    embeddings2 = [get_embedding(f) for f in faces2]

    results = []
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            sim = cosine_similarity(emb1, emb2)
            results.append({
                "face1_index": i,
                "face2_index": j,
                "similarity": round(sim, 3),
                "match": sim >= threshold
            })

    best_match = max(results, key=lambda x: x["similarity"])

    return {
        "num_faces_image1": len(faces1),
        "num_faces_image2": len(faces2),
        "threshold": threshold,
        "results": results,
        "best_match": best_match
    }


# ============================
# Manual Test
# ============================
if __name__ == "__main__":
    with open("BillGates.jpg", "rb") as f1, open("BillGatesFamily.webp", "rb") as f2:
        img1_bytes = f1.read()
        img2_bytes = f2.read()

    result = compare_faces(img1_bytes, img2_bytes)
    print(result)

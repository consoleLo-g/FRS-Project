from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Initialize once (global)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MTCNN detects faces and crops them to 160x160
mtcnn = MTCNN(image_size=160, margin=0, post_process=True, device=device)

# Face embedding model (pretrained on VGGFace2)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def extract_faces(img: Image.Image):
    """
    Detect all faces in an image and return aligned face tensors.
    Returns: list of torch tensors (each [3, 160, 160])
    """
    faces = mtcnn(img, return_prob=False)
    if faces is None:
        return []
    
    # If only one face detected, MTCNN returns a tensor not a list
    if isinstance(faces, torch.Tensor):
        faces = [faces]

    cleaned_faces = []
    for f in faces:
        # Ensure tensor shape is [3,160,160]
        if f.dim() == 4 and f.shape[0] == 1:
            f = f.squeeze(0)
        cleaned_faces.append(f)
    
    return cleaned_faces


def get_embedding(face_tensor: torch.Tensor):
    """
    Generate a 512-D embedding vector for a single face tensor.
    Input shape: [3, 160, 160]
    """
    face_tensor = face_tensor.unsqueeze(0).to(device)  # [1, 3, 160, 160]

    with torch.no_grad():
        embedding = resnet(face_tensor)  # [1, 512]

    return embedding.squeeze(0).cpu().numpy()  # → (512,)


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two 512-D embeddings.
    Range: [-1, 1], where 1 = identical faces.
    """
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def compare_faces(img1: Image.Image, img2: Image.Image, threshold=0.6):
    """
    Compare all faces in img2 against the first face in img1.
    Returns similarity scores and best match info.
    """
    faces1 = extract_faces(img1)
    faces2 = extract_faces(img2)

    if not faces1:
        return {"error": "No face detected in image1"}
    if not faces2:
        return {"error": "No face detected in image2"}

    emb1 = get_embedding(faces1[0])
    results = []

    for i, f2 in enumerate(faces2):
        emb2 = get_embedding(f2)
        sim = cosine_similarity(emb1, emb2)
        results.append({
            "face2_index": i,
            "similarity": round(sim, 3),
            "match": sim > threshold
        })

    best_match = max(results, key=lambda x: x["similarity"])
    return {
        "num_faces_image1": len(faces1),
        "num_faces_image2": len(faces2),
        "results": results,
        "best_match": best_match
    }

import torch
import numpy as np
from PIL import Image
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def preprocess_image(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def get_face_embedding(image_bytes):
    img = preprocess_image(image_bytes)
    face = mtcnn(img)

    if face is None:
        raise ValueError("No face detected in the image")

    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device))
    return embedding.cpu().numpy()

def compare_faces(img1_bytes, img2_bytes, threshold=0.8):
    try:
        emb1 = get_face_embedding(img1_bytes)
        emb2 = get_face_embedding(img2_bytes)
    except ValueError as e:
        return {"error": str(e), "match": False, "similarity": None}

    eps = 1e-10
    sim = np.dot(emb1, emb2.T) / ((np.linalg.norm(emb1) + eps) * (np.linalg.norm(emb2) + eps))
    sim = float(sim)

    return {
        "similarity": round(sim, 3),
        "match": sim >= threshold
    }

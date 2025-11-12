# 🧠 Face Recognition System (FRS)

An end-to-end **Human Face Detection & Recognition System (FRS)** built with **FastAPI**, designed to detect faces in images (CCTV frames or photos), extract embeddings, and recognize identities from a gallery database.  
This system is modular, production-ready, and optimized for CPU inference.

---

## 🚀 Features

- 🔍 **Face Detection** – Detects faces in an image using RetinaFace or MTCNN.  
- 🧬 **Face Embeddings** – Generates unique embeddings using a pretrained model (ArcFace / AdaFace / FaceNet).  
- 🗃️ **Identity Recognition** – Matches detected faces against a stored gallery using cosine similarity.  
- ⚙️ **FastAPI Microservice** – Provides RESTful endpoints (`/detect`, `/recognize`, `/add_identity`, `/list_identities`).  
- 🐳 **Docker Support** – Containerized for easy deployment.  
- ⚡ **CPU Optimized** – Inference-ready for systems without dedicated GPUs.  

---

## 🧩 Project Structure
```
FRS-Project/
│
├── app/
│   ├── api/          # FastAPI route handlers
│   ├── core/         # Configuration, settings
│   ├── models/       # ML model loading & inference
│   ├── utils/        # Helper functions (alignment, embedding)
│   ├── data/         # Sample images / datasets
│   ├── database/     # SQLite or Postgres integration
│   └── main.py       # FastAPI entry point
│
├── requirements.txt  # Python dependencies
├── .gitignore        # Files to ignore in Git
├── README.md         # Project documentation
└── venv/             # Local virtual environment (excluded from Git)

```
---

## 🧰 Tech Stack

- **Language:** Python 3.12  
- **Framework:** FastAPI  
- **Models:** RetinaFace / ArcFace / AdaFace (PyTorch or ONNX)  
- **Database:** SQLite (can extend to PostgreSQL)  
- **Containerization:** Docker  
- **Utilities:** OpenCV, NumPy, Faiss (for similarity search)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/FRS-Project.git
cd FRS-Project
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
(If requirements.txt is empty, it will be filled later when dependencies are added.)

### 4️⃣ Run FastAPI Server
```bash
uvicorn app.main:app --reload
```

### 5️⃣ Access API
Base URL → http://127.0.0.1:8000

Docs (Swagger UI) → http://127.0.0.1:8000/docs

Redoc UI → http://127.0.0.1:8000/redoc

## 🧠 API Endpoints Overview

| Endpoint           | Method | Description                        |
| ------------------ | ------ | ---------------------------------- |
| `/`                | GET    | Root health check                  |
| `/detect`          | POST   | Detect faces in uploaded image     |
| `/recognize`       | POST   | Recognize face and return identity |
| `/add_identity`    | POST   | Add a new identity to the gallery  |
| `/list_identities` | GET    | List all stored identities         |


## 🧮 Future Enhancements

- 🧱 Integrate ONNX Runtime for faster inference

- 🧠 Implement Faiss for scalable vector search

- 🧾 Add database schema for gallery management

- 📦 Build and deploy using Docker

- 📊 Add performance evaluation metrics (precision, recall, latency)

## 🧑‍💻 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to change.

## 📜 License

This project is for academic and learning purposes under an open license (MIT / educational use).

## 👤 Author

Gaurav Kumar
📧 gkgaurav343@gmail.com
🚀 Built as part of the Human Face Recognition Assignment

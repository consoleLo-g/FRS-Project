from fastapi import FastAPI

app = FastAPI(title="Face Recognition System (FRS)")

@app.get("/")
def root():
    return {"message": "Welcome to the Face Recognition Service 🚀"}

@app.post("/detect")
def detect_faces():
    return {"status": "Face detection endpoint ready"}

@app.post("/recognize")
def recognize_faces():
    return {"status": "Face recognition endpoint ready"}

@app.post("/add_identity")
def add_identity():
    return {"status": "Add identity endpoint ready"}

@app.get("/list_identities")
def list_identities():
    return {"status": "List identities endpoint ready"}

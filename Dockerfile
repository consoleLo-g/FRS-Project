# ---------- Base Image ----------
FROM python:3.12-slim

# ---------- Set Work Directory ----------
WORKDIR /app

# ---------- Environment Variables ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------- Install System Dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy Project Files ----------
COPY requirements.txt .

# ---------- Install Python Dependencies ----------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Copy Application Code ----------
COPY . .

# ---------- Expose Port ----------
EXPOSE 8000

# ---------- Run FastAPI App ----------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

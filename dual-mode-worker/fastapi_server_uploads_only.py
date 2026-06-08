from fastapi import FastAPI, UploadFile, File
import os
import shutil
from datetime import datetime

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_path = os.path.join(
        UPLOAD_DIR,
        f"{timestamp}_{image.filename}"
    )

    audio_path = os.path.join(
        UPLOAD_DIR,
        f"{timestamp}_{audio.filename}"
    )

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    return {
        "image_saved": image_path,
        "audio_saved": audio_path
    }

from fastapi import FastAPI, UploadFile, File
from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_infer_params,
    get_audio_embedding,
    run_pipeline,
)

import os
import shutil
from datetime import datetime
import torch
import numpy as np
import librosa
import time
import subprocess
import imageio
from collections import deque

app = FastAPI()

pipeline = None

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_video(frames_list, video_path, audio_path, fps):
    temp_video_path = video_path.replace(".mp4", "_tmp.mp4")

    with imageio.get_writer(
        temp_video_path,
        format="mp4",
        mode="I",
        fps=fps,
        codec="h264",
        ffmpeg_params=["-bf", "0"]
    ) as writer:

        for frames in frames_list:
            frames = frames.numpy().astype(np.uint8)

            for i in range(frames.shape[0]):
                writer.append_data(frames[i])

    cmd = [
        "ffmpeg",
        "-i", temp_video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        video_path,
        "-y"
    ]

    subprocess.run(cmd)

    os.remove(temp_video_path)


@app.on_event("startup")
def startup_event():
    global pipeline

    print("Loading SoulX pipeline...")

    pipeline = get_pipeline(
        world_size=1,
        ckpt_dir="models/SoulX-FlashHead-1_3B",
        wav2vec_dir="models/wav2vec2-base-960h",
        model_type="lite"
    )

    print("SoulX pipeline loaded.")

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/pipeline-status")
def pipeline_status():
    return {
        "loaded": pipeline is not None
    }

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


    get_base_data(
        pipeline,
        cond_image_path_or_dir=image_path,
        base_seed=42,
        use_face_crop=False
    )

    return {
        "base_data_loaded": True,
        "image_saved": image_path,
        "audio_saved": audio_path
    }


   

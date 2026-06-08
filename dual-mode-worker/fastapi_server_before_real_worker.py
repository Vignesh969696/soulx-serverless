from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
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
import threading
import uuid

app = FastAPI()

pipeline = None
jobs = {}

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

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        return {
            "status": "not_found"
        }

    return jobs[job_id]


@app.get("/download/{job_id}")
def download_video(job_id: str):
    if job_id not in jobs:
        return {
            "error": "job not found"
        }

    video_path = jobs[job_id].get("video_path")

    if not video_path:
        return {
            "error": "video not ready"
        }

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )


def background_job(job_id):
    jobs[job_id]["status"] = "running"

    time.sleep(10)

    jobs[job_id]["status"] = "completed"



@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "queued",
        "video_path": None
    }

    threading.Thread(
        target=background_job,
        args=(job_id,),
        daemon=True
    ).start()

    return {
        "job_id": job_id
    }


from fastapi import FastAPI, UploadFile, File

from flash_head.inference import get_pipeline

app = FastAPI()

pipeline = None

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
    return {"loaded": pipeline is not None}

@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    return {
        "image_filename": image.filename,
        "audio_filename": audio.filename
    }

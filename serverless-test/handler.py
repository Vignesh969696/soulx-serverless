import os
import sys
import runpod

print("START", flush=True)

print("CWD:", os.getcwd(), flush=True)

sys.path.insert(0, "/app")

print("PATH:", sys.path, flush=True)
print("APP CONTENTS:", os.listdir("/app"), flush=True)

print(
    "SoulX exists:",
    os.path.exists("/runpod-volume/models/SoulX-FlashHead-1_3B"),
    flush=True
)

print(
    "wav2vec exists:",
    os.path.exists("/runpod-volume/models/wav2vec2-base-960h"),
    flush=True
)

try:
    from flash_head.inference import get_pipeline
    print("FLASH_HEAD IMPORTED", flush=True)

    print("LOADING PIPELINE", flush=True)

    pipeline = get_pipeline(
        world_size=1,
        ckpt_dir="/runpod-volume/models/SoulX-FlashHead-1_3B",
        wav2vec_dir="/runpod-volume/models/wav2vec2-base-960h",
        model_type="lite"
    )

    print("PIPELINE LOADED", flush=True)

except Exception as e:
    print("PIPELINE LOAD FAILED:", e, flush=True)
    raise


def handler(job):
    return {
        "pipeline_loaded": pipeline is not None
    }


print("REGISTERING", flush=True)
runpod.serverless.start({"handler": handler})

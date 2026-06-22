print("START", flush=True)

import time
START_TIME = time.time()

import runpod
from flash_head.inference import get_pipeline

print("RUNPOD IMPORTED", flush=True)

print("LOADING PIPELINE", flush=True)

pipeline = get_pipeline(
    world_size=1,
    ckpt_dir="/workspace/models/SoulX-FlashHead-1_3B",
    wav2vec_dir="/workspace/models/wav2vec2-base-960h",
    model_type="lite"
)

print("PIPELINE LOADED", flush=True)


def handler(job):
    return {
        "pipeline_loaded": pipeline is not None,
        "uptime": time.time() - START_TIME
    }


print("REGISTERING", flush=True)
runpod.serverless.start({"handler": handler})

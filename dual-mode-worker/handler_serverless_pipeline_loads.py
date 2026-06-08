import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import runpod
from flash_head.inference import get_pipeline

print("Loading SoulX pipeline...")

pipeline = get_pipeline(
    world_size=1,
    ckpt_dir="models/SoulX-FlashHead-1_3B",
    wav2vec_dir="models/wav2vec2-base-960h",
    model_type="lite"
)

print("SoulX pipeline loaded.")

def handler(job):
    print("Received job:")
    print(job)

    return {
        "status": "worker alive",
        "pipeline_loaded": pipeline is not None,
        "job_received": True
    }



runpod.serverless.start({
    "handler": handler
})

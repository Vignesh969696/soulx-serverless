print("START")

import torch

print("TORCH IMPORTED")

def handler(job):
    print("HANDLER")
    return {
        "cuda": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count()
    }

import runpod

runpod.serverless.start({"handler": handler})

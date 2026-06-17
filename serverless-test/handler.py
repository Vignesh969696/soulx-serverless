print("START")

import torch
print("TORCH IMPORTED")

def handler(job):
    print("HANDLER")
    print(job)

    return {
        "received": job
    }

import runpod

runpod.serverless.start({"handler": handler})

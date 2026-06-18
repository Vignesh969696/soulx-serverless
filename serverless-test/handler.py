print("START")

with open("/tmp/test.txt", "w") as f:
    f.write("HELLO")

import torch
print("TORCH IMPORTED")

import runpod
print("RUNPOD IMPORTED")

def handler(job):
    print("HANDLER CALLED")

    return {
        "cuda_available": torch.cuda.is_available()
    }

print("REGISTERING")
runpod.serverless.start({"handler": handler})

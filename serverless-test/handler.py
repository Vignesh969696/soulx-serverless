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
        "torch_version": torch.__version__
    }

print("REGISTERING")
runpod.serverless.start({"handler": handler})

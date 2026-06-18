print("START")

import torch
print("TORCH IMPORTED")

import runpod
print("RUNPOD IMPORTED")

def handler(job):
    print("HANDLER CALLED")
    print(job)

    return job
    

print("REGISTERING")
runpod.serverless.start({"handler": handler})

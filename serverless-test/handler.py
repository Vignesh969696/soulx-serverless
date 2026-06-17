print("STARTING FILE")

import runpod

print("IMPORTED RUNPOD")

def handler(job):
    print("HANDLER CALLED")
    return {"message": "hello"}

print("REGISTERING")

runpod.serverless.start({"handler": handler})

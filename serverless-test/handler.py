print("START", flush=True)

import os
import sys
import runpod

print("RUNPOD IMPORTED", flush=True)
print("CWD:", os.getcwd(), flush=True)
print("PATH:", sys.path, flush=True)
print("APP CONTENTS:", os.listdir("/app"), flush=True)

def handler(job):
    return {"ok": True}

print("REGISTERING", flush=True)
runpod.serverless.start({"handler": handler})

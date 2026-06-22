import os
import sys
import runpod

print("START", flush=True)

print("CWD:", os.getcwd(), flush=True)
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
except Exception as e:
    print("FLASH_HEAD IMPORT FAILED:", e, flush=True)
    raise


def handler(job):
    return {"ok": True}


print("REGISTERING", flush=True)
runpod.serverless.start({"handler": handler})

print("START", flush=True)

import runpod
print("RUNPOD IMPORTED", flush=True)

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

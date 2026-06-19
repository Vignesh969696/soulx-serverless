print("START", flush=True)

import time
START_TIME = time.time()

import runpod
print("RUNPOD IMPORTED", flush=True)


def handler(job):
    uptime = time.time() - START_TIME

    print("HANDLER CALLED", flush=True)
    print("UPTIME", uptime, flush=True)

    return {
        "uptime": uptime
    }


print("REGISTERING", flush=True)
runpod.serverless.start({"handler": handler})

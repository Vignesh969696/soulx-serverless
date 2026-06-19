print("START")

import time
START_TIME = time.time()

import runpod
print("RUNPOD IMPORTED")


def handler(job):
    uptime = time.time() - START_TIME

    print("HANDLER CALLED")
    print("UPTIME", uptime)

    return {
        "uptime": uptime
    }


print("REGISTERING")
runpod.serverless.start({"handler": handler})

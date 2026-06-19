print("START")

import runpod
print("RUNPOD IMPORTED")


def handler(job):
    print("HANDLER CALLED")

    return {
        "version": 999
    }


print("REGISTERING")
runpod.serverless.start({"handler": handler})

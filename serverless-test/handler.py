print("START")

import runpod
print("RUNPOD IMPORTED")


def handler(job):
    print("HANDLER CALLED")

    return {
        "I_AM_THE_HANDLER": 12345
    }


print("REGISTERING")
runpod.serverless.start({"handler": handler})

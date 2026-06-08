import runpod

def handler(event):
    return {
        "status": "worker alive"
    }

runpod.serverless.start({
    "handler": handler
})

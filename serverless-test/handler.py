import os
import sys
import runpod

print("START", flush=True)

print("CWD:", os.getcwd(), flush=True)

sys.path.insert(0, "/app")

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
    from flash_head.inference import (
        get_pipeline,
        get_base_data,
        get_infer_params,
        get_audio_embedding,
        run_pipeline,
    )

    import torch
    import numpy as np
    import librosa
    import time
    import subprocess
    import imageio
    from collections import deque

    print("ALL IMPORTS WORK", flush=True)

    print("LOADING PIPELINE", flush=True)

    pipeline = get_pipeline(
        world_size=1,
        ckpt_dir="/runpod-volume/models/SoulX-FlashHead-1_3B",
        wav2vec_dir="/runpod-volume/models/wav2vec2-base-960h",
        model_type="lite"
    )

    print("PIPELINE LOADED", flush=True)

except Exception as e:
    print("SETUP FAILED:", e, flush=True)
    raise


def save_video(frames_list, video_path, audio_path, fps):
    temp_video_path = video_path.replace(".mp4", "_tmp.mp4")

    with imageio.get_writer(
        temp_video_path,
        format="mp4",
        mode="I",
        fps=fps,
        codec="h264",
        ffmpeg_params=["-bf", "0"]
    ) as writer:

        for frames in frames_list:
            frames = frames.numpy().astype(np.uint8)

            for i in range(frames.shape[0]):
                writer.append_data(frames[i])

    cmd = [
        "ffmpeg",
        "-i", temp_video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        video_path,
        "-y"
    ]

    subprocess.run(cmd, check=True)

    os.remove(temp_video_path)


def handler(job):

    image_path = "/app/serverless-test/test.jpg"
    audio_path = "/app/serverless-test/test.wav"

    print("CALLING get_base_data", flush=True)

    get_base_data(
        pipeline,
        cond_image_path_or_dir=image_path,
        base_seed=42,
        use_face_crop=False
    )

    print("GETTING PARAMS", flush=True)

    infer_params = get_infer_params()

    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]

    slice_len = frame_num - motion_frames_num

    print("LOADING AUDIO", flush=True)

    human_speech_array_all, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True
    )

    human_speech_array_slice_len = (
        slice_len * sample_rate // tgt_fps
    )

    cached_audio_length_sum = (
        sample_rate * cached_audio_duration
    )

    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num

    audio_dq = deque(
        [0.0] * cached_audio_length_sum,
        maxlen=cached_audio_length_sum
    )

    remainder = len(human_speech_array_all) % human_speech_array_slice_len

    if remainder > 0:
        pad_length = human_speech_array_slice_len - remainder

        human_speech_array_all = np.concatenate(
            [
                human_speech_array_all,
                np.zeros(
                    pad_length,
                    dtype=human_speech_array_all.dtype
                )
            ]
        )

    human_speech_array_slices = human_speech_array_all.reshape(
        -1,
        human_speech_array_slice_len
    )

    print("GENERATING VIDEO", flush=True)

    generated_list = []

    for chunk_idx, human_speech_array in enumerate(
        human_speech_array_slices
    ):

        audio_dq.extend(
            human_speech_array.tolist()
        )

        audio_array = np.array(audio_dq)

        audio_embedding = get_audio_embedding(
            pipeline,
            audio_array,
            audio_start_idx,
            audio_end_idx
        )

        video = run_pipeline(
            pipeline,
            audio_embedding
        )

        video = video[motion_frames_num:]

        generated_list.append(
            video.cpu()
        )

    torch.cuda.synchronize()

    output_path = "/tmp/test.mp4"

    print("SAVING VIDEO", flush=True)

    print("NUMBER OF GENERATED CHUNKS:", len(generated_list), flush=True)
    save_video(
        generated_list,
        output_path,
        audio_path,
        tgt_fps
    )

    print("VIDEO SAVED", flush=True)

    return {
        "exists": os.path.exists(output_path),
        "size": os.path.getsize(output_path),
        "chunks": len(human_speech_array_slices)
    }



print("REGISTERING", flush=True)
runpod.serverless.start({"handler": handler})

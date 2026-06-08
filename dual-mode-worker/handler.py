import os
import sys
import time
import subprocess
import imageio
import librosa
import numpy as np
import torch

from collections import deque
from datetime import datetime

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import runpod

from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_infer_params,
    get_audio_embedding,
    run_pipeline,
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading SoulX pipeline...")

pipeline = get_pipeline(
    world_size=1,
    ckpt_dir="/workspace/SoulX-FlashHead/models/SoulX-FlashHead-1_3B",
    wav2vec_dir="/workspace/SoulX-FlashHead/models/wav2vec2-base-960h",
    model_type="lite"
)

print("SoulX pipeline loaded.")


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

    subprocess.run(cmd)

    os.remove(temp_video_path)


def handler(job):

    print("Received job:")
    print(job)

    image_path = job["input"]["image_path"]
    audio_path = job["input"]["audio_path"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    get_base_data(
        pipeline,
        cond_image_path_or_dir=image_path,
        base_seed=42,
        use_face_crop=False
    )

    infer_params = get_infer_params()

    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]

    slice_len = frame_num - motion_frames_num

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

        pad_length = (
            human_speech_array_slice_len - remainder
        )

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

    generated_list = []

    start_time = time.time()

    for human_speech_array in human_speech_array_slices:

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

    elapsed = time.time() - start_time

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{timestamp}.mp4"
    )

    save_video(
        generated_list,
        output_path,
        audio_path,
        tgt_fps
    )

    return {
        "status": "completed",
        "video_path": output_path,
        "generation_time": elapsed
    }


runpod.serverless.start({
    "handler": handler
})

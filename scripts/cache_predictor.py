import argparse
import os

import codewithgpu
import cv2
import decord
import numpy as np
import pandas as pd
import torch
from diffusers.models import AutoencoderKLCogVideoX


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/libero")
    parser.add_argument("--output_dir", type=str, default="cache/predictor")
    parser.add_argument("--vae_path", type=str, default="models/vae")
    parser.add_argument("--chunk_split_path", type=str, default="data/libero/chunk_split.jsonl")
    return parser.parse_args()


demo_id = 0
chunk_id = 0
args = parse_args()
data_dir = args.data_dir
output_dir = args.output_dir
vae_path = args.vae_path
chunk_split_path = args.chunk_split_path

device, dtype = torch.device("cuda"), torch.float16
vae = AutoencoderKLCogVideoX.from_pretrained(vae_path, low_cpu_mem_usage=False)
vae = vae.to(device=device, dtype=dtype).eval()

features = {
    "caption": "string",
    "image_shape": ["int64"],
    "image": "bytes",
    "label": "int",
}

chunk_splits = pd.read_json(
    chunk_split_path,
    lines=True,
)

try:
    os.remove(os.path.join(output_dir, "00000.data"))
    os.remove(os.path.join(output_dir, "00000.index"))
    os.remove(os.path.join(output_dir, "METADATA"))
except OSError:
    pass
_, writer = os.makedirs(output_dir, exist_ok=True), codewithgpu.RecordWriter(output_dir, features)


def get_label(length):
    """Map chunk length in latent units {2, 4, 6, 8} to class {0, 1, 2, 3}."""
    if length == 2:
        return 0
    elif length == 4:
        return 1
    elif length == 6:
        return 2
    elif length == 8:
        return 3
    else:
        print(f"Invalid length: {length}")
        raise Exception("Invalid length")


for demoname in os.listdir(data_dir):
    demo_path = os.path.join(data_dir, demoname)
    vid = decord.VideoReader(os.path.join(demo_path, "video.mp4"))
    length = len(vid)
    vid = vid.get_batch(list(range(length))).asnumpy()
    latent_frame_num = length // 8 + 1
    frame_num = (latent_frame_num) * 8 + 1
    frame_ids = list(range(length))
    if frame_num > length:
        frame_ids += [length - 1] * (frame_num - length)
    vid = vid[frame_ids]
    x = torch.as_tensor(vid[None, ...].transpose((0, 4, 1, 2, 3))).to(device).to(dtype)

    image = cv2.imread(os.path.join(demo_path, "first_frame.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 127.5 - 1.0
    tensor_image = (
        torch.from_numpy(image).permute(2, 0, 1).unsqueeze(1).unsqueeze(0).to(device).to(dtype)
    )

    with open(os.path.join(demo_path, "prompt.txt"), "r", encoding="utf-8") as f:
        task = f.read().strip()

    chunk_split = chunk_splits.loc[chunk_splits["demo_path"] == demo_path, "chunk_ids"].values[0]

    with torch.no_grad():
        x = (
            (vae.encode(x.sub(127.5).div(127.5)).latent_dist.sample() * vae.config.scaling_factor)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        x_image = (
            (vae.encode(tensor_image).latent_dist.sample() * vae.config.scaling_factor)
            .squeeze(0)
            .cpu()
            .numpy()
        )

        for i in range(len(chunk_split) - 1):
            # skip chunks extending past the encoded latents (e.g. tails inside the
            # padded region)
            if chunk_split[i + 1] + 1 > x.shape[1]:
                print(
                    f"demo_{demo_id} has {x.shape[1]} frames, "
                    f"but chunk_split[{i+1}] is {chunk_split[i+1]}"
                )
                break
            chunk_length = chunk_split[i + 1] - chunk_split[i]
            x_image_split = np.expand_dims(x[:, chunk_split[i], :, :], axis=1) if i > 0 else x_image
            writer.write(
                {
                    "image_shape": x_image_split.shape,
                    "image": x_image_split.tobytes(),
                    "label": get_label(chunk_length),
                    "caption": task,
                }
            )

            print(f"Writing chunk_{chunk_id} in demo_{demo_id}.")
            chunk_id += 1
    demo_id += 1


print(f"Finished writing {chunk_id} chunks in {demo_id} demos.")
writer.close()

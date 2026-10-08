import argparse
import os

import codewithgpu
import cv2
import decord
import numpy as np
import torch
from diffusers.models import AutoencoderKLCogVideoX


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/libero")
    parser.add_argument("--output_dir", type=str, default="cache/expert")
    parser.add_argument("--vae_path", type=str, default="models/vae")
    parser.add_argument("--chunk_length", type=int, default=33, choices=[9, 17, 25, 33])
    return parser.parse_args()


args = parse_args()
demo_id = 0
chunk_id = 0
data_dir = args.data_dir
output_dir = args.output_dir
vae_path = args.vae_path

device, dtype = torch.device("cuda"), torch.float16
vae = AutoencoderKLCogVideoX.from_pretrained(vae_path, low_cpu_mem_usage=False)
vae = vae.to(device=device, dtype=dtype).eval()

features = {
    "moments": "bytes",
    "caption": "string",
    "shape": ["int64"],
    "image_shape": ["int64"],
    "image": "bytes",
}

try:
    os.remove(os.path.join(output_dir, "00000.data"))
    os.remove(os.path.join(output_dir, "00000.index"))
    os.remove(os.path.join(output_dir, "METADATA"))
except OSError:
    pass
_, writer = os.makedirs(output_dir, exist_ok=True), codewithgpu.RecordWriter(output_dir, features)

# for different experts, you may need to change chunk_length accordingly
chunk_length = args.chunk_length
# latent frames per chunk: 9/17/25/33 frames -> 3/5/7/9 latents
latent_chunk_length = chunk_length // 4 + 1

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

        start_id = 0
        end_id = 0

        while end_id < x.shape[1]:
            end_id = min(start_id + latent_chunk_length, x.shape[1])
            start_id = max(end_id - latent_chunk_length, 0)
            if end_id - start_id != latent_chunk_length:
                break
            x_split = x[:, start_id:end_id, :, :]
            x_image_split = (
                np.expand_dims(x[:, start_id, :, :], axis=1) if start_id > 0 else x_image
            )
            writer.write(
                {
                    "shape": x_split.shape,
                    "image_shape": x_image_split.shape,
                    "moments": x_split.tobytes(),
                    "image": x_image_split.tobytes(),
                    "caption": task,
                }
            )
            # consecutive chunks overlap by one latent conditioning frame
            start_id += latent_chunk_length - 1

            print(f"Writing chunk_{chunk_id} in demo_{demo_id}.")
            chunk_id += 1
    demo_id += 1

print(f"Finished writing {chunk_id} chunks in {demo_id} demos.")
writer.close()

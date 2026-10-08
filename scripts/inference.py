import PIL.Image
import numpy as np
from diffnext.pipelines import LongScapePipeline
from diffnext.utils import export_to_video
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/longscape")
    parser.add_argument("--output_path", type=str, default="examples/output_video.mp4")
    parser.add_argument("--first_frame_path", type=str, default="examples/first_frame.png")
    parser.add_argument(
        "--prompt", type=str, default="pick up the ketchup and place it in the basket."
    )
    return parser.parse_args()


args = parse_args()
model_path = args.model_path
output_path = args.output_path
first_frame_path = args.first_frame_path
prompt = args.prompt

pipe = LongScapePipeline.from_pretrained(
    model_path,
    local_files_only=True,
    low_cpu_mem_usage=False,
)

first_frame = np.array(PIL.Image.open(first_frame_path))

# set the max latent video length you want to generate
max_video_length = 50

video = pipe(
    prompt=prompt,
    image=first_frame,
    max_latent_length=max_video_length,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]
export_to_video(
    video,
    output_path,
    fps=10,
)

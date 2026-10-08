import argparse
import glob
import json
import os

import torch
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(description="将 .bin 权重转换为推理用的 safetensors 格式")
    parser.add_argument("--ckpt_dir", type=str, default="models/ckpt", help="checkpoint 根目录")
    parser.add_argument("--output_path", type=str, default="models/longscape", help="输出目录")
    return parser.parse_args()


def load_bin_shards(bin_path):
    # 加载目录下所有 .bin 分片并合并为一个 state_dict
    state_dict = {}
    for f in sorted(glob.glob(os.path.join(bin_path, "diffusion_pytorch_model*.bin"))):
        state_dict.update(torch.load(f, map_location="cpu"))
    return state_dict


def convert_transformer(bin_path, output_path):
    # 1. 加载权重
    print(f"开始加载权重: {bin_path}")
    state_dict = load_bin_shards(bin_path)
    print("加载完成")

    # 2. 按参数名顺序分割权重为三部分
    keys = list(state_dict.keys())
    n = len(keys) // 3
    parts = [keys[:n], keys[n : 2 * n], keys[2 * n :]]
    part_dict = [{k: state_dict[k] for k in parts[i]} for i in range(3)]

    # 3. 保存为 safetensors
    os.makedirs(output_path, exist_ok=True)
    files = [
        os.path.join(output_path, f"diffusion_pytorch_model-{i+1}-of-3.safetensors")
        for i in range(3)
    ]
    print("开始保存权重")
    for i in range(3):
        if os.path.exists(files[i]):
            os.remove(files[i])
        save_file(part_dict[i], files[i])
        print(f"第{i+1}部分保存完成")
    print("保存完成")

    # 4. 生成 index.json
    weight_map = {}
    total_size = 0
    for i, part_keys in enumerate(parts):
        for k in part_keys:
            weight_map[k] = os.path.basename(files[i])
            total_size += state_dict[k].numel()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(
        os.path.join(output_path, "diffusion_pytorch_model.safetensors.index.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(index, f, indent=2)


def convert_predictor(bin_path, output_path):
    print(f"开始加载权重: {bin_path}")
    state_dict = load_bin_shards(bin_path)
    print("加载完成")

    os.makedirs(output_path, exist_ok=True)
    print("开始保存权重")
    save_file(state_dict, os.path.join(output_path, "diffusion_pytorch_model.safetensors"))
    print("保存完成")


if __name__ == "__main__":
    args = parse_args()

    for i in range(4):
        convert_transformer(
            os.path.join(args.ckpt_dir, f"transformer_{i}"),
            os.path.join(args.output_path, f"transformer_{i}"),
        )
    convert_predictor(
        os.path.join(args.ckpt_dir, "predictor"),
        os.path.join(args.output_path, "predictor"),
    )

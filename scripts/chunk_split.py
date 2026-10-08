import os
import numpy as np
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/libero")
    parser.add_argument("--output_path", type=str, default="data/libero/chunk_split.jsonl")
    return parser.parse_args()


def get_keypoint(demo):
    """Return frame indices where the binary gripper state flips sign."""
    keypoint = []
    for i in range(1, len(demo)):
        if demo[i - 1][6] * demo[i][6] < 0:
            keypoint.append(i)
    return keypoint


def action_split_in_chunk(action, action_range, start, end):
    """Split [start, end) into chunks by motion magnitude, growth in latent units."""
    segments = [start]
    start_id = start
    end_id = start_id + 8
    while start_id < end:
        # candidate lengths in latent units (1 unit = 4 frames) -> 8/16/24/32 frames
        for chunk_length in (2, 4, 6, 8):
            end_id = start_id + 4 * chunk_length
            if end_id < end:
                chunk_action_range = (
                    np.max(action[start_id:end_id], axis=0)
                    - np.min(action[start_id:end_id], axis=0)
                )[0:6]
                if np.any(chunk_action_range > 0.5 * action_range):
                    break
            elif end_id == end:
                break
            else:
                # truncation fallback for segments shorter than the candidate;
                # unreachable in the pipeline since all boundaries are multiples of 8
                end_id = start_id + chunk_length * 2
                break
        start_id = end_id
        segments.append(start_id)
    return segments


def action_split(action, action_split_chunk):
    """Split each segment between forced boundaries by motion magnitude."""
    segments = []
    action_range = (np.max(action, axis=0) - np.min(action, axis=0))[0:6]
    for i in range(len(action_split_chunk) - 1):
        segments_chunk = action_split_in_chunk(
            action, action_range, action_split_chunk[i], action_split_chunk[i + 1]
        )
        segments.extend(segments_chunk)
    segments = sorted(set(segments))
    return segments


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    output_path = args.output_path
    compression_ratios = []
    chunk_lengths = []
    demo_info_list = []

    for demoname in os.listdir(data_dir):
        demo_path = os.path.join(data_dir, demoname)
        action_numpy = np.load(
            os.path.join(demo_path, "action.npy"),
            allow_pickle=True,
        )

        length = len(action_numpy)

        # pad the action sequence up to a multiple of 8 frames
        real_length = 8 * ((length + 7) // 8)
        action_split_chunk = [0, real_length]

        repeat = np.repeat(action_numpy[-1][np.newaxis, :], real_length - length, axis=0)
        action_numpy = np.vstack([action_numpy, repeat]) if length < real_length else action_numpy

        keypoint = get_keypoint(action_numpy)

        # force a boundary before/after every base chunk containing a gripper change,
        # so each of them becomes an independent chunk
        for kp in keypoint:

            start = max(0, (kp // 8) * 8)
            end = min(length, start + 8)

            if start not in action_split_chunk:
                action_split_chunk.append(start)
            if end not in action_split_chunk and end < length:
                action_split_chunk.append(end)

        action_split_chunk = action_split(action_numpy, action_split_chunk)

        # frame boundaries -> latent units (1 unit = 4 frames)
        chunk_ids = [x // 4 for x in action_split_chunk]

        chunk_length = [chunk_ids[i + 1] - chunk_ids[i] for i in range(len(chunk_ids) - 1)]
        chunk_lengths.extend(chunk_length)

        demo_info = {
            "demo_path": demo_path,
            "chunk_ids": chunk_ids,
            "compression_ratio": (length // 4) / len(chunk_ids),
        }
        demo_info_list.append(demo_info)

    with open(output_path, "w") as f:
        for demo_info in demo_info_list:
            f.write(json.dumps(demo_info) + "\n")

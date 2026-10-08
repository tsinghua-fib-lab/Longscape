<div align="center">

<h1>LongScape</h1>

</div>

We propose **LongScape**, a novel embodied world model that conducts adaptive chunk rollout according to the generation context, enabling stable long-horizon video generation. **LongScape** provides an action-guided, variable-length chunking mechanism that partitions video based on the semantic context of robotic actions.
**LongScape** leverages a Context-aware Mixture-of-Experts (CMoE) framework that adaptively activates specialized experts for each chunk during generation, guaranteeing high visual quality and seamless chunk transitions.
**LongScape** achieves stable and consistent long-horizon generation over extended rollouts.

# Table of Contents
- [1. Installation](#1-installation)
- [2. Cache](#2-cache)
- [3. Train](#3-train)
- [4. Inference](#4-inference)

# 1. Installation

<a id="from-source"></a>
Clone this repository to local disk and install:

```bash
cd LongScape
pip install -r requirements.txt
pip install .
```

# 2. Cache
Your raw data should be in the following format:

```
raw_data
|-- demo_0
|   |-- video.mp4
|   |-- first_frame.png
|   |-- action.npy
|   |-- prompt.txt
|-- demo_1
|-- demo_2
|-- ...
```

`prompt.txt` contains the natural-language task description used as the text condition for training and inference.

## 2.1. Build experts training cache
Experts training cache can be built by ```python ./scripts/cache_expert.py```. This script will split each demo into chunks of fixed length, and preprocess images or videos into VAE latents. Chunk length can be specified by ```--chunk_length``` argument.

## 2.2. Chunk split based on action context
Chunk split can be done by ```python ./scripts/chunk_split.py```. This script will split each demo into chunks based on action context and save result into a jsonl file.

## 2.3. Built predictor training cache 
Predictor training cache can be built by ```python ./scripts/cache_predictor.py```. This script will create cache based on chunk split results and the first VAE latent in chunks.

# 3. Train
Models can be trained by ```python ./scripts/train.py```. Different models can be trained by specifying ```--cfg``` argument. If you want to use DeepSpeed, you can specify ```--deepspeed``` argument.

## 3.1. Train experts
Experts can be trained by using ```./configs/train/expert.yml```

## 3.2. Train predictor
Predictor can be trained by using ```./configs/train/predictor.yml```

# 4. Inference

## 4.1. Convert checkpoint weights
The checkpoint weights are stored in `.bin` format. Convert them to the `safetensors` format expected at inference time:

```bash
python ./scripts/inference_prepare.py --ckpt_dir models/ckpt --output_path models/longscape
```

This converts the four transformer experts (`transformer_0` to `transformer_3`) and the chunk predictor under `--ckpt_dir` (default `models/ckpt`), and writes the inference-ready pipeline to `--output_path` (default `models/longscape`).

## 4.2. Generate videos
Inference can be done by ```python ./scripts/inference.py```.

# Acknowledgement
This project is based on the [**NOVA**](https://github.com/baaivision/NOVA), which is licensed under the Apache License, Version 2.0.

See the `LICENSE` file for the full license text.

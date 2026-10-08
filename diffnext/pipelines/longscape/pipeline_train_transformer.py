# Copyright (c) 2024-present, BAAI. All Rights Reserved.
# Created by LongScape Authors on 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications made by LongScape Authors on 2025:
# - Add LongScapeTrainingPipeline.
##############################################################################
"""LongScape T2V Action training pipeline."""

from typing import Dict
import numpy as np

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch

from diffnext.pipelines.builder import PIPELINES
from diffnext.pipelines.longscape.pipeline_utils import PipelineMixin


@PIPELINES.register("longscape_train_transformer")
class LongScapeTrainPipeline(DiffusionPipeline, PipelineMixin):
    """Pipeline for training LongScape T2V Action models."""

    _optional_components = ["transformer", "scheduler", "vae", "text_encoder", "tokenizer"]

    def __init__(
        self,
        transformer=None,
        scheduler=None,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        trust_remote_code=True,
    ):
        super(LongScapeTrainPipeline, self).__init__()
        self.vae = self.register_module(vae, "vae").eval()
        self.text_encoder = self.register_module(text_encoder, "text_encoder")
        self.tokenizer = self.register_module(tokenizer, "tokenizer")
        self.transformer = self.register_module(transformer, "transformer")
        self.scheduler = self.register_module(scheduler, "scheduler")

        self.transformer.set_scheduler(self.scheduler)
        self.transformer.set_vae(self.vae)

    @property
    def model(self) -> torch.nn.Module:
        """Return the trainable model."""
        return self.transformer

    def configure_model(self, checkpointing=0, config=None) -> torch.nn.Module:
        """Configure the trainable model."""
        ckpt_lvl = config.TRAIN.CHECKPOINTING if config else checkpointing
        (
            self.model.dit.enable_gradient_checkpointing()
            if ckpt_lvl > 0
            else self.model.dit.disable_gradient_checkpointing()
        )
        self.model.pipeline_preprocess = self.preprocess
        return self.model.train()

    def prepare_latents(self, inputs: Dict):
        """Prepare the video latents."""
        if "image" in inputs:
            inputs["c"] = torch.as_tensor(np.array(inputs.pop("image")), device=self.device).to(
                dtype=self.dtype
            )
        if "moments" in inputs:
            inputs["x"] = torch.as_tensor(np.array(inputs.pop("moments")), device=self.device).to(
                dtype=self.dtype
            )

    def encode_prompt(self, inputs: Dict):
        """Encode text prompts."""
        prompt = inputs.get("prompt", [])
        prompt_token_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.text_encoder(prompt_token_ids.to(self.text_encoder.device))[0]
        inputs["text_embedding"] = prompt_embedding

    def preprocess(self, inputs: Dict) -> Dict:
        """Define the pipeline preprocess at every call."""
        if not self.model.training:
            raise RuntimeError("Excepted a trainable model.")
        self.prepare_latents(inputs)
        self.encode_prompt(inputs)

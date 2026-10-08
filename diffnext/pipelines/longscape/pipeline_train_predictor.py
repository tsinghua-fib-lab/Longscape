# ------------------------------------------------------------------------
# Copyright (c) 2025-present, LongScape Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONditIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""LongScape Predictor training pipeline."""

from typing import Dict
import numpy as np

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch

from diffnext.pipelines.builder import PIPELINES
from diffnext.pipelines.longscape.pipeline_utils import PipelineMixin


@PIPELINES.register("longscape_train_predictor")
class LongScapeTrainPipeline_Predictor(DiffusionPipeline, PipelineMixin):
    """Pipeline for training chunk predictor models."""

    _optional_components = ["predictor", "text_encoder", "tokenizer"]

    def __init__(
        self,
        predictor=None,
        text_encoder=None,
        tokenizer=None,
        trust_remote_code=True,
    ):
        super(LongScapeTrainPipeline_Predictor, self).__init__()
        self.predictor = self.register_module(predictor, "predictor")
        self.text_encoder = self.register_module(text_encoder, "text_encoder")
        self.tokenizer = self.register_module(tokenizer, "tokenizer")
        self.max_text_seq_length = 226

    @property
    def model(self) -> torch.nn.Module:
        """Return the trainable model."""
        return self.predictor

    def configure_model(self, config=None) -> torch.nn.Module:
        """Configure the trainable model."""
        self.model.pipeline_preprocess = self.preprocess
        return self.model.train()

    def prepare_latents(self, inputs: Dict):
        """Prepare the video latents."""
        if "image" in inputs:
            inputs["latent"] = (
                torch.as_tensor(np.array(inputs.pop("image")), device=self.device)
                .to(dtype=self.dtype)
                .permute(0, 2, 1, 3, 4)
            )
        if "label" in inputs:
            inputs["label"] = torch.as_tensor(np.array(inputs.pop("label")), device=self.device).to(
                dtype=torch.long
            )

    def encode_prompt(self, inputs: Dict):
        """Encode text prompts."""
        prompt = inputs.get("prompt", [])
        prompt_token_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.text_encoder(prompt_token_ids.to(self.text_encoder.device))[0]
        inputs["text_emb"] = prompt_embedding

    def preprocess(self, inputs: Dict) -> Dict:
        """Define the pipeline preprocess at every call."""
        if not self.model.training:
            raise RuntimeError("Excepted a trainable model.")
        self.prepare_latents(inputs)
        self.encode_prompt(inputs)

# Copyright (c) 2024-present, BAAI. All Rights Reserved.
# Copyright (c) 2025 LongScape Authors.
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
# - Add LongScapePipeline class
##############################################################################
"""Video generation pipeline for LongScape."""

import torch
import math
import time
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Union, List
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from diffnext.image_processor import VaeImageProcessor
from diffnext.pipelines.builder import PIPELINES
from diffnext.pipelines.longscape.pipeline_utils import LongScapePipelineOutput, PipelineMixin
import logging
import inspect
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.schedulers import CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


@PIPELINES.register("longscape")
class LongScapePipeline(DiffusionPipeline, PipelineMixin):
    """LongScape autoregressive diffusion pipeline."""

    _optional_components = [
        "transformer_0",
        "transformer_1",
        "transformer_2",
        "transformer_3",
        "scheduler",
        "vae",
        "text_encoder",
        "tokenizer",
        "predictor",
    ]
    model_cpu_offload_seq = (
        "text_encoder->[transformer_0,transformer_1,transformer_2,transformer_3]->predictor->vae"
    )

    def __init__(
        self,
        transformer_0=None,
        transformer_1=None,
        transformer_2=None,
        transformer_3=None,
        scheduler=None,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        predictor=None,
    ):
        super(LongScapePipeline, self).__init__()
        # Load the pre-trained models or pass them as arguments
        self.scheduler = self.register_module(scheduler, "scheduler")
        self.vae = self.register_module(vae, "vae")
        self.text_encoder = self.register_module(text_encoder, "text_encoder")
        self.tokenizer = self.register_module(tokenizer, "tokenizer")
        self.transformer_0 = self.register_module(transformer_0, "transformer_0")
        self.transformer_1 = self.register_module(transformer_1, "transformer_1")
        self.transformer_2 = self.register_module(transformer_2, "transformer_2")
        self.transformer_3 = self.register_module(transformer_3, "transformer_3")
        self.predictor = self.register_module(predictor, "predictor")

        self.experts = [
            self.transformer_0,
            self.transformer_1,
            self.transformer_2,
            self.transformer_3,
        ]
        for expert in self.experts:
            expert.to(device="cpu")
        self.predictor.to(device="cpu")
        self.vae.to(device="cpu")
        self.text_encoder.to(device="cpu")

        self._guidance_scale = 6.0

        self.height, self.width = self.vae.config.sample_height, self.vae.config.sample_width

        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if getattr(self, "vae", None) else 4
        )
        self.vae_scaling_factor_image = (
            self.vae.config.scaling_factor if getattr(self, "vae", None) else 0.7
        )
        self.tokenizer_max_length = self.transformer_0.config.max_text_seq_length
        self.do_classifier_free_guidance = self._guidance_scale > 1.0

        self.use_dynamic_cfg = True

        self.use_rotary_positional_embeddings = True

        self.image_processor = VaeImageProcessor()

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = self.text_encoder.device if device is None else device
        dtype = self.text_encoder.dtype if dtype is None else dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, max_sequence_length - 1 : -1]
            )
            logging.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds

    # Copied from StableDiffusionPipeline.prepare_extra_step_kwargs
    def _prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from CogVideoXPipeline._prepare_rotary_positional_embeddings
    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (
            self.vae_scale_factor_spatial * self.transformer_0.config.patch_size
        )
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer_0.config.patch_size)

        p = self.transformer_0.config.patch_size
        p_t = self.transformer_0.config.patch_size_t

        base_size_width = self.transformer_0.config.sample_width // p
        base_size_height = self.transformer_0.config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer_0.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer_0.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        num_inference_steps=50,
        max_latent_length=1,
        guidance_scale=6,
        negative_prompt=None,
        image=None,
        generator=None,
        latents=None,
        output_type="pil",
    ) -> LongScapePipelineOutput:
        """The call function to the pipeline for generation.

        Args:
            prompt (str or List[str], *optional*):
                The prompt to be encoded.
            num_inference_steps (int, *optional*, defaults to 50):
                The number of denoising steps for each chunk.
            max_latent_length (int, *optional*, defaults to 1):
                The maximum number of latents to generate. ``1`` for image generation.
            guidance_scale (float, *optional*, defaults to 6):
                The classifier guidance scale.
            negative_prompt (str or List[str], *optional*):
                The prompt or prompts to guide what to not include in image generation.
            image (numpy.ndarray, *optional*):
                The image to be encoded as the first frame.
            generator (torch.Generator, *optional*):
                The random generator.
            latents (List[torch.Tensor], *optional*)
                A list of prefilled VAE latents.
            output_type (str, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.

        Returns:
            LongScapePipelineOutput: The pipeline output.
        """
        self.guidance_scale = guidance_scale
        inputs = {"generator": generator, **locals()}
        inputs["prompt"] = prompt
        inputs["prompt"] = (
            self.encode_prompt(
                prompt=inputs.get("prompt", None),
                negative_prompt=inputs.get("negative_prompt", None),
            )
            if inputs.get("prompt", None) is not None
            else None
        )
        inputs["latents"] = self.prepare_latents(image, generator, latents)
        inputs["batch_size"] = 1
        inputs["time"] = 0.0

        inputs.pop("self")
        self.generate_video(inputs)
        outputs = torch.cat(inputs["latents"], dim=1)
        if output_type != "latent":
            outputs = self.decode_latents(outputs)
        output_name = {4: "images", 5: "frames"}[len(outputs.shape)]
        outputs = self.image_processor.postprocess(outputs, output_type)
        return LongScapePipelineOutput(**{output_name: outputs})

    @torch.inference_mode()
    def generate_chunk(
        self,
        expert_id=0,
        num_inference_steps=50,
        guidance_scale=6.0,
        generator=None,
        latents=None,
        image_latents=None,
        prompt_embeds=None,
        image_rotary_emb=None,
    ):
        """Generate a batch of frames."""

        old_pred_original_sample = None

        extra_step_kwargs = self._prepare_extra_step_kwargs(generator, 0.0)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device
        )

        for i, t in enumerate(timesteps):

            latent_model_input = (
                torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            latent_image_input = (
                torch.cat([image_latents] * 2)
                if self.do_classifier_free_guidance
                else image_latents
            )
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # model forward (the scheduler interprets the output per its prediction_type)
            transformer_kwargs = {
                "hidden_states": latent_model_input,
                "encoder_hidden_states": prompt_embeds.to(latent_model_input.device),
                "timestep": timestep.to(latent_model_input.device),
                "image_rotary_emb": image_rotary_emb,
                "return_dict": False,
            }
            expert = self.experts[expert_id]
            noise_pred = expert.dit(**transformer_kwargs)[0]
            noise_pred = noise_pred.float()

            # perform guidance (dynamic CFG: cosine-decayed from full scale at the
            # first denoising step to 1 at the last)
            if self.use_dynamic_cfg:
                self._guidance_scale = 1 + guidance_scale * (
                    (
                        1
                        - math.cos(
                            math.pi * ((num_inference_steps - i) / num_inference_steps) ** 5.0
                        )
                    )
                    / 2
                )
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self._guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
            else:
                latents, old_pred_original_sample = self.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(self.dtype)

        return latents

    @torch.inference_mode()
    def generate_video(self, inputs: Dict):
        """Generate a batch of videos."""
        max_latent_length = inputs.get("max_latent_length", 1)
        num_inference_steps = inputs.get("num_inference_steps", 50)
        guidance_scale = inputs.get("guidance_scale", 6.0)

        latents = inputs.get("latents", [])
        dtype, device = latents[-1].dtype, latents[-1].device

        text_embedding = inputs.get("prompt", None).to(device).to(dtype)
        generator = inputs.get("generator", None)
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(42)

        def generate_noise(bs, chunk_length, channels, height, width, generator, dtype, device):
            return torch.randn(
                bs,
                chunk_length,
                channels,
                height,
                width,
                generator=generator,
                dtype=dtype,
                device=device,
            )

        def generate_padding(bs, chunk_length, channels, height, width, dtype, device):
            return torch.zeros(
                bs, chunk_length, channels, height, width, dtype=dtype, device=device
            )

        t = 0
        last_expert_id = -1
        self.predictor.to(device).to(dtype)

        pbar = tqdm(total=max_latent_length, desc="Generating video")
        while t < max_latent_length:
            chunk_start_time = time.time()

            # route with the last latent frame of the current chunk and the positive
            # text embedding (text_embedding = [negative; positive])
            latent = latents[-1]
            predictor_inputs = {
                "latent": latent,
                "text_emb": text_embedding[1].unsqueeze(0),
            }
            next_chunk_length = int(self.predictor(predictor_inputs))
            # latent chunk lengths 3/5/7/9 map to experts 0/1/2/3 (8/16/24/32 frames)
            expert_id = next_chunk_length // 2 - 1
            pred_noise = generate_noise(
                latent.shape[0],
                next_chunk_length,
                latent.shape[2],
                latent.shape[3],
                latent.shape[4],
                generator,
                dtype,
                device,
            )
            pred_noise = pred_noise * self.scheduler.init_noise_sigma
            padding = generate_padding(
                latent.shape[0],
                next_chunk_length - 1,
                latent.shape[2],
                latent.shape[3],
                latent.shape[4],
                dtype,
                device,
            )

            # condition on the last latent frame of the previous chunk; the remaining
            # frames are zero-padding (matches the training-time conditioning)
            image_latents = torch.cat([latent, padding], dim=1)

            image_rotary_emb = (
                self._prepare_rotary_positional_embeddings(
                    self.height,
                    self.width,
                    pred_noise.size(1),
                    self.device,
                )
                if self.use_rotary_positional_embeddings
                else None
            )
            chunk_middle_time_1 = time.time()
            expert = self.experts[expert_id]
            if expert_id != last_expert_id:
                if last_expert_id != -1:
                    self.experts[last_expert_id].to("cpu")
                expert.to(latent.device)
            chunk_middle_time_2 = time.time()
            chunk_latents = self.generate_chunk(
                expert_id=expert_id,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                latents=pred_noise,
                image_latents=image_latents,
                prompt_embeds=text_embedding,
                image_rotary_emb=image_rotary_emb,
            )
            # the first latent frame re-encodes the conditioning context; keep the rest
            for chunk in torch.split(chunk_latents, 1, dim=1)[1:]:
                latents.append(chunk.clone())

            # consecutive chunks overlap by one latent conditioning frame
            pbar.update(min(next_chunk_length - 1, max_latent_length - t))
            t += next_chunk_length - 1
            last_expert_id = expert_id
            chunk_end_time = time.time()
            inputs["time"] += (
                chunk_end_time - chunk_middle_time_2 + chunk_middle_time_1 - chunk_start_time
            )

        pbar.close()
        _ = [self.experts[i].to("cpu") for i in range(len(self.experts))]
        self.predictor.to("cpu")

    def encode_prompt(
        self,
        prompt,
        max_sequence_length=226,
        negative_prompt=None,
    ) -> torch.Tensor:
        """Encode text prompts.

        Args:
            prompt (str or List[str]):
                The prompt to be encoded.
            max_sequence_length (int, *optional*, defaults to 226):
                The maximum length of the tokenized prompt.
            negative_prompt (str or List[str], *optional*):
                The prompt or prompts not to guide the image generation.

        Returns:
            torch.Tensor: The prompt embedding, concatenated with the negative prompt
                embedding when guidance is enabled.
        """

        self.text_encoder.to("cuda")
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
        )

        if self.guidance_scale > 1.0:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
            )
            prompt_embeds = torch.cat(
                [negative_prompt_embeds.to(prompt_embeds.device), prompt_embeds], dim=0
            )

        self.text_encoder.to("cpu")

        return prompt_embeds

    def prepare_latents(
        self,
        image=None,
        generator=None,
        latents=None,
    ) -> List[torch.Tensor]:
        """Prepare the video latents.

        Args:
            image (numpy.ndarray, *optional*):
                The image to be encoded.
            generator (torch.Generator, *optional*):
                The random generator.
            latents (List[torch.Tensor], *optional*)
                A list of prefilled VAE latents.

        Returns:
            List[torch.Tensor]: The encoded latents.
        """
        if latents is not None:
            return latents
        latents = []
        if image is not None:
            latents.append(self.encode_image(image, generator))
        return latents

    def encode_image(self, image, generator=None) -> torch.Tensor:
        """Encode image prompt.

        Args:
            image (numpy.ndarray):
                The image to be encoded.
            num_images_per_prompt (int):
                The number of images that should be generated per prompt.
            generator (torch.Generator, *optional*):
                The random generator.

        Returns:
            torch.Tensor: The image embedding.
        """
        x = torch.as_tensor(image, device="cuda").to(dtype=self.dtype)
        self.vae.to(x.device)
        x = x.sub(127.5).div_(127.5).permute(2, 0, 1).unsqueeze_(1)
        # x = retrieve_latents(self.vae.encode(x.unsqueeze(0)), generator)
        with torch.no_grad():
            x = self.vae.encode(x.unsqueeze_(0)).latent_dist.sample()
        self.vae.to("cpu")
        return (self.vae_scaling_factor_image * x.permute(0, 2, 1, 3, 4)).to(dtype=self.dtype)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(
            0, 2, 1, 3, 4
        )  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        self.vae.to(latents.device)
        frames = self.vae.decode(latents).sample
        self.vae.to("cpu")
        return frames

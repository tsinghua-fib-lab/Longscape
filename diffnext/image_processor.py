# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
# Copyright 2025 LongScape Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications made by LongScape Authors on 2025:
# - Modified VaeImageProcessor for LongScape requirements.
# ------------------------------------------------------------------------
"""Image processor."""

from typing import List, Union

import numpy as np
import PIL.Image
import torch

from diffusers.configuration_utils import ConfigMixin


class VaeImageProcessor(ConfigMixin):
    """Image processor for VAE."""

    def postprocess(
        self, image: torch.Tensor, output_type: str = "pil"
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        """Postprocess the image output from tensor.

        Args:
            image (torch.Tensor):
                The image tensor.
            output_type (str, *optional*, defaults to `pil`):
                The output image type, can be one of `pil`, `np`, `pt`, `latent`.

        Returns:
            Union[PIL.Image.Image, np.ndarray, torch.Tensor]: The postprocessed image.
        """
        if output_type == "latent" or output_type == "pt":
            return image
        image = self.pt_to_numpy(image)
        if output_type == "np":
            return image
        if output_type == "pil" and len(image.shape) == 4:
            return self.numpy_to_pil(image)
        return image

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """Convert images from a torch tensor to a numpy array.

        Args:
            images (torch.Tensor):
                The image tensor.

        Returns:
            np.ndarry: The image array.
        """
        x = images.permute(0, 2, 3, 4, 1) if images.dim() == 5 else images.permute(0, 2, 3, 1)
        return x.mul(127.5).add_(127.5).clamp(0, 255).byte().cpu().numpy()

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
        """Convert images from a numpy array to a list of PIL objects.

        Args:
            images (np.ndarray):
                The image array.

        Returns:
            List[PIL.Image.Image]: A list of PIL images.
        """
        images = images[None, ...] if images.ndim == 3 else images
        return [PIL.Image.fromarray(image) for image in images]

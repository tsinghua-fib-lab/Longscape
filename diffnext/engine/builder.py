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
# - Deleted funtions related to ema model.
# ------------------------------------------------------------------------
"""Engine builders."""

import torch

from diffnext.config import cfg
from diffnext.engine import lr_scheduler


def build_optimizer(params, **kwargs):
    """Build the optimizer."""
    args = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
    optimizer = kwargs.pop("optimizer", cfg.SOLVER.OPTIMIZER)
    args.update(kwargs)
    args.setdefault("betas", (0.9, cfg.SOLVER.ADAM_BETA2)) if "Adam" in optimizer else None
    return getattr(torch.optim, optimizer)(params, **args)


def build_lr_scheduler(**kwargs):
    """Build the LR scheduler."""
    args = {
        "lr_max": cfg.SOLVER.BASE_LR * (cfg.NUM_GPUS * cfg.TRAIN.BATCH_SIZE) / 256,
        "lr_min": cfg.SOLVER.MIN_LR,
        "warmup_steps": cfg.SOLVER.WARM_UP_STEPS,
        "warmup_factor": cfg.SOLVER.WARM_UP_FACTOR,
        "max_steps": cfg.SOLVER.MAX_STEPS,
    }
    policy = kwargs.pop("policy", cfg.SOLVER.LR_POLICY)
    args.update(kwargs)
    if policy == "steps_with_decay":
        args["decay_steps"] = cfg.SOLVER.DECAY_STEPS
        args["decay_gamma"] = cfg.SOLVER.DECAY_GAMMA
        return lr_scheduler.MultiStepLR(**args)
    elif policy == "cosine_decay":
        return lr_scheduler.CosineLR(**args)
    return lr_scheduler.ConstantLR(**args)


def build_tensorboard(log_dir):
    """Build the tensorboard."""
    from diffnext.utils.tensorboard import TensorBoard

    if TensorBoard.is_available():
        return TensorBoard(log_dir)
    return None

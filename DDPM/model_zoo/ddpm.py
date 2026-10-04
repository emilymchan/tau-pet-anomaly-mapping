# Adapted from MONAI and 2024-miccai-bercea-thor/projects/thor

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import tempfile
import time
from typing import Union, List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from net_utils.inferer import DiffusionInferer
from net_utils.diffusion_unet import DiffusionModelUNet
from net_utils.ddim_scheduler import DDIMScheduler
from net_utils.ddpm_scheduler import DDPMScheduler
from net_utils.simplex_noise import generate_noise


class DDPM(nn.Module):

    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
        method="autoDDPM",
        train_scheduler="ddpm",
        inference_scheduler="ddpm",
        inference_steps=1000,
        noise_level_recon=300,
        noise_level_inpaint=50,
        noise_type="gaussian",
        prediction_type="epsilon",
        resample_steps=4,
        masking_threshold=-1,
        threshold_low=1,
        threshold_high=10000,
        image_path="",
        eta: float = 0.0,
    ):
        super().__init__()
        self.unet = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
        )
        self.method = method
        self.noise_level_recon = noise_level_recon
        self.noise_level_inpaint = noise_level_inpaint
        self.prediction_type = prediction_type
        self.resample_steps = resample_steps
        self.masking_threshold = masking_threshold
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.image_path = image_path
        self.eta = eta

        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

        # set up scheduler and timesteps
        if train_scheduler == "ddim":
            print("****** DIFFUSION: Using DDIM Scheduler for training ******")
            self.train_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                noise_type=noise_type,
                prediction_type=prediction_type,
            )
        elif train_scheduler == "ddpm":
            print("****** DIFFUSION: Using DDPM Scheduler for training ******")
            self.train_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                noise_type=noise_type,
                prediction_type=prediction_type,
            )
        else:
            raise NotImplementedError(
                f"{train_scheduler} does is not implemented for {self.__class__}"
            )

        if inference_scheduler == "ddim":
            print("****** DIFFUSION: Using DDIM Scheduler for inference ******")
            self.inference_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                noise_type=noise_type,
                prediction_type=prediction_type,
            )
        else:
            print("****** DIFFUSION: Using DDPM Scheduler for inference ******")
            self.inference_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                noise_type=noise_type,
                prediction_type=prediction_type,
            )

        self.inference_scheduler.set_timesteps(inference_steps)
        # self.device = torch.device(
        #     "cuda:" + str(0) if torch.cuda.is_available() else "cpu"
        # )

        # self.ano_map = AnomalyMap()

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        noise_level: Union[int, None],
        save_intermediates: Union[bool, None] = False,
        intermediate_steps: Union[int, None] = 100,
        conditioning: Union[torch.Tensor, None] = None,
        verbose: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            noise_level: noising step until which noise is added before sampling
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        image = input_noise
        timesteps = self.inference_scheduler.get_timesteps(noise_level)
        # print(self.inference_scheduler.alphas_cumprod[:6])
        # breakpoint()
        if verbose:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            timesteps = torch.full(
                (input_noise.shape[0],), t, device=input_noise.device, dtype=torch.long
            )
            model_output = self.unet(image, timesteps=timesteps, context=conditioning)

            # model_output = self.unet(
            #     image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning
            # )

            # 2. compute previous image: x_t -> x_t-1
            image, orig_image = self.inference_scheduler.step(model_output, t, image, eta=self.eta)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(orig_image)
        if save_intermediates:
            return image, intermediates
        else:
            return image, None

    @torch.no_grad()
    # function to noise and then sample from the noise given an image to get healthy reconstructions of anomalous input images
    def sample_from_image(
        self,
        inputs: torch.Tensor,
        noise_level: Union[int, None] = 500,
        save_intermediates: Union[bool, None] = False,
        intermediate_steps: Union[int, None] = 100,
        conditioning: Union[torch.Tensor, None] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Sample to specified noise level and use this as noisy input to sample back.
        Args:
            inputs: input images, NxCxHxW[xD]
            noise_level: noising step until which noise is added before
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        # noise = generate_noise(self.train_scheduler.noise_type, inputs, noise_level)

        # t = torch.full((inputs.shape[0],), noise_level, device=inputs.device).long()
        # noised_image = self.train_scheduler.add_noise(
        #     original_samples=inputs, noise=noise, timesteps=t
        # )
        threshold = 0.26  # tau negativity threshold 1.3

        # inputs_filtered = inputs.clone()

        # mask = inputs > threshold
        # # mask = F.max_pool3d(mask.float(), kernel_size=3, stride=1, padding=1) > 0 # dilate mask to include neighboring voxels
        # noise_for_mask = torch.randn_like(inputs)

        # # replace suspected anomalous voxels with noise
        # inputs_filtered[mask] = noise_for_mask[mask]
        inputs_filtered = torch.clamp(inputs, max=threshold)

        noise = generate_noise(self.train_scheduler.noise_type, inputs_filtered, noise_level)

        t = torch.full((inputs.shape[0],), noise_level, device=inputs.device).long()

        noised_image = self.train_scheduler.add_noise(
            original_samples=inputs_filtered,
            noise=noise,
            timesteps=t
        )

        image, intermediates = self.sample(
            input_noise=noised_image,
            noise_level=noise_level,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            verbose=verbose,
        )
        return image, {"z": intermediates}

    def forward(self, inputs, noise=None, timesteps=None, condition=None):
        # only for torch_summary to work
        if noise is None:
            noise = torch.randn_like(inputs)
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.train_scheduler.num_train_timesteps,
                (inputs.shape[0],),
                device=inputs.device,
            ).long()

        noisy_image = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=timesteps
        )
        # noisy_image = inputs
        return self.unet(x=noisy_image, timesteps=timesteps, context=condition)

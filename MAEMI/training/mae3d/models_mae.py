# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import numpy as np
import torch
from torch import nn
import SimpleITK as sitk

from timm.models.vision_transformer import Block

from .util.helpers import PatchEmbed, get_3d_sincos_pos_embed


class MaskedAutoencoderViT_3D(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size: tuple,
        patch_size: tuple,
        in_chans,
        embed_dim=256 * 3,
        depth=24,
        num_heads=16,
        decoder_embed_dim=128 * 3,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        ignore_abnormal_tau=False,
        tau_threshold=0.37,
        ignore_abnormal_amyloid=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # check sizes and calc patch numbers
        self.img_size = np.array(img_size)
        self.patch_size = np.array(patch_size)
        assert not (
            self.img_size % self.patch_size
        ).any(), f"""
            img_size modulo patch_size error"""
        self.num_patches = (self.img_size // self.patch_size).astype(int)
        total_num_patches = np.prod(self.num_patches)
        self.in_chans = in_chans
        # --------------------------------------------------------------------------
        # Average pooling for binary mask - ignoring abnormal tau values
        self.avg_pool = nn.AvgPool3d(tuple(patch_size), stride=tuple(patch_size))
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, total_num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.ignore_abnormal_tau = ignore_abnormal_tau
        self.tau_threshold = tau_threshold
        self.ignore_abnormal_amyloid = ignore_abnormal_amyloid
        # if input_mask_file is not None:
        #     ds = sitk.ReadImage(input_mask_file)
        #     input_mask = sitk.GetArrayFromImage(ds)
        #     input_mask = np.transpose(input_mask, (2, 1, 0))
        #     self.input_mask = np.expand_dims(input_mask, axis=0)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, total_num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, np.prod(patch_size) * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.num_patches, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W, D)
        x: (N, L, patch_size**3 * C)
        """
        p = self.patch_size  # = (p_h, p_w, p_d)
        c = self.in_chans
        h, w, d = self.num_patches

        x = imgs.reshape(shape=(imgs.shape[0], c, h, p[0], w, p[1], d, p[2]))
        x = torch.einsum("nchpwqdo->nhwdpqoc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, np.prod(p) * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 * C)
        imgs: (N, C, H, W, D)
        """
        p = self.patch_size
        h, w, d = self.num_patches
        assert (
            h * w * d == x.shape[1]
        ), f"""
            size error: ({h}, {w}, {d}) vs {x.shape[1]}"""

        x = x.reshape(shape=(x.shape[0], h, w, d, p[0], p[1], p[2], -1))
        x = torch.einsum("nhwdpqoc->nchpwqdo", x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p[0], w * p[1], d * p[2]))
        return imgs

    def random_masking_ignore_abnormalTau(self, x, mask_ratio, x_pooled):
        """
        Perform per-sample random masking by per-sample shuffling, but don't
        keep any patches with an average tau that is abnormal.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        But this assumes only one channel input currently
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # Take each batch separately since will have different num of abnormal patches
        # Currently assumes only one input channel
        ids_shuffle = torch.zeros([N, L], device=x.device, dtype=torch.int64)
        ids_restore = torch.zeros([N, L], device=x.device, dtype=torch.int64)
        for b in range(N):
            # Get ids of abnormal patches
            # 0.26 for tau 1.3; 0.22 for tau 1.1
            abnormal_batch_ids = (x_pooled[b, :] > self.tau_threshold).nonzero().squeeze(1)
            ids = (x_pooled[b, :] <= self.tau_threshold).nonzero().squeeze(1)

            # Shuffle only non-abnormal patches
            noise = torch.rand(len(ids), device=x.device)  # noise in [0, 1]
            # if len_keep > len(ids):
            #     print("Warning: including some abnormal patches")

            # sort noise for each sample
            ids_shuffle_normal = ids[
                torch.argsort(noise)
            ]  # ascend: small is keep, large is remove
            if abnormal_batch_ids.shape[0] == 0:
                ids_shuffle[b, :] = ids_shuffle_normal
            else:
                ids_shuffle[b, :] = torch.concat(
                    [ids_shuffle_normal, abnormal_batch_ids]
                )
            ids_restore[b, :] = torch.argsort(ids_shuffle[b, :])

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_ignore_abnormalAmyloid(self, x, mask_ratio, mask_pooled):
        """
        Perform per-sample random masking by per-sample shuffling, but only use patches
        indicated with the input mask.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        But this assumes only one channel input currently
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # Take each batch separately since will have different num of abnormal patches
        # Currently assumes only one input channel
        ids_shuffle = torch.zeros([N, L], device=x.device, dtype=torch.int64)
        ids_restore = torch.zeros([N, L], device=x.device, dtype=torch.int64)
        for b in range(N):
            # Get ids of abnormal patches
            abnormal_batch_ids = (mask_pooled[b, :] < 0.2).nonzero().squeeze(1)
            ids = (mask_pooled[b, :] >= 0.2).nonzero().squeeze(1)

            # Shuffle only non-abnormal patches
            noise = torch.rand(len(ids), device=x.device)  # noise in [0, 1]
            # if len_keep > len(ids):
            #     print("Warning: including some abnormal patches")

            # sort noise for each sample
            ids_shuffle_normal = ids[
                torch.argsort(noise)
            ]  # ascend: small is keep, large is remove
            if abnormal_batch_ids.shape[0] == 0:
                ids_shuffle[b, :] = ids_shuffle_normal
            else:
                # take some of the abnormal patches and shuffle them into the abnormal patches
                # to help guide the recon, otherwise only using cerebellum is not idea
                keep_abnormal_ratio = 0.05
                noise_abnormal = torch.rand(len(abnormal_batch_ids), device=x.device)
                ids_shuffle_abnormal = abnormal_batch_ids[torch.argsort(noise_abnormal)]
                keep_abnormal_len = int(keep_abnormal_ratio * len(abnormal_batch_ids))
                keep_abnormal = ids_shuffle_abnormal[:keep_abnormal_len]
                abnormal_rest = ids_shuffle_abnormal[keep_abnormal_len:]

                ids_shuffle_normal = torch.concat([ids_shuffle_normal, keep_abnormal])
                noise = torch.rand(len(ids_shuffle_normal), device=x.device)
                ids_shuffle_normal = ids_shuffle_abnormal[torch.argsort(noise)]

                ids_shuffle[b, :] = torch.concat([ids_shuffle_normal, abnormal_rest])
            ids_restore[b, :] = torch.argsort(ids_shuffle[b, :])

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_ignore_abnormal_tau(self, x, mask_ratio):
        x_pooled = self.avg_pool(x)
        x_pooled = x_pooled.flatten(2).transpose(1, 2).squeeze(2)

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking_ignore_abnormalTau(
            x, mask_ratio, x_pooled
        )

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_ignore_abnormal_amyloid(self, x, mask_ratio, input_mask):
        mask_pooled = self.avg_pool(input_mask.float())
        mask_pooled = mask_pooled.flatten(2).transpose(1, 2).squeeze(2)

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking_ignore_abnormalAmyloid(
            x, mask_ratio, mask_pooled
        )

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        imgs,
        mask_ratio=0.75,
        input_mask=None,
    ):
        if self.ignore_abnormal_tau:
            latent, mask, ids_restore = self.forward_encoder_ignore_abnormal_tau(
                imgs, mask_ratio
            )
        elif self.ignore_abnormal_amyloid:
            latent, mask, ids_restore = self.forward_encoder_ignore_abnormal_amyloid(
                imgs, mask_ratio, input_mask
            )
        else:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*p*c]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# use decoder depth=4 from SpartiotemporalMAE
def mae_vit_base_dec384d8b(**kwargs):
    model = MaskedAutoencoderViT_3D(
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_base = mae_vit_base_dec384d8b  # decoder: 384 dim, 8 blocks

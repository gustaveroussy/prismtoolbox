import math
import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer

from .utils import retrieve_pretrained_weight_transforms_from_dict


class PathoDuet(VisionTransformer):
    def __init__(self, pretext_token=True, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # inserting a new token
        self.num_prefix_tokens += 1 if pretext_token else 0
        self.pretext_token = (
            nn.Parameter(torch.ones(1, 1, self.embed_dim)) if pretext_token else None
        )
        embed_len = (
            self.patch_embed.num_patches
            if self.no_embed_class
            else self.patch_embed.num_patches + 1
        )
        embed_len += 1 if pretext_token else 0
        self.embed_len = embed_len

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.pretext_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(
                6.0
                / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim)
            )
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "pretext_token"}

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((self.pretext_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((self.pretext_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def _ref_embed(self, ref):
        B, C, H, W = ref.shape
        ref = self.patch_embed.proj(ref)
        if self.patch_embed.flatten:
            ref = ref.flatten(2).transpose(1, 2)  # BCHW -> BNC
        ref = self.patch_embed.norm(ref)
        return ref

    def _pos_embed_with_ref(self, x, ref):
        pretext_tokens = self.pretext_token.expand(x.shape[0], -1, -1) * 0 + ref
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((pretext_tokens, x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((pretext_tokens, x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, ref=None):
        x = self.patch_embed(x)
        if ref is None:
            x = self._pos_embed(x)
        else:
            ref = self._ref_embed(ref).mean(dim=1, keepdim=True)
            x = self._pos_embed_with_ref(x, ref)
        # x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            raise NotImplementedError  # function not defined in original code
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, 1]
            )
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, ref=None):
        x_out = self.forward_features(x, ref)
        x = self.forward_head(x_out)
        return x

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert (
            self.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        assert (
            self.num_prefix_tokens == 2
        ), "Assuming two and only two tokens, [pretext][cls]"
        pe_token = torch.zeros([1, 2, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


def create_pathoduet_embedder(weights):
    model = PathoDuet(pretext_token=True, global_pool="avg")
    model.head = nn.Identity()
    state_dict, pretrained_transforms = retrieve_pretrained_weight_transforms_from_dict(
        "PathoDuet", weights
    )
    model.load_state_dict(state_dict, strict=False)
    return model, pretrained_transforms

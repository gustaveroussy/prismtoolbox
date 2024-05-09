import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
from huggingface_hub import hf_hub_download
from .coca_model import CoCa, resize_pos_embed

CFG_FILE = Path(__file__).parent / "config.json"


def read_state_dict(checkpoint_path: str, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path):
    state_dict = read_state_dict(checkpoint_path)
    resize_pos_embed(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)


class ConchModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.encode_image(x, proj_contrast=False, normalize=False)
        return x


def create_conch_embedder(_):
    checkpoint_path = "hf_hub:MahmoodLab/conch"
    with open(CFG_FILE, "r") as f:
        model_cfg = json.load(f)
    _ = model_cfg.pop("custom_text", None)

    model = CoCa(**model_cfg)

    checkpoint_path = hf_hub_download(
        checkpoint_path[len("hf_hub:") :], filename="pytorch_model.bin"
    )

    load_checkpoint(model, checkpoint_path)

    pretrained_transforms = transforms.Compose(
        [
            transforms.Resize(
                model.visual.image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(model.visual.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    model = ConchModel(model)

    return model, pretrained_transforms

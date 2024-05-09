import torch.nn as nn
from functools import partial
from transformers import AutoImageProcessor, ViTModel


class Phikon(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    def forward(self, x):
        x = x["pixel_values"].squeeze(1)
        output = self.model(pixel_values=x)
        return output.last_hidden_state[:, 0, :]


def create_phikon_embedder(_):
    model = Phikon()
    pretrained_transforms = partial(
        AutoImageProcessor.from_pretrained("owkin/phikon"), return_tensors="pt"
    )
    return model, pretrained_transforms

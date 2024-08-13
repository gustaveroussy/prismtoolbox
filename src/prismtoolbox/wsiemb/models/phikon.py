import logging
from functools import partial

import torch.nn as nn
from transformers import AutoImageProcessor, ViTModel

log = logging.getLogger(__name__)


class Phikon(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    def forward(self, x):
        x = x["pixel_values"].squeeze(1)
        output = self.model(pixel_values=x)
        return output.last_hidden_state[:, 0, :]


def create_phikon_embedder(weights=None):
    if weights is not None:
        log.warning("Weights are not used in this model, they will be ignored.")
    model = Phikon()
    pretrained_transforms = partial(
        AutoImageProcessor.from_pretrained("owkin/phikon"), return_tensors="pt"
    )
    return model, pretrained_transforms

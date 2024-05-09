import timm
import torch.nn as nn
import torch.hub as hub
from functools import partial
from torchvision.models import get_model_weights
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoConfig, AutoModel, AutoImageProcessor

from .utils import check_weight_in_dict, retrieve_pretrained_weight_transforms_from_dict


class GenericEmbedderTorchvision:
    def __init__(self, name, weights=None):
        self.name = name
        self.weights = weights
        self.torchvision_weights = [str(w) for w in get_model_weights(name)]
        (
            self.state_dict,
            self.pretrained_transforms,
        ) = self.retrieve_pretrained_state_dict_transforms()
        self.model = self.construct_model()

    def construct_model(self):
        model = hub.load("pytorch/vision", self.name.lower())
        model.fc = nn.Identity()

    def retrieve_pretrained_state_dict_transforms(self):
        if f"{self.name}_Weights.{self.weights}" in self.torchvision_weights:
            weights = hub.load(
                "pytorch/vision", "get_weight", name=f"{self.name}_Weights.{self.weights}"
            )
            state_dict, pretrained_transforms = (
                weights.get_state_dict(),
                weights.transforms(),
            )
        elif self.weights.startswith("https://"):
            state_dict = hub.load_state_dict_from_url(self.weights)
            pretrained_transforms = None
        elif check_weight_in_dict(self.weights):
            (
                state_dict,
                pretrained_transforms,
            ) = retrieve_pretrained_weight_transforms_from_dict(self.name, self.weights)
        else:
            raise ValueError(
                f"Pretrained weights {self.weights} not found for model {self.name}"
            )
        return state_dict, pretrained_transforms

    def clean_state_dict(self):
        if self.weights == "ciga":
            state_dict = self.state_dict.copy()
            for key in list(state_dict.keys()):
                if "fc" in key:
                    state_dict.pop(key)
                    continue
                state_dict[
                    key.replace("model.", "").replace("resnet.", "")
                ] = state_dict.pop(key)
            self.state_dict = state_dict

    def load_state_dict(self):
        self.clean_state_dict()
        self.model.load_state_dict(self.state_dict)


class GenericEmbedderTimm:
    def __init__(self, name, pretrained=True, weights=None, **kwargs):
        self.name = name
        self.pretrained = pretrained
        self.weights = weights
        self.model = self.construct_model(**kwargs)
        self.pretrained_transforms = self.retrieve_pretrained_transforms()

    def construct_model(self, **kwargs):
        model = timm.create_model(self.name, pretrained=self.pretrained, **kwargs)
        return model

    def retrieve_pretrained_transforms(self):
        if self.pretrained:
            pretrained_transform = create_transform(
                **resolve_data_config(self.model.pretrained_cfg, model=self.model)
            )
        elif self.weights:
            raise NotImplementedError
        else:
            raise ValueError(
                "Pretained is False and no weights provided. Cannot retrieve pretrained transforms."
            )
        return pretrained_transform


class TransformersBasedModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x["pixel_values"].squeeze(1)
        output = self.model(pixel_values=x)
        return output.last_hidden_state[:, 0, :]


class GenericEmbedderTransformers:
    def __init__(self, name, pretrained=True, weights=None, **kwargs):
        self.name = name
        self.pretrained = pretrained
        self.weights = weights
        self.config = self.retrieve_config()
        self.model = self.construct_model(**kwargs)
        self.pretrained_transforms = self.retrieve_pretrained_transforms()

    def retrieve_config(self):
        return AutoConfig.from_pretrained(self.name)

    def construct_model(self, **kwargs):
        if self.pretrained:
            return TransformersBasedModel(AutoModel.from_pretrained(self.name, **kwargs))
        elif self.weights:
            return TransformersBasedModel(
                AutoModel.from_pretrained(self.weights, **kwargs)
            )
        else:
            return TransformersBasedModel(AutoModel.from_config(self.config, **kwargs))

    def retrieve_pretrained_transforms(self):
        if self.pretrained:
            pretrained_transforms = partial(
                AutoImageProcessor.from_pretrained(self.name), return_tensors="pt"
            )
        elif self.weights:
            pretrained_transforms = partial(
                AutoImageProcessor.from_pretrained(self.weights), return_tensors="pt"
            )
        else:
            raise ValueError(
                "Pretained is False and no weights provided. Cannot retrieve pretrained transforms."
            )
        return pretrained_transforms


def create_torchvision_embedder(name, weights=None):
    embedder = GenericEmbedderTorchvision(name, weights)
    if weights is not None:
        embedder.load_state_dict()
    return embedder.model, embedder.pretrained_transforms


def create_timm_embedder(name, pretrained=True, weights=None, **kwargs):
    embedder = GenericEmbedderTimm(name, pretrained, weights, **kwargs)
    return embedder.model, embedder.pretrained_transforms


def create_transformers_embedder(name, pretrained=True, weights=None, **kwargs):
    embedder = GenericEmbedderTransformers(name, pretrained, weights, **kwargs)
    return embedder.model, embedder.pretrained_transforms

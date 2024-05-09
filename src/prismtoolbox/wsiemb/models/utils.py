import torch
import torchvision.transforms as transforms

weights_dict = {
    "ciga": {
        "url": "https://github.com/ozanciga/self-supervised-histopathology/releases/download/nativetenpercent/pytorchnative_tenpercent_resnet18.ckpt",
        "transforms": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
        "meta": {
            "input_space": "RGB",
            "input_range": [0, 1],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "image_size": [3, 224, 224],
            "arch": "ResNet18",
            "name": "ciga",
            "checkpoint_key": "state_dict",
        },
    },
    "pathoduet_HE": {
        "url": "https://github.com/loic-lb/HistoSSLModels/releases/download/PathoDuet/checkpoint_HE.pth",
        "transforms": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ],
        ),
        "meta": {
            "input_space": "RGB",
            "input_range": [0, 1],
            "mean": None,
            "std": None,
            "image_size": [3, 224, 224],
            "arch": "PathoDuet",
            "name": "pathoduet_HE",
            "checkpoint_key": None,
        },
    },
    "pathoduet_IHC": {
        "url": "https://github.com/loic-lb/HistoSSLModels/releases/download/PathoDuet/checkpoint_IHC.pth",
        "transforms": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ],
        ),
        "meta": {
            "input_space": "RGB",
            "input_range": [0, 1],
            "mean": None,
            "std": None,
            "image_size": [3, 224, 224],
            "arch": "PathoDuet",
            "name": "pathoduet_IHC",
            "checkpoint_key": None,
        },
    },
}


def check_weight_in_dict(weights):
    if weights in weights_dict.keys():
        return True
    else:
        return False


def retrieve_pretrained_weight_transforms_from_dict(name, weights):
    if weights in weights_dict.keys():
        weights = weights_dict[weights]
        if weights["meta"]["arch"] != name:
            raise ValueError(
                f"Provided weights are for {weights['meta']['arch']} and not for {name}"
            )
        state_dict = torch.hub.load_state_dict_from_url(weights["url"])
        checkpoint_key = weights["meta"]["checkpoint_key"]
        if checkpoint_key:
            state_dict = state_dict[checkpoint_key]
        return state_dict, weights["transforms"]
    else:
        raise ValueError(f"Invalid weights {weights} for model {name}")

import torch
import torch.nn as nn
from .architectures import define_G


class Generator(nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=1,
        ngf=64,
        netG="unet_256",
        use_dropout=True,
        dropout_value=0.5,
        norm="instance",
        r=60,
        segmentation=True,
    ):
        super(Generator, self).__init__()
        self.unet_model = define_G(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            netG=netG,
            use_dropout=use_dropout,
            dropout_value=dropout_value,
            norm=norm,
            init_type="normal",
            init_gain=0.02,
            bias_last_conv=True,
        )
        # Compression of the sigmoid.
        self.r = r
        self.sigmoid = lambda x: torch.sigmoid(self.r * x)
        self.segmentation = segmentation

    def forward(self, input):
        mask = self.unet_model(input)
        if self.segmentation:
            mask = self.sigmoid(mask)
        return mask


def create_sop_segmenter(netG="unet_256", norm="instance", weights=None):
    model = Generator(netG=netG, norm=norm)
    if weights:
        model.load_state_dict(torch.load(weights)["generator_ihc_to_mask_state_dict"])
    return model

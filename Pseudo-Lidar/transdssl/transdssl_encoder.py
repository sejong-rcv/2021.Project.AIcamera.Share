import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    SoftAttDepth,
    Interpolate,
    _make_scratch,
)

from transdssl.swintf import SwinTransformer

def _make_fusion_block(features, use_norm, onlyATT=False):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        layer_norm=use_norm,
        expand=False,
        align_corners=True,
    )


class TransDSSL_encoder(BaseModel):
    def __init__(
        self,
        head,
        infer=False,
        features=256,
        backbone="S",
        channels_last=False,
        use_norm=False,
    ):
        non_negative = True
        super(TransDSSL_encoder, self).__init__()
        self.infer=infer
        self.channels_last = channels_last


        ##Swin Small
        self.Swin=SwinTransformer(
                                embed_dim=96,
                                depths=[2, 2, 18, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                ape=False,
                                drop_path_rate=0.3,
                                patch_norm=True,
                                use_checkpoint=False
                            )

    def forward(self, x, epoch=0):
        _, layer_1, layer_2, layer_3, layer_4 = self.Swin(x)
        return layer_1, layer_2, layer_3, layer_4 

class TRANSDSSLEncoder(TransDSSL_encoder):
    def __init__(
        self, path=None,infer=False,   **kwargs    ):
        features = kwargs["features"] if "features" in kwargs else 256
        self.infer=infer
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        )

        super().__init__(head,infer, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, rgb, epoch=0):
        x=rgb
        features = super().forward(x, epoch=epoch)        

        return features

from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
)

import torch.nn as nn

from decoder import MAnetDecoder

# MAnet: https://ieeexplore.ieee.org/abstract/document/9201310

class MAnet(SegmentationModel):
    def __init__(
        self,
        encoder_depth: int = 5,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()

        self.encoder = get_encoder(
            "mit_b1",
            in_channels=in_channels,
            depth=encoder_depth,
            weights="imagenet",
        )

        self.decoder = MAnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
        )

        self.dropout = nn.Dropout2d(0.2)

        self.name = "manet-mit_b1"
        self.classification_head = None
        self.initialize()

    def forward(self, x):
        features = self.encoder(x)
        # features = self.dropout_forward_list(*features)

        decoder_output = self.decoder(*features)
        decoder_output = self.dropout(decoder_output)

        masks = self.segmentation_head(decoder_output)

        return masks

    def dropout_forward_list(self, *features):
        return [self.dropout(f) for f in features]
    

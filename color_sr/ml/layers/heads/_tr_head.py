from typing import Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from ..transformer import TransformerPredictor
from ..kan.sep_kan_layer import SepKANLayer1D
from ..transformer.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from ..transformer.position_encoding import PositionEmbeddingSine


class TRHead(nn.Module):

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: List[int] = [3, 2],
        feature_dim: int = 7,
        nheads: int = 8,
        nlayers: int = 6,
        dim_feedforward: int = 1024,
        normalize_before=False,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super(TRHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_dim = feature_dim
        self.dropout = dropout

        feature_dim = feature_dim * feature_dim + 1

        # decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=in_channels,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            normalize_before=False)
        encoder_norm = nn.LayerNorm(in_channels) if normalize_before else None
        self.prob_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=nlayers,
            norm=encoder_norm,
        )
        self.pos_embedding = nn.Embedding(feature_dim, in_channels)

        self.x_feed = nn.Linear(in_channels, out_channels[0])
        self.y_feed = nn.Linear(in_channels, out_channels[1])

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        B, C, N = x.shape

        x = x.permute(2, 0, 1)
        y = y.permute(2, 0, 1)

        xx = self.x_feed(x[0]).view((B, -1))
        yy = self.y_feed(y[0]).view((B, -1))


        pos_embed = self.pos_embedding.weight.unsqueeze(1).repeat(1, B, 1)

        y = self.prob_decoder(y, x, pos=pos_embed, query_pos=pos_embed)
        y = y[0]

        y = self.y_feed(y[0]).view((B, -1))

        return xx, yy

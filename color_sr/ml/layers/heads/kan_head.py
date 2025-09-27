from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from openeye.ml.layers.kan import SepKANLayer1D, SepKANLayer2D
from openeye.ml.layers.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder, PositionEmbeddingSine


class KANHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        kan_channels: int = 3,
        out_channels: int = 2,
        **kwargs: Any,
    ):
        super(KANHead, self).__init__()

        HIDDEN_DIM = 32
        LOCAL_DIM = 32
        DIM_FEEDFORWARD = 128
        NUM_LAYERS = 6
        NHEADS = 8
        DROPOUT = 0.1
        ACTIVATION = "relu"
        NORMALIZE_BEFORE = False

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kan_channels = kan_channels
        self.dropout = DROPOUT

        self.kan_decoder = SepKANLayer1D(in_channels=kan_channels,
                                         out_channels=out_channels,
                                         grid_size=5,
                                         spline_order=3,
                                         residual_std=0.1,
                                         grid_range=[0, 1])

        self.kan_size = self.kan_decoder.size

        # input
        if in_channels != HIDDEN_DIM:
            self.input_proj = nn.Conv2d(in_channels, HIDDEN_DIM, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()

        # weights
        encoder_layer = TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NHEADS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            activation=ACTIVATION,
            normalize_before=NORMALIZE_BEFORE)
        encoder_norm = nn.LayerNorm(HIDDEN_DIM) if NORMALIZE_BEFORE else None
        self.weight_encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                                 num_layers=NUM_LAYERS,
                                                 norm=encoder_norm)
        self.weight_proj = nn.Conv1d(HIDDEN_DIM, self.kan_size, kernel_size=1)

        # value
        encoder_layer = TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NHEADS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            activation=ACTIVATION,
            normalize_before=NORMALIZE_BEFORE)
        encoder_norm = nn.LayerNorm(HIDDEN_DIM) if NORMALIZE_BEFORE else None
        self.value_encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                                num_layers=NUM_LAYERS,
                                                norm=encoder_norm)
        self.value_proj = nn.Conv1d(HIDDEN_DIM,
                                    self.kan_channels,
                                    kernel_size=1)

        # decoder
        decoder_layer = TransformerDecoderLayer(HIDDEN_DIM, NHEADS,
                                                DIM_FEEDFORWARD, DROPOUT,
                                                ACTIVATION, NORMALIZE_BEFORE)
        decoder_norm = nn.LayerNorm(HIDDEN_DIM) if NORMALIZE_BEFORE else None
        self.prob_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=NUM_LAYERS,
            norm=decoder_norm,
        )
        self.prob_proj = nn.Conv1d(HIDDEN_DIM, out_channels, kernel_size=1)

        # positional encoding
        N_steps = HIDDEN_DIM // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.pos_embedding = nn.Embedding(7 * 7, HIDDEN_DIM)

        # feed
        self.feed = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features: torch.Tensor):

        features = self.input_proj(features)
        B, C, H, W = features.shape
        features = features.flatten(2).permute(2, 0, 1)

        pos_embed = self.pos_embedding.weight.unsqueeze(1).repeat(1, B, 1)

        weight_features = self.weight_encoder(features,
                                              src_key_padding_mask=None,
                                              pos=pos_embed)
        weight_features = F.dropout(weight_features, p=self.dropout)
        weights = self.weight_proj(weight_features.permute(1, 2, 0))

        value_features = self.value_encoder(features,
                                            src_key_padding_mask=None,
                                            pos=pos_embed)
        value_features = F.dropout1d(value_features, p=self.dropout)
        values = self.value_proj(value_features.permute(1, 2, 0))

        prob_features = self.prob_decoder(value_features,
                                          weight_features,
                                          pos=pos_embed,
                                          query_pos=pos_embed)

        prob = self.prob_proj(prob_features[-1].permute(1, 2, 0))
        prob = F.dropout1d(prob, p=self.dropout)
        prob = F.softmax(prob, dim=-1)

        maps = self.kan_decoder(F.sigmoid(values), weights)
        gaze = self.feed(maps * prob)
        """
        import torchvision
        grid = torchvision.utils.make_grid(prob.view(B,-1,7,7)[:,:1], nrow=1)
        torchvision.utils.save_image(grid, '/home/andk/temp/values.png')
        """

        return gaze.view((B, -1))

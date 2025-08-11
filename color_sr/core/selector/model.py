from ..config.model import ModelType, EncoderType, EncoderParams, HeadType, HeadParams
from ..config import Config
from typing import Union
from color_sr.ml.models import (GSRKanModel, ConvEncoder, UnetEncoder, DETrEncoder, CMEncoder, KANHead)


class ModelSelector:

    def encoder(config: Config) -> Union[UnetEncoder, DETrEncoder]:
        encoder_config = config.model.params.encoder
        match encoder_config.type:
            case EncoderType.cm:
                return CMEncoder(
                    in_channels=encoder_config.in_channels,
                    out_channels=encoder_config.out_channels,
                )
            case EncoderType.conv:
                head_config = config.model.params.head
                return ConvEncoder(
                    in_channels=encoder_config.in_channels,
                    w_channels=encoder_config.out_channels,
                    v_channels=head_config.in_channels,
                )
            case EncoderType.unet:
                return UnetEncoder(
                    in_channels=encoder_config.in_channels,
                    out_channels=encoder_config.out_channels,
                )
            case EncoderType.detr:
                head_config = config.model.params.head
                return DETrEncoder(
                    in_channels=encoder_config.in_channels,
                    mem_channels=encoder_config.out_channels,
                    hs_channels=head_config.in_channels,
                    d_model=encoder_config.d_model,
                    nhead=encoder_config.nhead,
                    num_decoder_layers=encoder_config.num_decoder_layers,
                    dim_feedforward=encoder_config.dim_feedforward,
                    dropout=encoder_config.dropout,
                )
            case _:
                raise ValueError(f'Unupported encoder type f{encoder_config.type}')

    def head(config: HeadParams) -> Union[KANHead]:
        head_config =config.model.params.head
        match head_config.type:
            case HeadType.kan:
                return KANHead(in_channels=head_config.in_channels,
                               out_channels=head_config.out_channels,
                               grid_size=head_config.grid_size,
                               spline_order=head_config.spline_order,
                               residual_std=head_config.residual_std,
                               grid_range=head_config.grid_range)
            case _:
                raise ValueError(f'Unupported head type f{head_config.type}')

    def select(config: Config) -> Union[GSRKanModel]:
        match config.model.type:
            case ModelType.gsr_kan:
                head = ModelSelector.head(config)
                encoder = ModelSelector.encoder(config)
                return GSRKanModel(encoder=encoder, head=head, scale = config.data.scale)
            case _:
                raise ValueError(f'Unupported model type f{config.model.type}')

# Python libraries
from typing import Any
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from openeye.ml.utils.gtools import cart_to_sph, sph_to_cart
import kornia.geometry.conversions as K


class GazeHead(nn.Module):

    def __init__(self, **kwargs: Any):
        super(GazeHead, self).__init__()

    def forward(
        self,
        gaze_quat: torch.Tensor,
        gaze_origin: torch.Tensor = None,
    ):
        gaze_matrix = K.quaternion_to_rotation_matrix(gaze_quat)
        gaze_vector = gaze_matrix[...,2]

        if gaze_origin is None:
            gaze_target = None
        else:
            z = gaze_origin[..., 2:] / gaze_vector[..., 2:]
            gaze_target = gaze_origin - gaze_vector * z

        return gaze_vector, gaze_target

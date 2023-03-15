from src.models.components.glow.act_norm import ActNorm
from src.models.components.glow.affine_coupling import AffineCoupling
from src.models.components.glow.base_flow_model import FlowSequential
from src.models.components.glow.invertible_conv import InvertibleConv


class FlowStep(FlowSequential):
    def __init__(self, in_channels: int, hid_channels: int):
        super().__init__(
            ActNorm(in_channels),
            InvertibleConv(in_channels),
            AffineCoupling(in_channels, hid_channels),
        )


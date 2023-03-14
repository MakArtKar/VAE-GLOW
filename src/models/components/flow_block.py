from src.models.components.flow_step import FlowSequential, FlowStep
from src.models.components.split import Split
from src.models.components.squeeze import Squeeze


class FlowBlock(FlowSequential):
    def __init__(self, depth: int, in_channels: int, hid_channels: int):
        super().__init__(
            Squeeze(),
            *[FlowStep(4 * in_channels, hid_channels) for _ in range(depth)],
            Split(),
        )

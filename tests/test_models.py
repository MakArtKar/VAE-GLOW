import pytest
import torch

from src.models.components.glow import ActNorm, AffineCoupling, FlowBlock, FlowStep, InvertibleConv, Split, Squeeze


def run_model(model, batch_size: int = 4, in_channels: int = 8, image_size: int = 16, **kwargs):
    device = 'cuda:0' if torch.cuda.device_count() > 0 else 'cpu'
    x = torch.randn(batch_size, in_channels, image_size, image_size).to(device)
    model.to(device)
    out = model(x, **kwargs)
    return x, out


def check_reconstruction(model, in_channels: int = 8, image_size: int = 16, atol=1e-6, **kwargs):
    x, out = run_model(model, in_channels=in_channels, image_size=image_size, **kwargs)
    out_z, z = out[0], out[1] if len(out) > 2 else None
    recon_x, _ = model(out_z, reverse=True, z=z, **kwargs)
    assert x.shape == recon_x.shape, f'{x.shape} != {recon_x.shape}'
    assert torch.allclose(x, recon_x, atol=atol), torch.nn.functional.l1_loss(x, recon_x, reduction='none').max()


@pytest.mark.parametrize("channels", [8])
def test_act_norm(channels: int):
    act_norm = ActNorm(channels)
    check_reconstruction(act_norm, in_channels=channels)


@pytest.mark.parametrize("in_channels,hid_channels", [(4, 8)])
def test_affine_coupling(in_channels: int, hid_channels: int):
    affine_coupling = AffineCoupling(in_channels, hid_channels)
    check_reconstruction(affine_coupling, in_channels=in_channels)


@pytest.mark.parametrize("in_channels", [8])
def test_invertible_conv(in_channels: int):
    invertible_conv = InvertibleConv(in_channels)
    check_reconstruction(invertible_conv, in_channels=in_channels, atol=2e-6)


def test_squeeze():
    squeeze = Squeeze()
    check_reconstruction(squeeze)


def test_split():
    split = Split()
    check_reconstruction(split)


@pytest.mark.parametrize("in_channels,hid_channels", [(128, 512)])
def test_flow_step(in_channels: int, hid_channels: int):
    flow_step = FlowStep(in_channels, hid_channels)
    check_reconstruction(flow_step, in_channels=in_channels, atol=1e-3)


@pytest.mark.parametrize("depth,in_channels,hid_channels", [(32, 128, 512)])
def test_flow_block(depth: int, in_channels: int, hid_channels: int):
    flow_block = FlowBlock(depth, in_channels, hid_channels)
    check_reconstruction(flow_block, in_channels=in_channels, atol=2e-2)

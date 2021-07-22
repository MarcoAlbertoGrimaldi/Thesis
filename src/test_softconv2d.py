from pytest import fixture
import torch

from src.softconv2d import SoftConv2d


@fixture
def x():
    return torch.randn([4, 3, 32, 32], requires_grad=True)


@fixture
def m():
    m = torch.zeros([1, 1, 32, 32])
    m[:, :, 14:27, 13:22] = torch.randint(0, 10, [1, 1, 13, 9]) / 10
    return m


def test_forward(x, m):
    conv = SoftConv2d(3, 3, 3, 1, 'zero', 1, 1)
    y, nm = conv(x, m)
    assert y.shape == x.shape and nm.shape == m.shape


def test_stride(x, m):
    conv = SoftConv2d(3, 3, 3, 2, 'zero', 1, 1)
    y, nm = conv(x, m)
    assert y.shape == torch.Size([4, 3, 16, 16]) and nm.shape == torch.Size([1, 1, 16, 16])


def test_dilation(x, m):
    conv = SoftConv2d(3, 3, 3, 2, 'zero', 2, 2)
    y, nm = conv(x, m)
    assert y.shape == torch.Size([4, 3, 16, 16]) and nm.shape == torch.Size([1, 1, 16, 16])
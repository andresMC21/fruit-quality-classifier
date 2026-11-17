import pytest
import torch

from src.models.cnn import augment, build_cnn, count_parameters


def test_output_shape():
    m = build_cnn().eval()
    assert m(torch.rand(4, 3, 128, 128)).shape == (4, 3)


def test_param_count():
    assert count_parameters(build_cnn()) == 101923


def test_augment_shape_range():
    torch.manual_seed(0)
    x = torch.rand(5, 3, 32, 32)
    for _ in range(20):
        y = augment(x)
        assert y.shape == x.shape
        assert y.min() >= 0 and y.max() <= 1


def test_bad_channels():
    with pytest.raises(ValueError):
        build_cnn(input_shape=(128, 128, 1))

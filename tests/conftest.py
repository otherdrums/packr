"""Shared fixtures for PackR tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

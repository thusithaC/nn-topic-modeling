import pytest
from utils.data_loader import TwitterDataset


def test_create_instance():
    dataloader = TwitterDataset("TRAIN", 10)
    dataloader.init()
    assert dataloader is not None
    assert len(dataloader.tweets) == 10
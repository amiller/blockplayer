from blockplayer import dataset
import os
import sys


def test_download():
    dataset.download()


def test_dataset():
    dataset.load_random_dataset()

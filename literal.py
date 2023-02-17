from dataclasses import dataclass


@dataclass
class Folder:
    data = "./data/"
    """
    data folder의 상대경로
    """
    data_preprocess = "./data/aug_preprocess/"
    """
    data/preprocess/의 상대경로
    """
    data_raw = "./data/raw/"
    """
    data/raw/의 상대경로
    """
    data_aug_preprocess = "./data/train_aug/"


@dataclass
class RawDataColumns:
    img_path = "img_path"
    """img_path"""
    label = "label"
    """label"""
    length = "length"
    """length"""
    seq_probs = "seq_probs"
    """seq_probs"""


@dataclass
class DatasetColumns:
    pixel_values = "pixel_values"
    """pixel_values"""
    labels = "labels"
    """labels"""

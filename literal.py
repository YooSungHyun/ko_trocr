from dataclasses import dataclass


@dataclass
class Folder:
    data = "./data/"
    """
    data folder의 상대경로
    """
    data_preprocess = "./data/preprocess/"
    """
    data/preprocess/의 상대경로
    """


@dataclass
class RawDataColumns:
    img_path = "img_path"
    """img_path"""
    label = "label"
    """label"""
    length = "length"
    """length"""


@dataclass
class DatasetColumns:
    pixel_values = "pixel_values"
    """pixel_values"""
    labels = "labels"
    """labels"""

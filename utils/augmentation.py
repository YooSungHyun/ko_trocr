import random

from straug.blur import DefocusBlur, GaussianBlur, GlassBlur, MotionBlur
from straug.camera import Brightness, Contrast, Pixelate
from straug.geometry import Rotate
from straug.noise import GaussianNoise
from straug.weather import Fog, Frost

from literal import DatasetColumns

string_augs = [
    GaussianBlur(),
    DefocusBlur(),
    MotionBlur(),
    GaussianNoise(),
    Frost(),
    Fog(),
    Contrast(),
    Brightness(),
    Pixelate(),
]


def augmentation(raw):
    random_choice = random.random()
    if random_choice > 0.5:
        rand_idx = random.randint(0, len(string_augs) - 1)
        raw[DatasetColumns.pixel_values] = [
            string_augs[rand_idx](image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
        ]
        random_choice2 = random.random()
        if random_choice2 > 0.5:
            raw[DatasetColumns.pixel_values] = [
                Rotate()(image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
            ]
    return raw

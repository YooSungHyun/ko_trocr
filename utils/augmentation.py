import random

from straug.blur import DefocusBlur, GaussianBlur, GlassBlur, MotionBlur
from straug.camera import Brightness, Contrast, Pixelate
from straug.geometry import Rotate
from straug.noise import GaussianNoise, ImpulseNoise, ShotNoise, SpeckleNoise
from straug.pattern import Grid, HGrid, VGrid
from straug.process import Color, Sharpness
from straug.weather import Fog, Frost, Rain, Snow
from torchvision.transforms import Compose

from literal import DatasetColumns

aug1 = Compose([Snow(),GlassBlur(),])
aug2 = Compose([GlassBlur(),DefocusBlur()])
aug3 = Compose([Pixelate(),GlassBlur()])
aug4 = Rotate()
string_augs = [
    aug1,
    aug2,
    aug3,
    aug4,
]


def augmentation(raw):
    random_choice = random.random()
    if random_choice < 0.8:
        rand_idx = random.randint(0, len(string_augs) - 1)
        raw[DatasetColumns.pixel_values] = [
            string_augs[rand_idx](image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
        ]
    return raw

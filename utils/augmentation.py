import random

import cv2
import numpy as np
from PIL import Image
from straug.blur import DefocusBlur, GaussianBlur, GlassBlur, MotionBlur
from straug.camera import Brightness, Contrast, JpegCompression, Pixelate
from straug.geometry import Rotate
from straug.noise import GaussianNoise, ImpulseNoise, ShotNoise, SpeckleNoise
from straug.pattern import Grid, HGrid, VGrid
from straug.process import AutoContrast, Color, Sharpness
from straug.weather import Fog, Frost, Rain, Snow
from torchvision.transforms import ColorJitter, Compose, Grayscale, ToTensor

from literal import DatasetColumns


class CustomRotate(Rotate):
    def __init__(self, square_side=224, rng=None):
        super().__init__(square_side=square_side)

    def __call__(self, img, iscurve=False, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size

        if h != self.side or w != self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        b = [15,30]
        if mag < 0 or mag >= len(b):
            index = 1
        else:
            index = mag
        rotate_angle = b[index]

        angle = self.rng.uniform(rotate_angle - 20, rotate_angle)
        if self.rng.uniform(0, 1) < 0.5:
            angle = -angle

        img = img.rotate(angle=angle, resample=Image.BICUBIC, expand=not iscurve)
        img = img.resize((w, h), Image.BICUBIC)

        return img

# DefocusBlur, GaussianBlur, GlassBlur
aug1 = Compose([Pixelate(),GlassBlur()]) # 찌그러짐
aug2 = Compose([GlassBlur(),DefocusBlur()])
aug3 = Compose([GaussianNoise(),Grayscale(num_output_channels=3)]) # 어린아이 노이즈
aug4 = AutoContrast()
rotate = CustomRotate()
string_augs = [
    aug1,
    aug2,
    aug3,
    aug4,
    rotate
]


def augmentation(raw):
    if random.random() < 0.9:
        rand_idx = random.randint(0, len(string_augs) - 1)
        raw[DatasetColumns.pixel_values] = [
            string_augs[rand_idx](image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
        ]
    # if random.random() < 0.75:
    #     rand_idx = random.randint(0, len(string_augs) - 1)
    #     raw[DatasetColumns.pixel_values] = [
    #         rotate(image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
    #     ]
    return raw



# 92.7


# aug1 = Compose([Snow(),GlassBlur(),])
# aug2 = Compose([GlassBlur(),DefocusBlur()])
# aug3 = Compose([Pixelate(),GlassBlur()])
# aug4 = Rotate()
# string_augs = [
#     aug1,
#     aug2,
#     aug3,
#     aug4,
# ]


# def augmentation(raw):
#     random_choice = random.random()
#     if random_choice < 0.8:
#         rand_idx = random.randint(0, len(string_augs) - 1)
#         raw[DatasetColumns.pixel_values] = [
#             string_augs[rand_idx](image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
#         ]
#     return raw


# 테스트중 멈춤(new-aug(custom rotate, grayscale noise))
# aug1 = Compose([JpegCompression(),GlassBlur()])
# aug2 = Compose([GlassBlur(),DefocusBlur()])
# aug3 = Compose([Pixelate(),GlassBlur()])
# aug4 = Compose([GaussianNoise(),Grayscale(num_output_channels=3)]) # 어린아이 노이즈
# aug4 = Compose([ShotNoise(),Grayscale(num_output_channels=3)])
# aug5 = Compose([SpeckleNoise(),Grayscale(num_output_channels=3)])
# #aug5 = CustomRotate()
# string_augs = [
#     aug1,
#     aug2,
#     aug3,
#     aug4,
#     aug5
# ]


# def augmentation(raw):
#     if random.random() < 0.8:
#         rand_idx = random.randint(0, len(string_augs) - 1)
#         raw[DatasetColumns.pixel_values] = [
#             string_augs[rand_idx](image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
#         ]
#     if random.random() < 0.5:
#         rand_idx = random.randint(0, len(string_augs) - 1)
#         raw[DatasetColumns.pixel_values] = [
#             CustomRotate()(image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
#         ]
#     return raw
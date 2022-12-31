import random

from straug.blur import DefocusBlur, GaussianBlur, GlassBlur, MotionBlur
from straug.camera import Brightness, Contrast, Pixelate
from straug.noise import GaussianNoise
from straug.weather import Fog, Frost

from literal import DatasetColumns
import numpy as np
from PIL import Image


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


class Rotate:
    def __init__(self, square_side=224, rng=None):
        self.side = square_side
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img1, img2, iscurve=False, mag=-1, prob=1.0):
        if self.rng.uniform(0, 1) > prob:
            return img1, img2

        w, h = img1.size
        w2, h2 = img2.size

        if h != self.side or w != self.side:
            img1 = img1.resize((self.side, self.side), Image.Resampling.BICUBIC)
        if h2 != self.side or w2 != self.side:
            img2 = img2.resize((self.side, self.side), Image.Resampling.BICUBIC)
        b = [15, 30, 45]
        if mag < 0 or mag >= len(b):
            index = 1
        else:
            index = mag
        rotate_angle = b[index]

        angle = self.rng.uniform(rotate_angle - 20, rotate_angle)
        if self.rng.uniform(0, 1) < 0.5:
            angle = -angle

        img1 = img1.rotate(angle=angle, resample=Image.Resampling.BICUBIC, expand=not iscurve)
        img1 = img1.resize((w, h), Image.Resampling.BICUBIC)
        img2 = img2.rotate(angle=angle, resample=Image.Resampling.BICUBIC, expand=not iscurve)
        img2 = img2.resize((w, h), Image.Resampling.BICUBIC)

        return img1, img2


def augmentation(raw):
    white_rgb = [255, 255, 255]
    random_choice = random.random()
    if random_choice > 0.5:
        rand_idx = random.randint(0, len(string_augs) - 1)
        raw[DatasetColumns.pixel_values] = [
            string_augs[rand_idx](image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
        ]
    random_choice2 = random.random()
    if random_choice2 > 0.3:
        rotate_func = Rotate()
        for image in raw[DatasetColumns.pixel_values]:
            edge_rgb_mean = np.around(
                np.mean(
                    np.stack(
                        (
                            np.array(image)[0][0],
                            np.array(image)[0][-1],
                            np.array(image)[-1][0],
                            np.array(image)[-1][-1],
                        )
                    ),
                    axis=0,
                )
            )
            angle_chk_image = Image.fromarray(np.full((image.size[0], image.size[1], 3), white_rgb, dtype=np.uint8))
            rot_image, rot_angle_chk_image = rotate_func(image.convert("RGB"), angle_chk_image)
            rot_image = np.array(rot_image)
            row_col_space = np.where(np.all(np.array(rot_angle_chk_image) != np.array(white_rgb), axis=-1))
            for row, col in zip(row_col_space[0], row_col_space[1]):
                rot_image[row][col] = white_rgb
            raw[DatasetColumns.pixel_values] = [Image.fromarray(rot_image)]
    return raw

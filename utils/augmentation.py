import random

from straug.blur import DefocusBlur, GaussianBlur, MotionBlur, GlassBlur
from skimage.filters import gaussian
from straug.camera import Pixelate, JpegCompression
from straug.noise import GaussianNoise
from straug.geometry import Rotate

from literal import DatasetColumns
import numpy as np
from PIL import Image
from torchvision.transforms import Compose
import cv2


class Custom_Rotate(Rotate):
    def __call__(self, img1, img2, iscurve=False, mag=-1, prob=1.0):
        """b변수에 45를 추가하면, 45도까지 기울어질 수 있는 경우의 수가 추가됩니다."""
        if self.rng.uniform(0, 1) > prob:
            return img1, img2

        w, h = img1.size
        w2, h2 = img2.size

        if h != self.side or w != self.side:
            img1 = img1.resize((self.side, self.side), Image.Resampling.BICUBIC)
        if h2 != self.side or w2 != self.side:
            img2 = img2.resize((self.side, self.side), Image.Resampling.BICUBIC)
        b = [20]
        if len(b) == 1:
            index = 0
        elif mag < 0 or mag >= len(b):
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


class Custom_GlassBlur(GlassBlur):
    def __call__(self, img, mag=-1, prob=1.0):
        """모든 로직은 동일합니다 FutureWarning을 위한 multichannel -> channel_axis 변경"""
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        # c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        c = [(0.45, 1, 1), (0.6, 1, 2), (0.75, 1, 2)]  # , (1, 2, 3)] #prev 2 levels only
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag

        c = c[index]

        img = np.uint8(gaussian(np.asarray(img) / 255.0, sigma=c[0], channel_axis=True) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for y in range(h - c[1], c[1], -1):
                for x in range(w - c[1], c[1], -1):
                    dx, dy = self.rng.integers(-c[1], c[1], size=(2,))
                    y_prime, x_prime = y + dy, x + dx
                    # swap
                    img[y, x], img[y_prime, x_prime] = img[y_prime, x_prime], img[y, x]

        img = np.clip(gaussian(img / 255.0, sigma=c[0], channel_axis=True), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


shake_aug = [DefocusBlur(), MotionBlur(), Custom_GlassBlur()]
resolution_aug = [Pixelate(), GaussianBlur(), GaussianNoise(), JpegCompression()]
# contrast = Contrast()
rotate = Custom_Rotate(square_side=192)


def augmentation(raw):
    white_rgb = [255, 255, 255]
    rotate_random = random.random()
    if rotate_random > 0.5:
        rotate_result = list()
        for image in raw[DatasetColumns.pixel_values]:
            angle_chk_image = Image.fromarray(np.full((image.size[0], image.size[1], 3), white_rgb, dtype=np.uint8))
            rot_image, rot_angle_chk_image = rotate(image.convert("RGB"), angle_chk_image)
            rot_image = np.array(rot_image)
            row_col_space = np.where(np.all(np.array(rot_angle_chk_image) != np.array(white_rgb), axis=-1))
            for row, col in zip(row_col_space[0], row_col_space[1]):
                rot_image[row][col] = white_rgb
            rotate_result.append(Image.fromarray(rot_image))
        raw[DatasetColumns.pixel_values] = rotate_result
    aug_choice = random.random()
    if aug_choice > 0.2:
        shake_idx = random.randint(0, len(shake_aug) - 1)
        resolution_idx = random.randint(0, len(resolution_aug) - 1)
        string_augs = Compose([resolution_aug[resolution_idx], shake_aug[shake_idx]])
        raw[DatasetColumns.pixel_values] = [
            string_augs(image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
        ]
    return raw


def sharpening(raw):
    sharpening_result = list()
    for image in raw[DatasetColumns.pixel_values]:
        img = np.array(image)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_f = img_ycrcb[:, :, 0].astype(np.float32)
        img_blr = cv2.GaussianBlur(img_f, (0, 0), 2.0)
        img_ycrcb[:, :, 0] = np.clip(2.0 * img_f - img_blr, 0, 255).astype(np.uint8)
        dst = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
        sharpening_result.append(Image.fromarray(dst))
    raw[DatasetColumns.pixel_values] = [Image.fromarray(dst)]
    return raw

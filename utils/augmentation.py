import random
from typing import Any, Callable, Dict, Tuple
import pandas as pd
import numpy as np
from PIL import Image
from skimage.filters import gaussian
from straug.blur import DefocusBlur, GaussianBlur, GlassBlur, MotionBlur
from straug.camera import JpegCompression, Pixelate
from straug.geometry import Rotate
from straug.noise import GaussianNoise
from torchvision.transforms import Compose, Grayscale
import albumentations as A
from literal import DatasetColumns
import math
import cv2
import unicodedata
from tacobox import Taco

random.seed(42)


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


class Augmentator:
    def __init__(
        self, aug_with_compose_prob: float = 0.8, rotation_prob: float = 0.5, rotation_square_side: int = 192
    ):
        self.__white_rgb = [255, 255, 255]
        self.shake_aug = [DefocusBlur(), MotionBlur(), Custom_GlassBlur()]
        self.resolution_aug = [
            Pixelate(),
            GaussianBlur(),
            Compose([GaussianNoise(), Grayscale(num_output_channels=3)]),
            JpegCompression(),
        ]
        self.rotate = Custom_Rotate(square_side=rotation_square_side)
        self.__rotation_prob = rotation_prob
        self.__aug_with_compose_prob = aug_with_compose_prob

    def augmentation(self, raw: Dict[str, Any]):
        if random.random() < self.__rotation_prob:
            raw[DatasetColumns.pixel_values] = self.__rotate_with_white_background(raw=raw)
        if random.random() < self.__aug_with_compose_prob:
            string_augs = self.__aug_with_compose()
            raw[DatasetColumns.pixel_values] = [
                string_augs(image.convert("RGB")) for image in raw[DatasetColumns.pixel_values]
            ]
        return raw

    def __rotate_with_white_background(self, raw: Dict[str, Any]) -> Image:
        rotate_result = list()
        for image in raw[DatasetColumns.pixel_values]:
            angle_chk_image = Image.fromarray(
                np.full((image.size[0], image.size[1], 3), self.__white_rgb, dtype=np.uint8)
            )
            rot_image, rot_angle_chk_image = self.rotate(image.convert("RGB"), angle_chk_image)
            rot_image = np.array(rot_image)
            row_col_space = np.where(np.all(np.array(rot_angle_chk_image) != np.array(self.__white_rgb), axis=-1))
            for row, col in zip(row_col_space[0], row_col_space[1]):
                rot_image[row][col] = self.__white_rgb
            rotate_result.append(Image.fromarray(rot_image))
        return rotate_result

    def __aug_with_compose(self) -> Callable:
        """
        Edit by Cleaning
        """
        resolution_idx = random.randint(0, len(self.resolution_aug) - 1)
        shake_idx = random.randint(0, len(self.shake_aug) - 1)
        string_augs = Compose([self.resolution_aug[resolution_idx], self.shake_aug[shake_idx]])
        return string_augs


class ChildWrittenAugmentator:
    def __init__(self, aug_with_compose_prob: float = 0.8):
        self.__aug_with_compose_prob = aug_with_compose_prob

    def augmentation(self, raw: Dict[str, Any]):
        augs = list()
        max_holes = random.randint(1, 5)
        height_size = np.array(raw[DatasetColumns.pixel_values][0]).shape[0]
        min_height_size = math.ceil(height_size / 10)
        max_height_size = min_height_size * 2
        max_height = random.randint(min_height_size, max_height_size)

        width_size = np.array(raw[DatasetColumns.pixel_values][0]).shape[1]
        min_width_size = math.ceil(width_size / 5)
        max_width = random.randint(min_width_size, width_size)

        black_cutout = A.CoarseDropout(
            p=1,
            max_height=max_height,
            max_width=max_width,
            min_holes=1,
            max_holes=max_holes,
            min_height=1,
            min_width=1,
            fill_value=0,
        )
        augs.append(black_cutout)
        if random.random() < self.__aug_with_compose_prob:
            augmentation_idx = random.randint(0, len(augs) - 1)
            raw[DatasetColumns.pixel_values] = [
                Image.fromarray(augs[augmentation_idx](image=np.array(image.convert("RGB")))["image"])
                for image in raw[DatasetColumns.pixel_values]
            ]
        return raw


class RandomConcat:
    def __init__(
        self,
        train_csv_path: str,
        image_size: Tuple[int, int],
        aug_with_compose_prob: float = 0.8,
        vconcat_flag: bool = False,
        hconcat_flag: bool = True,
    ):
        self.image_size = image_size
        self.train_csv = pd.read_csv(train_csv_path)
        self.__aug_with_compose_prob = aug_with_compose_prob
        self.vconcat_flag = vconcat_flag
        self.hconcat_flag = hconcat_flag
        self.max_label_len = max(self.train_csv["label"].apply(lambda x: len(x)))

    def augmentation(self, raw: Dict[str, Any]):
        labels = raw["labels"]
        images = raw["pixel_values"]
        image_results = list()
        label_results = list()
        for idx, ori_img in enumerate(images):
            np_ori_img = np.array(ori_img)
            label = unicodedata.normalize("NFC", labels[idx])
            while (np_ori_img.shape[0] > 50 and np_ori_img.shape[1] > 50) and len(label) <= self.max_label_len // 2:
                concat_idx = random.randrange(0, len(self.train_csv))
                concat_csv_row = self.train_csv.iloc[concat_idx]
                concat_label = concat_csv_row["label"]
                concat_img_path = concat_csv_row["img_path"]
                concat_img = cv2.imread(concat_img_path)
                if (
                    concat_img.shape[0] <= 50
                    or concat_img.shape[1] <= 50
                    or len(concat_label) > self.max_label_len // 2
                ):
                    continue
                # 둘다 True인 경우 hconcat 결과로 반영되도록 의도함. vconcat은 해도 되는게 맞을지 모르겠음
                if self.vconcat_flag:
                    resize_ori_img = self.image_processor.resize(
                        np_ori_img, size=[self.image_size[1], np_ori_img.shape[0]]
                    )
                    resize_concat_img = self.image_processor.resize(
                        concat_img, size=[self.image_size[1], concat_img.shape[0]]
                    )
                    if random.random() < 0.5:
                        img_result = cv2.vconcat([resize_ori_img, resize_concat_img])
                        label_result = label + concat_label
                    else:
                        img_result = cv2.vconcat([resize_concat_img, resize_ori_img])
                        label_result = concat_label + label
                    img_result = cv2.resize(img_result, self.image_size)
                if self.hconcat_flag:
                    resize_ori_img = cv2.resize(np_ori_img, (np_ori_img.shape[1], self.image_size[0]))
                    resize_concat_img = cv2.resize(concat_img, (np_ori_img.shape[1], self.image_size[0]))
                    if random.random() < 0.5:
                        img_result = cv2.hconcat([resize_ori_img, resize_concat_img])
                        label_result = label + concat_label
                    else:
                        img_result = cv2.hconcat([resize_concat_img, resize_ori_img])
                        label_result = concat_label + label
                    img_result = cv2.resize(img_result, self.image_size)
                if label != concat_label:
                    image_results.append(Image.fromarray(img_result))
                    label_results.append(unicodedata.normalize("NFKD", label_result))
                    break
        if (len(image_results) > 0 and len(label_results) > 0) and random.random() < self.__aug_with_compose_prob:
            raw[DatasetColumns.pixel_values] = image_results
            raw[DatasetColumns.labels] = label_results
        return raw


class TACo:
    def __init__(self):
        # creating taco object for augmentation (checkout Easter2.0 paper)
        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10,
        )

    def apply_taco_augmentations(self, input_img):
        if random.random() <= 0.15:
            augmented_img = self.mytaco.apply_vertical_taco(input_img, corruption_type="random")
        else:
            augmented_img = input_img
        if random.random() <= 0.15:
            augmented_img = self.mytaco.apply_vertical_taco(augmented_img, corruption_type="random")
        else:
            augmented_img = input_img
        return augmented_img

    def augmentation(self, raw: Dict[str, Any]):
        for image in raw[DatasetColumns.pixel_values]:
            if image.size[0] > 150 and image.size[1] > 100:
                taco_image = self.apply_taco_augmentations(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY))
                rgb_taco_image = cv2.cvtColor(taco_image.astype(np.float32), cv2.COLOR_GRAY2RGB)
                raw[DatasetColumns.pixel_values] = [Image.fromarray(rgb_taco_image.astype(np.uint8))]
        return raw

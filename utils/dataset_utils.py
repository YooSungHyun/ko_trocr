import json
import os
import re
from unicodedata import normalize

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
import cv2
from tqdm import tqdm

from literal import DatasetColumns, Folder, RawDataColumns
from typing import Tuple
from collections import deque


def to_subchar(string: str) -> str:
    """
    유니코드 NFKD로 정규화
    """
    return normalize("NFKD", string)


def clean_text(text: str) -> str:
    """
    텍스트의 자모 및 공백을 삭제
    ex) "바ㄴ나 나" -> "바나나"
    """
    text = re.sub(r"[^가-힣]", "", text)
    return text


def preprocess_img_path(path: str):
    if not path.startswith(Folder.data):
        path = path.replace("./", Folder.data)
    return path


def get_dataset(csv_path: os.PathLike, is_sub_char=True) -> Dataset:
    """
    csv의 경로를 입력받아 Dataset을 리턴
    is_sub_char: "snunlp/KR-BERT-char16424"와 같은 sub_char tokenizer일 경우 True, 일반적인 토크나이저일 경우에는 False
    feature: pixel_values(PIL image), labels(str)

    """
    df = pd.read_csv(csv_path)

    data_dict = {DatasetColumns.pixel_values: df[RawDataColumns.img_path].tolist()}

    if RawDataColumns.label in df.columns:
        if is_sub_char:
            df[RawDataColumns.label] = df[RawDataColumns.label].apply(to_subchar)
        data_dict[DatasetColumns.labels] = df[RawDataColumns.label].tolist()

    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.cast_column(DatasetColumns.pixel_values, datasets.Image())
    return dataset


def vector_to_text_jpg(array, width, height, save_folder_name, img_name):
    save_path = f"./{save_folder_name}/{img_name}.png"
    if os.path.exists(save_path) == False:
        px = 1 / plt.rcParams["figure.dpi"]  # inch to pixel
        fig = plt.figure(frameon=False, figsize=(width * px, height * px))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        for array1 in array:
            x1, y1 = array1
            x1, y1 = np.array(x1), np.array(y1) * -1
            ax.plot(x1, y1, c="black")

            plt.xticks([])
            plt.yticks([])

        plt.savefig(save_path)
        plt.close()
    return save_path


def vector_to_save_image(json_path_list, save_folder_name):
    if os.path.isdir(f"./{save_folder_name}") == False:
        os.mkdir(f"./{save_folder_name}")
    img_path_list, label_list, sample_id_list = [], [], []
    for path in tqdm(json_path_list):
        with open(path) as f:
            sample_data = json.load(f)
            sample_id = sample_data["id"]
            sample_id_list.append(sample_id)
            width = sample_data["width"]
            height = sample_data["height"]
            vector = sample_data["strokes_converted"]
            if "label" in sample_data:
                label = sample_data["label"]
                label_list.append(label)
            img_path = vector_to_text_jpg(vector, width, height, save_folder_name, sample_id)
            img_path_list.append(img_path)

    df = pd.DataFrame(columns=["id", "img_path", "label"])
    df["id"] = sample_id_list
    df["img_path"] = img_path_list
    if len(label_list) > 0:
        df["label"] = label_list
    return df


def resize_with_pad(
    image: np.array, new_shape: Tuple[int, int], padding_color: Tuple[int] = (255, 255, 255)
) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple([int(x * ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image


def concat_img_label_long(first_img_path, second_img_path, first_label, second_label):
    # stackmix 논문은 이미지를 n개로 합칩니다. 4+1+1도 가능
    # 현재 구현체는 2개로만 합칩니다. 4+2, 3+3 등
    img_shapes = set()
    main_img = cv2.imread(first_img_path)
    main_img_shape = main_img.shape[:-1]
    sub_img = cv2.imread(second_img_path)
    sub_img_shape = sub_img.shape[:-1]

    def __concat_long_label(first_label, second_label):
        concated_label = ""
        concated_label = first_label + second_label
        if first_label > second_label:
            result = True
        elif first_label < second_label:
            result = False
        else:
            result = None
        return (result, concated_label)

    is_big_first, concated_label = __concat_long_label(first_label, second_label)
    # image의 shape이 다르면 concat이 되지 않으므로, 걔들 중 가장 큰 이미지로 선택하기 위함
    if is_big_first == True:
        main_img_shape = list(map(lambda x: x // 2, main_img_shape))
    elif is_big_first == False:
        sub_img_shape = list(map(lambda x: x // 2, sub_img_shape))
    # text 길이가 같은경우 비슷할 확률이 높으므로 추가 처리 X
    img_shapes.update(main_img_shape, sub_img_shape)
    # ****************** image를 resize하고 concat하는 작업
    img_resize_max = max(img_shapes)
    # TODO Resize하는 이미지로 1자리가 많이 이상해지지 않도록 최대한 보정
    if len(first_label) == 1:
        resized_main_img = resize_with_pad(main_img, (img_resize_max, img_resize_max))
    else:
        resized_main_img = cv2.resize(main_img, (img_resize_max, img_resize_max))
    if len(second_label) == 1:
        resized_sub_img = resize_with_pad(sub_img, (img_resize_max, img_resize_max))
    else:
        resized_sub_img = cv2.resize(sub_img, (img_resize_max, img_resize_max))
    concat_img_result = cv2.hconcat([resized_main_img, resized_sub_img])
    return concat_img_result, concated_label


def concat_img_label_short(img_paths_labels):
    concat_imgs = deque()
    img_shapes = set()
    concated_label = ""
    # ****************** 실제로 필요한 이미지와 label을 완성하는 작업
    for img_path_label in img_paths_labels:
        concated_label += img_path_label[2]
        img = cv2.imread(img_path_label[1])
        # image의 shape이 다르면 concat이 되지 않으므로, 걔들 중 가장 큰 이미지로 선택하기 위함
        img_shapes.update(img.shape[:-1])
        concat_imgs.append(img)
    # ****************** image를 resize하고 concat하는 작업
    img_resize_max = max(img_shapes)
    # pop을 이용해서 for문 안쓰고 좀 더 깔끔하게 처리.
    concat_img_result = cv2.resize(concat_imgs.popleft(), (img_resize_max, img_resize_max))
    while concat_imgs:
        concat_img = cv2.resize(concat_imgs.popleft(), (img_resize_max, img_resize_max))
        concat_img_result = cv2.hconcat([concat_img_result, concat_img])
    return concat_img_result, concated_label


def make_result_dict(id, label):
    # ****************** pandas로 활용할 데이터 정리
    concat_dict = {
        "id": id,
        "img_path": os.path.join(Folder.data_aug_preprocess, f"{id}"),
        "label": label,
        "length": len(label),
    }
    return concat_dict

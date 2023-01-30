import json
import os
import re
from unicodedata import normalize

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from PIL import Image
from tqdm import tqdm

from literal import DatasetColumns, Folder, RawDataColumns


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


def vector_to_text_png(array, width, height, save_folder_name, img_name):
    save_path = f"./{save_folder_name}/{img_name}.jpg"
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
            img_path = vector_to_text_png(vector, width, height, save_folder_name, sample_id)
            img_path_list.append(img_path)

    df = pd.DataFrame(columns=["id", "img_path", "label"])
    df["id"] = sample_id_list
    df["img_path"] = img_path_list
    if len(label_list) > 0:
        df["label"] = label_list
    return df

import os
import re
from unicodedata import normalize

import pandas as pd
from datasets import Dataset, Image

from literal import DatasetColumns, RawDataColumns


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
    text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ ]", "", text)
    return text


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
    dataset = dataset.cast_column(DatasetColumns.pixel_values, Image())
    return dataset

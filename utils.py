import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unicodedata import normalize

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Image
from transformers import TrOCRProcessor

from literal import DatasetColumns, RawDataColumns


def to_subchar(string):
    return normalize("NFKD", string)
    
def del_blank(text:str):
    text = text.replace(" ","")
    return text


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_dataset(csv_path: os.PathLike):
    df = pd.read_csv(csv_path)

    data_dict = {DatasetColumns.pixel_values: df[RawDataColumns.img_path]}

    if RawDataColumns.label in df.columns:
        df[RawDataColumns.label] = df[RawDataColumns.label].apply(to_subchar)
        data_dict[DatasetColumns.labels] = df[RawDataColumns.label]

    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.cast_column(DatasetColumns.pixel_values, Image())
    return dataset


# def preprocess(raw: Dict[str, Any], processor: TrOCRProcessor) -> Dict[str, Any]:
#     image_tensor = processor(raw[DatasetColumns.image], return_tensors="pt").pixel_values
#     raw["pixel_values"] = image_tensor

#     if DatasetColumns.text in raw:
#         raw["labels"] = processor.tokenizer(raw[DatasetColumns.text], return_tensors="pt").input_ids

#     return raw


def compute_metrics(pred, processor: TrOCRProcessor):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

    references = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    predictions = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # accuracy_metric = evaluate.load("wer")

    # wer_score = accuracy_metric.compute(references=references, predictions=predictions)
    acc = 0
    for i in range(len(references)):
        if references[i] == predictions[i]:
            acc += 1
        # else:
        #     print(f"{i}> label: {references[i]} <-> {predictions[i]}")

    acc = acc / len(references)

    # # print(references)
    # # print("###")
    # # print(predictions)

    return {"accuracy": acc}


@dataclass
class DataCollatorForOCR:
    processor: TrOCRProcessor
    padding = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # print(features)
        images = [feature[DatasetColumns.pixel_values] for feature in features]
        batch = self.processor(images, return_tensors=self.return_tensors)
        if DatasetColumns.labels in features[0]:
            texts = [feature[DatasetColumns.labels] for feature in features]
            labels = self.processor.tokenizer(
                texts, padding=self.padding, return_tensors=self.return_tensors
            ).input_ids
            batch["labels"] = labels
        return batch

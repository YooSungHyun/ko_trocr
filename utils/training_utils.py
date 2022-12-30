import random
from os import PathLike
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizerBase, TrOCRProcessor
from transformers.trainer_utils import EvalPrediction

from literal import RawDataColumns
from utils.dataset_utils import to_subchar


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def __get_total_label(
    train_csv_path: Union[None, PathLike] = None, valid_csv_path: Union[None, PathLike] = None, is_sub_char=True
) -> List[str]:
    if not train_csv_path is None:
        train_df = pd.read_csv(train_csv_path)
    if not valid_csv_path is None:
        valid_df = pd.read_csv(valid_csv_path)

    total_df = pd.concat([train_df, valid_df]).reset_index(drop=True)

    total_labels = total_df[RawDataColumns.label]
    total_labels = list(set(total_labels))
    total_labels.sort()
    if is_sub_char:
        total_labels = list(map(lambda x: to_subchar(x), total_labels))
    return total_labels


def add_label_tokens(
    tokenizer: PreTrainedTokenizerBase,
    train_csv_path: Union[None, PathLike] = None,
    valid_csv_path: Union[None, PathLike] = None,
    is_sub_char: bool = False,
) -> None:
    """
    label을 토크나이징 해서 unk 토큰에 해당하는 단어들을 vocab에 추가해주는 함수
    """

    total_labels = __get_total_label(train_csv_path, valid_csv_path, is_sub_char)
    tokenized_labels = tokenizer(total_labels).input_ids
    unks = []
    for idx, tokenized_label in enumerate(tokenized_labels):
        if tokenizer.unk_token_id in tokenized_label:
            unks.append(total_labels[idx])
    new_tokens = list(set(unks))

    tokenizer.add_tokens(new_tokens)
    return


def has_unk_token(
    tokenizer: PreTrainedTokenizerBase,
    train_csv_path: Union[None, PathLike] = None,
    valid_csv_path: Union[None, PathLike] = None,
    is_sub_char: bool = False,
) -> bool:
    """
    label을 토크나이징 수행시 unk 토큰이 있을 경우 True, 없으면 False
    """
    total_labels = __get_total_label(train_csv_path, valid_csv_path, is_sub_char)
    has_unk = False
    tokenized_labels = tokenizer(total_labels).input_ids
    for tokenized_label in tokenized_labels:
        if tokenizer.unk_token_id in tokenized_label:
            has_unk = True
            break

    return has_unk


def compute_metrics(pred: EvalPrediction, processor: TrOCRProcessor) -> Dict[str:float]:
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

    references = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    predictions = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    acc = 0
    for i in range(len(references)):
        if references[i] == predictions[i]:
            acc += 1

    acc = acc / len(references)

    return {"accuracy": acc}

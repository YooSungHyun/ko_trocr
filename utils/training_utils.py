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
import unicodedata


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def compute_metrics(pred: EvalPrediction, processor: TrOCRProcessor) -> Dict[str, float]:
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

    references = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    predictions = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    for ref, pred in list(zip(references, predictions)):
        if unicodedata.normalize("NFC", ref) == "통심통":
            print(pred)
            break

    acc = 0
    for i in range(len(references)):
        if references[i] == predictions[i]:
            acc += 1

    acc = acc / len(references)

    return {"accuracy": acc}

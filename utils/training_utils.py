import random

import numpy as np
import torch
from transformers import TrOCRProcessor
from transformers.trainer_utils import EvalPrediction


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def compute_metrics(pred: EvalPrediction, processor: TrOCRProcessor):
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

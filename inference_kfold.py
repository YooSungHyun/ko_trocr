import os
import time
import unicodedata

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    HfArgumentParser,
    Seq2SeqTrainer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from arguments import DatasetsArguments, ModelArguments, MyTrainingArguments
from literal import RawDataColumns
from utils import DataCollatorForOCR
from utils.dataset_utils import clean_text, get_dataset


def main(model_args: ModelArguments, dataset_args: DatasetsArguments, training_args: MyTrainingArguments):
    test_dataset = get_dataset(dataset_args.test_csv_path)
    processor = TrOCRProcessor.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_beams = training_args.generation_num_beams
    # config.no_repeat_ngram_size = 3
    # config.length_penalty = 2.0

    model = VisionEncoderDecoderModel.from_pretrained(model_args.model_name_or_path, config=config)
    fold_name = model_args.model_name_or_path.split('/')[-1]
    print(fold_name)
    data_collator = DataCollatorForOCR(processor=processor)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=False,
    )

    model.to("cuda")
    preds = []
    scores = []
    for test_data in tqdm(test_loader, total=len(test_dataset) // training_args.per_device_eval_batch_size + 1):
        output = model.generate(**test_data.to("cuda"), output_scores=True, return_dict_in_generate=True)
        preds.append(output.sequences.to("cpu"))
        scores.append(output.sequences_scores.to("cpu"))

    # print(preds)
    # print(scores)

    #     break
    labels = []
    seq_scores = []
    for pred in preds:
        ocr_result = processor.tokenizer.batch_decode(pred, skip_special_tokens=True)
        ocr_result = list(map(lambda x: unicodedata.normalize("NFC", x), ocr_result))
        labels.extend(ocr_result)
    for score in scores:
        seq_scores.extend(score.tolist())
    sub = pd.read_csv("data/sample_submission.csv")
    sub[RawDataColumns.label] = labels
    sub[RawDataColumns.label] = sub[RawDataColumns.label].apply(clean_text)
    sub["seq_scores"] = seq_scores
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)
    csv_name = time.strftime("%Y-%H-%M-%S") + fold_name +".csv"
    sub.to_csv(os.path.join(training_args.output_dir, csv_name), index=False)

    pass


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args=model_args, dataset_args=dataset_args, training_args=training_args)

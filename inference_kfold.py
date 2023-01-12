import os
import time
import unicodedata

import numpy as np
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
from utils.inference_utils import Seq2SeqTrainerForBeamScoring


def main(model_args: ModelArguments, dataset_args: DatasetsArguments, training_args: MyTrainingArguments):
    test_dataset = get_dataset(dataset_args.test_csv_path)
    processor = TrOCRProcessor.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_beams = training_args.generation_num_beams
    # config.no_repeat_ngram_size = 3
    # config.length_penalty = 2.0

    model = VisionEncoderDecoderModel.from_pretrained(model_args.model_name_or_path, config=config)
    fold_name = model_args.model_name_or_path.split("/")[-1]
    print(fold_name)
    data_collator = DataCollatorForOCR(processor=processor)

    trainer = Seq2SeqTrainerForBeamScoring(
        model=model,
        data_collator=data_collator,
        args=training_args,
    )
    gen_kwargs = {
        "num_beams": training_args.generation_num_beams,
        "output_scores": True,
        "return_dict_in_generate": True,
        # "num_return_sequences": training_args.generation_num_beams,
    }
    output = trainer.predict(test_dataset, **gen_kwargs)
    ocr_beam_result = processor.tokenizer.batch_decode(output.predictions, skip_special_tokens=True)
    ocr_beam_result = list(map(lambda x: unicodedata.normalize("NFC", x), ocr_beam_result))
    ocr_beam_result = list(map(lambda x: clean_text(x), ocr_beam_result))
    ocr_beam_probs = np.exp(output.sequences_scores).tolist()

    # print(ocr_beam_result)
    # print(ocr_beam_probs)

    sub = pd.read_csv("data/sample_submission.csv")

    sub[RawDataColumns.label] = ocr_beam_result
    sub["seq_probs"] = ocr_beam_probs
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)
    csv_name = f'{time.strftime("%Y-%m-%d-%H-%M")}_{fold_name}.csv'
    if training_args.local_rank == 0:
        sub.to_csv(os.path.join(training_args.output_dir, csv_name), index=False)

    pass


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args=model_args, dataset_args=dataset_args, training_args=training_args)

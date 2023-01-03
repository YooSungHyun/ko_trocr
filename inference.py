import os
import time
import unicodedata

import pandas as pd
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
from utils.augmentation import sharpening


def main(model_args: ModelArguments, dataset_args: DatasetsArguments, training_args: MyTrainingArguments):
    test_dataset = get_dataset(dataset_args.test_csv_path)
    # test_dataset.set_transform(sharpening)
    processor = TrOCRProcessor.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_beams = training_args.generation_num_beams
    model = VisionEncoderDecoderModel.from_pretrained(model_args.model_name_or_path, config=config)
    data_collator = DataCollatorForOCR(processor=processor)
    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
    )
    # gen_kwargs = {
    #     "num_beams": training_args.generation_num_beams,
    #     "do_sample": True,
    #     "num_beam_groups": 1,
    # }
    output = trainer.predict(test_dataset)
    ocr_result = processor.tokenizer.batch_decode(output.predictions, skip_special_tokens=True)
    ocr_result = list(map(lambda x: unicodedata.normalize("NFC", x), ocr_result))
    sub = pd.read_csv(dataset_args.submission_csv_path)
    sub[RawDataColumns.label] = ocr_result
    sub[RawDataColumns.label] = sub[RawDataColumns.label].apply(clean_text)
    if not os.path.exists("sub"):
        os.mkdir("sub")
    csv_name = time.strftime("%Y-%H-%M-%S") + ".csv"
    sub.to_csv(os.path.join("sub", csv_name), index=False)

    pass


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args=model_args, dataset_args=dataset_args, training_args=training_args)

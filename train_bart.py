import logging
import os
from functools import partial
from unicodedata import normalize

from setproctitle import setproctitle
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    TrOCRProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)
from transformers.trainer_utils import is_main_process

from arguments import DatasetsArguments, ModelArguments, MyTrainingArguments
from utils import DataCollatorForOCR, compute_metrics, get_dataset, seed_everything

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["WANDB_DISABLED"] = "true"
NUM_GPU = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
KYOWON_POS = ["쫠", "턺", "쫬", "촁", "뷩"]


def main(model_args: ModelArguments, dataset_args: DatasetsArguments, training_args: MyTrainingArguments):
    setproctitle("kyowon")
    seed_everything(training_args.seed)

    # 데이터 로드
    train_dataset = get_dataset(dataset_args.train_csv_path)
    valid_dataset = get_dataset(dataset_args.valid_csv_path)

    # 모델, 컨피그 ,프로세서 로드
    vision_model_name = model_args.encoder_model_name_or_path
    text_model_name = model_args.decoder_model_name_or_path

    vision_config = AutoConfig.from_pretrained(vision_model_name)
    text_config = AutoConfig.from_pretrained(text_model_name)

    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config=vision_config, decoder_config=text_config
    )
    image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": KYOWON_POS})
    ocr_processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # config 설정
    config.decoder_start_token_id = config.decoder.decoder_start_token_id
    config.pad_token_id = config.decoder.pad_token_id
    config.vocab_size = config.decoder.vocab_size

    # set beam search parameters
    config.eos_token_id = ocr_processor.tokenizer.eos_token_id
    config.max_length = 30
    config.early_stopping = True
    config.no_repeat_ngram_size = 3
    config.length_penalty = 2.0
    config.num_beams = 5

    # config.save_pretrained(training_args.output_dir)

    # encoder_add_pooling_layer=False
    # https://github.com/huggingface/transformers/issues/7924 ddp 관련
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=vision_model_name,
        decoder_pretrained_model_name_or_path=text_model_name,
        encoder_add_pooling_layer=False,
    )
    model.config = config

    # 데이터 콜레이터 로드
    data_collator = DataCollatorForOCR(processor=ocr_processor)

    # 로깅 스텝 설정 -> 한 에폭에 5번
    total_batch = training_args.train_batch_size * training_args.gradient_accumulation_steps * NUM_GPU
    one_epoch_len = len(train_dataset) // total_batch
    training_args.eval_steps = one_epoch_len // 5
    training_args.save_steps = one_epoch_len // 5
    training_args.logging_steps = one_epoch_len // 10

    if training_args.local_rank == 0:
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_name,
        )

    # compute_metrics에 processor 할당
    compute_metrics_with_processor = partial(compute_metrics, processor=ocr_processor, model_config=config)

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_processor,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
    )

    trainer.train()
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
        ocr_processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    main(model_args=model_args, dataset_args=dataset_args, training_args=training_args)

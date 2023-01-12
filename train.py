import logging
import os
from functools import partial

from setproctitle import setproctitle
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    TrOCRProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)
from transformers.trainer_utils import is_main_process

from arguments import DatasetsArguments, ModelArguments, MyTrainingArguments
from utils import DataCollatorForGptOCR, DataCollatorForOCR
from utils.augmentation import Augmentator
from utils.dataset_utils import get_dataset
from utils.training_utils import (
    add_label_tokens,
    compute_metrics,
    has_unk_token,
    seed_everything,
)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # true면 데드락 -> 살펴보기????
# os.environ["WANDB_DISABLED"] = "true"
NUM_GPU = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
logger = logging.getLogger(__name__)


def main(model_args: ModelArguments, dataset_args: DatasetsArguments, training_args: MyTrainingArguments):
    setproctitle("kyowon")
    seed_everything(training_args.seed)
    vision_model_name = model_args.encoder_model_name_or_path
    text_model_name = model_args.decoder_model_name_or_path

    # 데이터 로드
    is_sub_char = False
    if text_model_name == "snunlp/KR-BERT-char16424":
        is_sub_char = True

    image_processor = AutoImageProcessor.from_pretrained(vision_model_name)

    augmentator = Augmentator(
        aug_with_compose_prob=0.8, rotation_prob=0.5, rotation_square_side=min(image_processor.size.values())
    )  # min? max?
    train_dataset = get_dataset(dataset_args.train_csv_path, is_sub_char=is_sub_char)
    train_dataset.set_transform(augmentator.augmentation)
    valid_dataset = get_dataset(dataset_args.valid_csv_path, is_sub_char=is_sub_char)

    # 모델, 컨피그 ,프로세서 로드

    if "gpt" in text_model_name:
        # print("in")
        tokenizer = AutoTokenizer.from_pretrained(
            text_model_name,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # label에 unk 토큰이 있으면 vocab에 추가시켜줌
    has_unk = has_unk_token(
        tokenizer=tokenizer,
        train_csv_path=dataset_args.train_csv_path,
        valid_csv_path=dataset_args.valid_csv_path,
        is_sub_char=is_sub_char,
    )
    if has_unk:
        # 이 부분은 추가 검증 필요->lm head를 바꾸면 가중치가 의미가 없어지기 때문
        logger.info(f"tokenized labels has unk token\nadd new tokens")
        logger.info(f"before len(tokenizer): {len(tokenizer)}")
        # add_label_tokens(
        #     tokenizer=tokenizer,
        #     train_csv_path=dataset_args.train_csv_path,
        #     valid_csv_path=dataset_args.valid_csv_path,
        #     is_sub_char=is_sub_char,
        # )
        logger.info(f"after len(tokenizer): {len(tokenizer)}")
    vision_config = AutoConfig.from_pretrained(vision_model_name)
    text_config = AutoConfig.from_pretrained(text_model_name)

    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config=vision_config, decoder_config=text_config
    )
    ocr_processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # config 설정
    config.decoder.vocab_size = len(tokenizer)
    config.decoder_start_token_id = (
        tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    )
    config.pad_token_id = ocr_processor.tokenizer.pad_token_id
    config.vocab_size = config.decoder.vocab_size

    # set beam search parameters
    config.eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    config.max_length = 32  # arg로 받을 수 있게 수정 "snunlp/KR-BERT-char16424"의 경우 최대 길이가 16, 넉넉히 32를 주면 될듯?
    # config.early_stopping = True
    # config.no_repeat_ngram_size = 3
    # config.length_penalty = 2.0
    config.num_beams = 10

    # encoder_add_pooling_layer=False
    # https://github.com/huggingface/transformers/issues/7924 ddp 관련
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=vision_model_name,
        decoder_pretrained_model_name_or_path=text_model_name,
        encoder_add_pooling_layer=False,
        decoder_vocab_size=config.vocab_size,
        decoder_ignore_mismatched_sizes=True,
    )
    model.config = config

    # 데이터 콜레이터 로드
    if "gpt" in text_model_name:
        data_collator = DataCollatorForGptOCR(processor=ocr_processor)
    else:
        data_collator = DataCollatorForOCR(processor=ocr_processor)

    # 로깅 스텝 설정 -> 한 에폭에 5번
    # 세이브 스텝 -> 한 에폭에 2번
    total_batch = training_args.train_batch_size * training_args.gradient_accumulation_steps * NUM_GPU
    one_epoch_len = len(train_dataset) // total_batch
    total_steps = training_args.num_train_epochs * one_epoch_len
    training_args.eval_steps = total_steps // 10
    training_args.save_steps = total_steps // 10
    training_args.logging_steps = one_epoch_len // 10

    if training_args.local_rank == 0:
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_name,
        )

    # compute_metrics에 processor 할당
    compute_metrics_with_processor = partial(compute_metrics, processor=ocr_processor)

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

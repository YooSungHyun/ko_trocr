import logging
import os
from functools import partial

from setproctitle import setproctitle
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
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
from utils.training_utils import add_label_tokens, compute_metrics, has_unk_token, seed_everything

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # true면 데드락 -> 살펴보기????
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
        aug_with_compose_prob=0.8, rotation_prob=0.5, rotation_square_side=max(image_processor.size.values())
    )
    train_dataset = get_dataset(dataset_args.train_csv_path, is_sub_char=is_sub_char)
    train_dataset.set_transform(augmentator.augmentation)
    valid_dataset = get_dataset(dataset_args.valid_csv_path, is_sub_char=is_sub_char)

    """ Tokenizer and vocab process
        Decoder or Encoder + LMHead have to need another process style

        unk token task is check unk vocab and add to vocab forcing
    """
    if "gpt" in text_model_name:
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

    has_unk = has_unk_token(
        tokenizer=tokenizer,
        train_csv_path=dataset_args.train_csv_path,
        valid_csv_path=dataset_args.valid_csv_path,
        is_sub_char=is_sub_char,
    )
    if has_unk:
        # TODO: 이 부분은 추가 검증 필요->lm head를 바꾸면 가중치가 의미가 없어지기 때문
        logger.info("tokenized labels has unk token\nadd new tokens")
        logger.info(f"before len(tokenizer): {len(tokenizer)}")
        # add_label_tokens(
        #     tokenizer=tokenizer,
        #     train_csv_path=dataset_args.train_csv_path,
        #     valid_csv_path=dataset_args.valid_csv_path,
        #     is_sub_char=is_sub_char,
        # )
        logger.info(f"after len(tokenizer): {len(tokenizer)}")

    """ Goal = MAKE VisionEncoderDecoderModel(Just Load TROCR Encoder + Text Decoder)
    1. Load TROCR Config only vision_encoder
        1) load Full TROCR Config
        2) get only Encoder Config
    2. Load Text Config (Encoder only or Decoder Style)
    3. Set new VisionEncoderDecoderConfig
    4. Additional Config Setting
    5. Same as 1, 2 for Each Model
    6. Set new VisionEncoderDecoderModel
    """
    # Step 1. Load TROCR Config only vision_encoder
    encoder_config = AutoConfig.from_pretrained(vision_model_name).encoder
    encoder_config.add_pooling_layer = False  # https://github.com/huggingface/transformers/issues/7924 ddp error

    # Step 2. Load Text Config (Encoder only or Decoder Style)
    decoder_config = AutoConfig.from_pretrained(text_model_name)

    # Step 3. Set new VisionEncoderDecoderConfig
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config=encoder_config, decoder_config=decoder_config
    )

    # Step 4. Additional Config Setting ##########################################################################
    ocr_processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    config.vocab_size = config.decoder.vocab_size
    config.decoder_start_token_id = (
        tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    )
    config.pad_token_id = ocr_processor.tokenizer.pad_token_id

    # Setting BeamSearch parameters
    config.eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    config.max_length = model_args.max_length
    config.num_beams = model_args.num_beams
    ###############################################################################################################

    # Step 5-1. Load TROCR Model only vision_encoder
    encoder_model = VisionEncoderDecoderModel.from_pretrained(vision_model_name).get_encoder()

    # Step 5-2. Load Text Model (Encoder only or Decoder Style)
    decoder_model = AutoModelForCausalLM.from_pretrained(
        text_model_name,
        is_decoder=True,
        add_cross_attention=True,
        vocab_size=len(tokenizer),
        ignore_mismatched_sizes=True,
    )  # Model Key args MUST in from_pretrained time

    # Step 6. Set new VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel(config, encoder_model, decoder_model)

    # Load DataCollator For Training Step
    if "gpt" in text_model_name:
        data_collator = DataCollatorForGptOCR(processor=ocr_processor)
    else:
        data_collator = DataCollatorForOCR(processor=ocr_processor)

    # Logging Step -> 10 Times in 1 Epoch
    logging_times = 10
    # Eval and Save Step -> 2 Times in 1 Epoch
    eval_save_times = 2
    total_batch = training_args.train_batch_size * training_args.gradient_accumulation_steps * NUM_GPU
    one_epoch_len = len(train_dataset) // total_batch
    training_args.logging_steps = one_epoch_len // logging_times
    training_args.eval_steps = one_epoch_len // eval_save_times
    training_args.save_steps = one_epoch_len // eval_save_times

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

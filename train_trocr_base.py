import logging
import os
from functools import partial

from setproctitle import setproctitle
from transformers import (
    AutoConfig,
    AutoImageProcessor,
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
from utils import DataCollatorForOCR
from utils.augmentation import ChildWrittenAugmentator
from utils.dataset_utils import get_dataset
from utils.training_utils import compute_metrics, seed_everything

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
    augmentator = ChildWrittenAugmentator(aug_with_compose_prob=0.8)
    train_dataset = get_dataset(dataset_args.train_csv_path, is_sub_char=is_sub_char)
    # train_dataset.set_transform(augmentator.augmentation)
    valid_dataset = get_dataset(dataset_args.valid_csv_path, is_sub_char=is_sub_char)

    """ Tokenizer and vocab process
        Decoder or Encoder + LMHead have to need another process style

        unk token task is check unk vocab and add to vocab forcing
    """
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

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
    # encoder_config = AutoConfig.from_pretrained(vision_model_name)
    # encoder_config.add_pooling_layer = False  # https://github.com/huggingface/transformers/issues/7924 ddp error

    # Step 2. Load Text Config (Encoder only or Decoder Style)
    # decoder_config = AutoConfig.from_pretrained(text_model_name)

    # Step 3. Set new VisionEncoderDecoderConfig
    config = VisionEncoderDecoderConfig.from_pretrained(vision_model_name)
    config.hidden_size = max(config.encoder.hidden_size, config.decoder.hidden_size)  # for deepspeed

    # Step 4. Additional Config Setting ##########################################################################
    ocr_processor = TrOCRProcessor.from_pretrained(vision_model_name)
    # config.vocab_size = config.decoder.vocab_size
    # config.decoder_start_token_id = (
    # tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    # )
    # config.pad_token_id = ocr_processor.tokenizer.pad_token_id

    # Setting BeamSearch parameters
    # config.eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    config.max_length = training_args.generation_max_length
    config.num_beams = training_args.generation_num_beams
    ###############################################################################################################

    # Step 5-1. Load TROCR Model only vision_encoder
    model = VisionEncoderDecoderModel.from_pretrained(vision_model_name)
    for name, param in model.named_parameters():
        if name.startswith("encoder.embeddings"):
            param.requires_grad = False
        elif name.startswith("decoder.roberta.embeddings"):
            param.requires_grad = False
    model.config = config
    # Step 5-2. Load Text Model (Encoder only or Decoder Style)
    # decoder_model = AutoModelForCausalLM.from_pretrained(
    #     text_model_name,
    #     is_decoder=True,
    #     add_cross_attention=True,
    #     vocab_size=len(tokenizer),
    #     ignore_mismatched_sizes=True,
    #     use_cache=True,
    # )  # Model Key args MUST in from_pretrained time

    # Step 6. Set new VisionEncoderDecoderModel
    # model = VisionEncoderDecoderModel(config, encoder_model, decoder_model)

    # Load DataCollator For Training Step
    data_collator = DataCollatorForOCR(processor=ocr_processor)

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
    if training_args.local_rank == 0:
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_name,
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

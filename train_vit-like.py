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
from utils import DataCollatorForOCR
from utils.augmentation import Augmentator
from utils.dataset_utils import get_dataset
from utils.training_utils import compute_metrics, seed_everything

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def main(model_args: ModelArguments, dataset_args: DatasetsArguments, training_args: MyTrainingArguments):
    setproctitle("kyowon")
    seed_everything(training_args.seed)
    vision_model_name = model_args.encoder_model_name_or_path
    text_model_name = model_args.decoder_model_name_or_path

    """
    goal: training VisionEncoderDecoderModel(TROCR) with pretrained ViT-like model(ViT, Swin etc...) and PLM model (BERT, RoBERTa, etc...)
    process
        1) load processor
        2) setting config
            2-1) load config & set config
            2-2) set config for beam search
        3) load dataset with agumentation & load datacollator
        4) load model
        5) load trainer and train
        6) save model with config and processor
    """

    # 1) load processor #####################################################################
    image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    ocr_processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    #########################################################################################

    # 2) setting config ############################################################################################
    # 2-1 ) load config & set config
    vision_config = AutoConfig.from_pretrained(vision_model_name)
    text_config = AutoConfig.from_pretrained(text_model_name)
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config=vision_config, decoder_config=text_config
    )
    config.decoder.vocab_size = len(tokenizer)
    config.vocab_size = config.decoder.vocab_size

    # 2-2) set config for beam search
    config.decoder_start_token_id = (
        tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    )
    config.pad_token_id = ocr_processor.tokenizer.pad_token_id
    config.eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    config.max_length = training_args.generation_max_length
    config.num_beams = training_args.generation_num_beams
    ################################################################################################################

    # 3) load dataset with agumentation & load datacollator ####################################################
    augmentator = Augmentator(
        aug_with_compose_prob=0.8, rotation_prob=0.5, rotation_square_side=max(image_processor.size.values())
    )

    # KR-BERT-char16424 is using sub character tokenizer, need preprocessing label
    # set parameter "is_sub_char = True" in get_dataset, function do normalize string NFC to NFKD
    is_sub_char = False
    if text_model_name == "snunlp/KR-BERT-char16424":
        is_sub_char = True

    train_dataset = get_dataset(dataset_args.train_csv_path, is_sub_char=is_sub_char)
    train_dataset.set_transform(augmentator.augmentation)
    valid_dataset = get_dataset(dataset_args.valid_csv_path, is_sub_char=is_sub_char)
    data_collator = DataCollatorForOCR(processor=ocr_processor)
    #############################################################################################################

    # 4) load model ####################################################
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
    ####################################################################

    if training_args.local_rank == 0:
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_name,
        )

    # 5) load trainer and train ########################################################
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
    ####################################################################################

    # 6) save model with config and processor ###################
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
        ocr_processor.save_pretrained(training_args.output_dir)
    #############################################################


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    main(model_args=model_args, dataset_args=dataset_args, training_args=training_args)

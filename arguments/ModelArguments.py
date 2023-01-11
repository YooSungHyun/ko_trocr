from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    encoder_model_name_or_path: str = field(default=None)
    decoder_model_name_or_path: str = field(default=None)
    model_name_or_path: str = field(default=None)
    num_beams: int = field(default=10)
    max_length: int = field(default=32)

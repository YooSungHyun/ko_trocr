from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    wandb_project: Optional[str] = field(default="", metadata={"help": "wandb project name for logging"})
    wandb_entity: Optional[str] = field(
        default="", metadata={"help": "wandb entity name(your wandb (id/team name) for logging"}
    )
    wandb_name: Optional[str] = field(default="", metadata={"help": "wandb job name for logging"})

    pass

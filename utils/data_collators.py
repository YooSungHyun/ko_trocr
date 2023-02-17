from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from transformers import TrOCRProcessor

from literal import DatasetColumns


@dataclass
class DataCollatorForOCR:
    processor: TrOCRProcessor
    padding = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [feature[DatasetColumns.pixel_values].convert("RGB") for feature in features]

        batch = self.processor.image_processor(images, return_tensors=self.return_tensors)
        if DatasetColumns.labels in features[0]:
            texts = [feature[DatasetColumns.labels] for feature in features]
            labels = self.processor.tokenizer(
                texts, padding=self.padding, return_tensors=self.return_tensors
            ).input_ids
            batch["labels"] = labels
        return batch

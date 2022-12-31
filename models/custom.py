import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, VisionEncoderDecoderModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput


class EnsembleModel(VisionEncoderDecoderModel):
    main_input_name = "pixel_values"

    def __init__(self, fold_root_path, num_beams):
        super().__init__()
        self.fold_models = []
        fold_folders = os.listdir(fold_root_path)
        for fold_folder in fold_folders:
            model_path = os.path.join(fold_root_path, fold_folder)
            config = AutoConfig.from_pretrained(model_path)
            config.num_beams = num_beams
            model = VisionEncoderDecoderModel.from_pretrained(model_path, config=config)
            model.eval()
            self.fold_models.append(model)
        self.config = config
        # self.lm_head = nn.Linear(1, config.vocab_size, bias=False)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        logit_list = []

        for i in range(len(self.fold_models)):
            output = self.fold_models[i](
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
            logit_list.append(output.logits)

        logits_sum = torch.sum(logit_list)

        return Seq2SeqLMOutput(loss=None, logits=logits_sum)

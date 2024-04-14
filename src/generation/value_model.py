from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import Linear
from transformers import AutoTokenizer, BertModel


class ValueModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        max_length: int,
        device: torch.device,
        weights_path: Path | None = None,
    ):
        super(ValueModel, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.linear_to_logit = Linear(self.model.config.hidden_size, 1)
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        self.linear_to_logit.to(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def get_value(self, generated_sequence: str, source_sequence: str) -> torch.Tensor:
        tokenized_source = self.tokenizer(
            source_sequence,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        tokenized_generated = self.tokenizer(
            generated_sequence,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        output = self.model(
            input_ids=torch.concatenate(
                [tokenized_source["input_ids"], tokenized_generated["input_ids"]], dim=-1
            ).to(self.device),
            attention_mask=torch.concatenate(
                [tokenized_source["attention_mask"], tokenized_generated["attention_mask"]], dim=-1
            ).to(self.device),
        )
        pooled = output.pooler_output.flatten()

        logit = self.linear_to_logit(pooled)
        return logit

    def eval(self):
        return self.model.eval()

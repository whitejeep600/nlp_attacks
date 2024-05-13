from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import Linear
from torch.optim import AdamW
from transformers import AutoTokenizer, BertModel


class GanDiscriminatorModule(nn.Module):
    def __init__(self, model_name: str, device: torch.device, weights_path: Path | None = None):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.model.config.hidden_dropout_prob = 0.01
        self.linear_to_logits = Linear(self.model.config.hidden_size, 2)
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location=device))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bert_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = bert_output.pooler_output
        logits = self.linear_to_logits(pooled)
        return logits


class GANDiscriminator:
    def __init__(
        self, device: torch.device, max_length: int, lr: float, weights_path: Path | None = None
    ):
        super().__init__()
        model_name = "bert-base-uncased"
        self.module = GanDiscriminatorModule(model_name, device, weights_path)
        self.module.to(device)
        self.module.train()
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = AdamW(self.module.parameters(), lr=lr)

    def prepare_batch(self, sentences_batch: list[str]) -> dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            sentences_batch,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        input_ids.requires_grad_ = True
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.module(batch["input_ids"], batch["attention_mask"])

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def weight_norm_sum(self) -> torch.Tensor:
        return torch.stack([(p.data * p.data).sum() for p in self.module.parameters()]).sum()

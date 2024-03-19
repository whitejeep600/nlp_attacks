import torch
from torch import nn
from torch.nn import Linear
from transformers import AdamW, AutoTokenizer, BertModel


class GANDiscriminator(nn.Module):
    def __init__(self, device: str, max_length: int, lr: float):
        super().__init__()
        model_name = "bert-base-uncased"
        self.model = BertModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.train()
        self.linear_to_logits = Linear(self.model.config.hidden_size, 2)
        self.linear_to_logits.to(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = AdamW(self.parameters(), lr=lr)

    def forward(self, sentences_batch: list[str]) -> torch.Tensor:
        tokenized = self.tokenizer(
            sentences_batch,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        bert_output = self.model(
            input_ids=tokenized["input_ids"].to(self.device),
            attention_mask=tokenized["attention_mask"].to(self.device),
        )

        pooled = bert_output.pooler_output
        logits = self.linear_to_logits(pooled)
        return logits

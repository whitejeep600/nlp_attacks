import torch
from torch.nn import Linear
from transformers import AutoTokenizer, BertModel


class ValueModel(torch.nn.Module):
    def __init__(self, model_name: str, max_length: int, device: str):
        super(ValueModel, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.linear_to_logit = Linear(max_length * 2, 1)
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
                [
                    tokenized_source["input_ids"].flatten(),
                    tokenized_generated["input_ids"].flatten()
                ],
                dim=-1
            ),
            attention_mask=torch.concatenate(
                [
                    tokenized_source["attention_mask"].flatten(),
                    tokenized_generated["attention_mask"].flatten()
                ],
                dim=-1
            )
        )
        pooled = output.pooler_output

        logit = self.linear_to_logit(
            pooled
        )
        return logit

    def eval(self) -> None:
        self.model.eval()


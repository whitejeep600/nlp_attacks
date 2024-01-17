from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SST2Dataset(Dataset):
    def __init__(self, dataset_csv_path: Path, tokenizer: PreTrainedTokenizer, max_length: int):
        super().__init__()
        source_df = pd.read_csv(dataset_csv_path)

        self.df = source_df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        text = self.df.iloc[i, :]["sentence"]
        label = self.df.iloc[i, :]["label"]
        id = self.df.iloc[i, :]["idx"]
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        # Why tokenize and the untokenize? Well, this way we make sure there is
        # consistency, for example, with regard to text length after truncation.
        original_seq = self.tokenizer.decode(
            tokenized_text["input_ids"].flatten(), skip_special_tokens=True
        )
        return {
            "input_ids": tokenized_text["input_ids"].flatten(),
            "attention_mask": tokenized_text["attention_mask"].flatten(),
            "original_seq": original_seq,
            "label": torch.tensor(label),
            "id": torch.tensor(id),
        }

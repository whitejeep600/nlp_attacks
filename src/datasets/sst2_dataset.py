from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.constants import ID, LABEL, SENTENCE


# We might  want to filter out short samples, which in the sst2 dataset are often incomplete
# sentences (for some reason). Training on such short sentences is not appropriate for some
# experiments, and simply hard. Also, for the unidirectional task, training on already negative
# examples is not very informative, so we want a possibility to filter them out.
class SST2Dataset(Dataset):
    def __init__(
        self,
        dataset_csv_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        min_length: int | None = None,
        filter_label: int | None = None,
    ):
        super().__init__()
        source_df = pd.read_csv(dataset_csv_path)

        if filter_label is not None:
            source_df = source_df[source_df[LABEL] == filter_label]

        if min_length is not None:
            source_df = source_df[source_df[SENTENCE].apply(lambda x: len(x.split())) >= min_length]

        self.df = source_df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        text = self.df.iloc[i, :][SENTENCE]
        label = self.df.iloc[i, :][LABEL]
        id = self.df.iloc[i, :][ID]
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
            LABEL: torch.tensor(label),
            ID: torch.tensor(id),
        }

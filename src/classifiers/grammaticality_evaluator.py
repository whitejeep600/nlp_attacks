import textattack
import torch
import transformers
from textattack.model_args import HUGGINGFACE_MODELS

from src.classifiers.base_classifier import Classifier


class GrammaticalityEvaluator(Classifier):
    def __init__(self, device: torch.device):
        super().__init__()

        raw_name = "bert-base-uncased-cola"
        textattack_model_name = HUGGINGFACE_MODELS[raw_name]
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            textattack_model_name
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(textattack_model_name, use_fast=True)
        model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        self.model = model

        self.device = device
        self.model.to(device)
        self.model.model.eval()

    def evaluate_texts(self, texts: list[str], return_probs=False) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(texts)
        if not return_probs:
            return logits
        else:
            probs = torch.softmax(logits, dim=1)
            return probs

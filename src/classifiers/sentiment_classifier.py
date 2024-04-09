import textattack
import torch
import transformers
from textattack.model_args import HUGGINGFACE_MODELS
from textattack.models.helpers import WordCNNForClassification

from src.classifiers.base_classifier import Classifier


class SentimentClassifier(Classifier):
    def __init__(self, device: str, raw_name: str = "bert-base-uncased-sst2"):
        super().__init__()

        if raw_name in HUGGINGFACE_MODELS:
            textattack_model_name = HUGGINGFACE_MODELS[raw_name]
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                textattack_model_name
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                textattack_model_name, use_fast=True
            )
            model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
            self.model = model

            self.device = device
            self.model.to(device)
            self.model.model.eval()
        else:
            # Only the cnn-sst2 model is supported for now, idk if it can be done more elegantly
            # with the interface provided by the TextAttack library.
            self.model = WordCNNForClassification.from_pretrained(raw_name)
            self.device = device
            self.model.to(device)
            self.model.eval()

        self.raw_name = raw_name

    def evaluate_texts(self, texts: list[str], return_probs=False) -> torch.Tensor:
        with torch.no_grad():
            if self.raw_name == "cnn-sst2":
                logits = self.model(self.model.tokenizer.batch_encode(texts).to(self.device))
            else:
                logits = self.model(texts)
        if not return_probs:
            return logits
        else:
            probs = torch.softmax(logits, dim=1)
            return probs

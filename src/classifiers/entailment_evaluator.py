import textattack
import torch
import transformers
from textattack.model_args import HUGGINGFACE_MODELS

from src.classifiers.base_classifier import Classifier


class EntailmentEvaluator(Classifier):
    def __init__(self, device: torch.device):
        super().__init__()

        raw_name = "distilbert-base-cased-snli"
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

        self.contradiction_code = 2
        self.entailment_code = 0
        self.neutral_code = 1

    def evaluate_text_pairs(self, texts: list[tuple[str, str]], return_probs=False) -> torch.Tensor:
        prepared_inputs = [
            f"Premise: {premise} \nHypothesis: {hypothesis}" for (premise, hypothesis) in texts
        ]
        with torch.no_grad():
            logits = self.model(prepared_inputs)
        if not return_probs:
            return logits
        else:
            probs = torch.softmax(logits, dim=1)
            return probs

    def eval(self):
        self.model.model.eval()

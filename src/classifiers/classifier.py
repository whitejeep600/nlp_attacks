# This is a general class representing models that output binary classification scores for
# a piece of text. In the context of this repository, they are the models that will be subjected
# to attacks (victim models).
import textattack
import torch
import transformers
from textattack.model_args import HUGGINGFACE_MODELS


class Classifier:
    def __init__(self):
        pass

    def classify(self, text: str) -> torch.Tensor:
        raise NotImplementedError


class SentimentClassifier(Classifier):
    def __init__(self, device: str):
        super().__init__()

        raw_name = "bert-base-uncased-sst2"
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


class GrammaticalityEvaluator(Classifier):
    def __init__(self, device: str):
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


class EntailmentClassifier(Classifier):
    def __init__(self, device: str):
        super().__init__()

        raw_name = "bert-base-uncased-snli"
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

        self.contradiction_code = 0
        self.entailment_code = 1
        self.neutral_code = 2

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

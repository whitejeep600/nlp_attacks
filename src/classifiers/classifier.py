# This is a general class representing models that output binary classification scores for
# a piece of text. In the context of this repository, they are the models that will be subjected
# to attacks (victim models).
import torch
from textattack.models.helpers import WordCNNForClassification


class Classifier:
    def __init__(self):
        pass

    def classify(self, text: str) -> torch.Tensor:
        raise NotImplementedError


class SentimentClassifier(Classifier):
    def __init__(self, device: str):
        super().__init__()
        self.model = WordCNNForClassification.from_pretrained("cnn-sst2")
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate_texts(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            return self.model(
                torch.IntTensor(self.model.tokenizer.batch_encode(texts)).to(self.device)
            )

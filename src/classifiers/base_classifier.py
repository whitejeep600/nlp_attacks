# This is a general class representing models that output binary classification scores for
# a piece of text. In the context of this repository, they are the models that will be subjected
# to attacks (victim models).
import torch


class Classifier:
    def __init__(self):
        pass

    def classify(self, text: str) -> torch.Tensor:
        raise NotImplementedError

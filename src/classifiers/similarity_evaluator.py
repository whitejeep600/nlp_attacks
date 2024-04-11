import torch
from sentence_transformers import SentenceTransformer

from src.utils import get_ceil_power_of_2


class SimilarityEvaluator:
    def __init__(self, model_name: str, device: torch.device):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.eval()

    def evaluate(self, text1: str, text2: str) -> float:
        embeddings = [self.model.encode(text, convert_to_tensor=True) for text in [text1, text2]]
        return torch.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

    def batch_evaluate(self, texts1: list[str], texts2: list[str]) -> list[float]:
        encodings_1 = self.model.encode(
            texts1, convert_to_tensor=True, batch_size=get_ceil_power_of_2(len(texts1))
        )
        encodings_2 = self.model.encode(
            texts2, convert_to_tensor=True, batch_size=get_ceil_power_of_2(len(texts1))
        )
        return [
            torch.cosine_similarity(encoding_1, encoding_2, dim=0).item()
            for (encoding_1, encoding_2) in zip(encodings_1, encodings_2)
        ]

    def evaluate_many_to_one(self, many: list[str], one: str) -> list[float]:
        whole_reference_encoding = self.model.encode(one, convert_to_tensor=True)
        prefix_encodings = self.model.encode(
            many, convert_to_tensor=True, batch_size=get_ceil_power_of_2(len(many))
        )

        return [
            torch.cosine_similarity(whole_reference_encoding, prefix_encoding, dim=0).item()
            for prefix_encoding in prefix_encodings
        ]

    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()


def get_similarity_scores(
    batch_prefixes: list[list[str]],
    original_seqs: list[str],
    similarity_evaluator: SimilarityEvaluator,
    device: str,
) -> list[torch.Tensor]:
    similarity_scores = [
        torch.Tensor(similarity_evaluator.evaluate_many_to_one(sample_prefixes, original_seq)).to(
            device
        )
        for sample_prefixes, original_seq in zip(batch_prefixes, original_seqs)
    ]
    return similarity_scores

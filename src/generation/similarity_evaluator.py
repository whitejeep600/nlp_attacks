import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor


class SimilarityEvaluator:
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.whole_reference_encoding: Tensor | None = None
        self.tokens: list[str] = []
        self.absolute_ind_to_sentence_ind: dict[int, tuple[int, int]] = {}
        self.sentences: list[list[str]] = []
        self.sentence_encodings: list[Tensor] = []
        self.sentence_pos_tags: list[list[str]] = []
        self.eval()

    def evaluate(self, text1: str, text2: str) -> float:
        embeddings = [self.model.encode(text, convert_to_tensor=True) for text in [text1, text2]]
        return torch.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

    def evaluate_prefixes(self, prefixes: list[str], text: str) -> list[float]:
        whole_reference_encoding = self.model.encode(text, convert_to_tensor=True)
        prefix_encodings = [
            self.model.encode(prefix, convert_to_tensor=True)
            for prefix in prefixes
        ]
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
    device: str
) -> list[torch.Tensor]:
    similarity_scores = [
        torch.Tensor(
            similarity_evaluator.evaluate_prefixes(sample_prefixes, original_seq)
        ).to(device)
        for sample_prefixes, original_seq in zip(batch_prefixes, original_seqs)
    ]
    return similarity_scores
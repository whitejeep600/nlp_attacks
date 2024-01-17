from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.datasets.sst2_dataset import SST2Dataset
from src.generation.generative_bart import GenerativeBart
from src.generation.similarity_evaluator import SimilarityEvaluator
from src.generation.value_model import ValueModel
from src.utils import get_available_torch_device


def train(
        echo: GenerativeBart,
        similarity_evaluator: SimilarityEvaluator,
        value_model: ValueModel,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        n_epochs: int,
        attacker_lr: float,
        value_lr: float,
        device: str,
        max_len: int,
        save_path: Path
):
    pass


def main(
        source_model_name: str,
        similarity_evaluator_name: str,
        value_model_name: str,
        train_split_path: Path,
        eval_split_path: Path,
        max_len: int,
        batch_size: int,
        n_epochs: int,
        attacker_lr: float,
        value_lr: float,
        save_path: Path
):
    device = get_available_torch_device()
    similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name, device)
    value_model = ValueModel(value_model_name, max_len, device)
    echo = GenerativeBart(source_model_name, max_len, device)
    train_dataset = SST2Dataset(
        train_split_path,
        echo.tokenizer,
        max_len,
    )
    eval_dataset = SST2Dataset(
        eval_split_path,
        echo.tokenizer,
        max_len,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    train(
        echo,
        similarity_evaluator,
        value_model,
        train_dataloader,
        eval_dataloader,
        n_epochs,
        attacker_lr,
        value_lr,
        device,
        max_len,
        save_path,
    )


if __name__ == "__main__":
    echo_params = yaml.safe_load(open("params.yaml"))["attacks.generative"]

    source_model_name = echo_params["source_model_name"]
    similarity_evaluator_name = echo_params["similarity_evaluator_name"]
    value_model_name = echo_params["value_model_name"]

    train_split_path = Path(echo_params["train_split_path"])
    eval_split_path = Path(echo_params["eval_split_path"])

    max_len = int(echo_params["max_len"])
    batch_size = int(echo_params["batch_size"])
    n_epochs = int(echo_params["n_epochs"])
    attacker_lr = float(echo_params["attacker_lr"])
    value_lr = float(echo_params["value_lr"])

    save_path = Path(echo_params["save_path"])

    main(
        source_model_name,
        similarity_evaluator_name,
        value_model_name,
        train_split_path,
        eval_split_path,
        max_len,
        batch_size,
        n_epochs,
        attacker_lr,
        value_lr,
        save_path,
    )

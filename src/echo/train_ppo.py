from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classifiers.similarity_evaluator import SimilarityEvaluator, get_similarity_scores
from src.constants import EVAL, TRAIN
from src.datasets.sst2_dataset import SST2Dataset
from src.generation.generative_bart import GenerativeBart
from src.generation.ppo_trainer import PPOTrainer
from src.generation.value_model import ValueModel
from src.utils import get_available_torch_devices, get_next_run_subdir_name


def get_rewards_and_nonstandard_metrics(
    batch: dict,
    batch_prefixes: list[list[str]],
    original_seqs: list[str],
    similarity_evaluator: SimilarityEvaluator,
    device: str,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    similarity_scores = get_similarity_scores(
        batch_prefixes, original_seqs, similarity_evaluator, device
    )

    stats = {
        "similarity_scores": float(torch.mean(torch.concat(similarity_scores, dim=0)).item()),
    }
    return similarity_scores, stats


def train(
    echo: GenerativeBart,
    similarity_evaluator: SimilarityEvaluator,
    value_model: ValueModel,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    n_epochs: int,
    attacker_lr: float,
    value_lr: float,
    device: torch.device,
    max_len: int,
    save_dir: Path,
    call_parameters_save_path: Path,
    params_to_save: dict,
    n_max_train_batches: int | None = None,
):
    run_subdir_name = get_next_run_subdir_name(save_dir)
    save_dir = save_dir / run_subdir_name
    params_to_save.update({"save_dir": str(save_dir)})
    save_dir.mkdir(parents=True, exist_ok=True)
    call_parameters_save_path.parent.mkdir(parents=True, exist_ok=True)

    echo_optimizer = AdamW(echo.parameters(), lr=attacker_lr)
    value_optimizer = AdamW(value_model.parameters(), lr=value_lr)

    similarity_evaluator.eval()

    reward_function = partial(
        get_rewards_and_nonstandard_metrics,
        similarity_evaluator=similarity_evaluator,
        device=device,
    )
    ppo_trainer = PPOTrainer(
        echo,
        reward_function,
        value_model,
        echo_optimizer,
        value_optimizer,
        max_len,
        device,
        save_dir,
        call_parameters_save_path,
        params_to_save,
    )
    best_mean_final_rewards: float | None = None
    best_epoch = -1

    for i in tqdm(range(n_epochs), desc="training...", position=0):
        ppo_trainer.iteration(train_dataloader, TRAIN, n_max_batches=n_max_train_batches)
        with torch.no_grad():
            new_mean_final_rewards = ppo_trainer.iteration(eval_dataloader, EVAL)
        ppo_trainer.conclude_epoch()
        if best_mean_final_rewards is None or new_mean_final_rewards > best_mean_final_rewards:
            best_epoch = i
            best_mean_final_rewards = new_mean_final_rewards
            ppo_trainer.save_trained_models()

    ppo_trainer.save_stuff(best_epoch)


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
    save_dir: Path,
    call_parameters_save_path: Path,
    params_to_save: dict,
    n_max_train_samples: int | None = None,
):
    devices = get_available_torch_devices()
    if len(devices) > 1:
        trainer_device = devices[0]
        value_device = devices[1]
    else:
        trainer_device = devices[0]
        value_device = devices[0]
    similarity_evaluator = SimilarityEvaluator(similarity_evaluator_name, trainer_device)
    value_model = ValueModel(value_model_name, max_len, value_device)
    echo = GenerativeBart(source_model_name, max_len, [trainer_device])
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

    if n_max_train_samples is not None:
        n_max_train_batches = n_max_train_samples // batch_size
    else:
        n_max_train_batches = None

    train(
        echo,
        similarity_evaluator,
        value_model,
        train_dataloader,
        eval_dataloader,
        n_epochs,
        attacker_lr,
        value_lr,
        trainer_device,
        max_len,
        save_dir,
        call_parameters_save_path,
        params_to_save,
        n_max_train_batches,
    )


if __name__ == "__main__":
    echo_params = yaml.safe_load(open("params.yaml"))["src.echo.train_ppo"]

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
    n_max_train_samples: int | None = echo_params["n_max_train_samples"]

    save_dir = Path(echo_params["save_dir"])
    call_parameters_save_path = Path(echo_params["call_parameters_save_path"])

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
        save_dir,
        call_parameters_save_path,
        echo_params,
        n_max_train_samples,
    )

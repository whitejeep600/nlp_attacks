from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classifiers.similarity_evaluator import SimilarityEvaluator
from src.constants import EVAL, TRAIN
from src.datasets.sst2_dataset import SST2Dataset
from src.generation.dpo_trainer import DPOTrainer, RewardCalculator
from src.generation.generative_bart import GenerativeBart
from src.utils import get_available_torch_devices, get_next_run_subdir_name


def get_similarity_scores_and_nonstandard_metrics(
    prompt: str,
    generations: list[str],
    similarity_evaluator: SimilarityEvaluator,
) -> tuple[list[float], list[dict[str, float]]]:
    similarity_scores = similarity_evaluator.evaluate_many_to_one(generations, prompt)

    stats = [
        {
            "similarity_score": similarity_score,
        }
        for similarity_score in similarity_scores
    ]

    return similarity_scores, stats


class EchoMetricCalculator(RewardCalculator):
    def __init__(self, similarity_evaluator: SimilarityEvaluator):
        super().__init__()
        self.similarity_evaluator = similarity_evaluator

    def get_similarity_scores_and_nonstandard_metrics(
        self,
        prompt: str,
        generations: list[str],
    ) -> tuple[list[float], list[dict[str, float]]]:
        similarity_scores = self.similarity_evaluator.evaluate_many_to_one(generations, prompt)

        stats = [
            {
                "similarity_score": similarity_score,
            }
            for similarity_score in similarity_scores
        ]

        return similarity_scores, stats


def train(
    echo: GenerativeBart,
    similarity_evaluator: SimilarityEvaluator,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    n_epochs: int,
    attacker_lr: float,
    beta: float,
    temperature: float,
    reference_model: GenerativeBart,
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

    similarity_evaluator.eval()

    metric_calculator = EchoMetricCalculator(
        similarity_evaluator=similarity_evaluator,
    )
    dpo_trainer = DPOTrainer(
        trained_model=echo,
        metric_calculator=metric_calculator,
        trained_model_optimizer=echo_optimizer,
        beta=beta,
        temperature=temperature,
        attacker_lr=attacker_lr,
        max_len=max_len,
        reference_model=reference_model,
        save_dir=save_dir,
        call_parameters_save_path=call_parameters_save_path,
        params_to_save=params_to_save,
    )
    best_mean_final_rewards: float | None = None
    best_epoch = -1

    for i in tqdm(range(n_epochs), desc="training...", position=0):
        dpo_trainer.iteration(train_dataloader, TRAIN, n_max_batches=n_max_train_batches)
        with torch.no_grad():
            new_mean_final_rewards = dpo_trainer.iteration(eval_dataloader, EVAL)
        dpo_trainer.conclude_epoch()
        if best_mean_final_rewards is None or new_mean_final_rewards > best_mean_final_rewards:
            best_epoch = i
            best_mean_final_rewards = new_mean_final_rewards
            dpo_trainer.save_trained_models()

    dpo_trainer.save_stuff(best_epoch)


def main(
    source_model_name: str,
    similarity_evaluator_name: str,
    train_split_path: Path,
    eval_split_path: Path,
    max_len: int,
    batch_size: int,
    n_epochs: int,
    attacker_lr: float,
    beta: float,
    temperature,
    save_dir: Path,
    call_parameters_save_path: Path,
    params_to_save: dict,
    n_max_train_samples: int | None = None,
):
    devices = get_available_torch_devices()
    if len(devices) > 1:
        trained_model_device = devices[0]
        similarity_evaluator_device = devices[1]
        reference_model_device = devices[1]
    else:
        trained_model_device = devices[0]
        similarity_evaluator_device = devices[0]
        reference_model_device = devices[0]
    similarity_evaluator = SimilarityEvaluator(
        similarity_evaluator_name, similarity_evaluator_device
    )
    echo = GenerativeBart(source_model_name, max_len, [trained_model_device])
    reference_model = GenerativeBart(source_model_name, max_len, [reference_model_device])

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
        train_dataloader,
        eval_dataloader,
        n_epochs,
        attacker_lr,
        beta,
        temperature,
        reference_model,
        max_len,
        save_dir,
        call_parameters_save_path,
        params_to_save,
        n_max_train_batches,
    )


if __name__ == "__main__":
    echo_params = yaml.safe_load(open("params.yaml"))["src.echo.train_dpo"]

    source_model_name = echo_params["source_model_name"]
    similarity_evaluator_name = echo_params["similarity_evaluator_name"]

    train_split_path = Path(echo_params["train_split_path"])
    eval_split_path = Path(echo_params["eval_split_path"])

    max_len = int(echo_params["max_len"])
    batch_size = int(echo_params["batch_size"])
    n_epochs = int(echo_params["n_epochs"])
    attacker_lr = float(echo_params["attacker_lr"])
    beta = float(echo_params["beta"])
    temperature = float(echo_params["temperature"])
    n_max_train_samples: int | None = echo_params["n_max_train_samples"]

    save_dir = Path(echo_params["save_dir"])
    call_parameters_save_path = Path(echo_params["call_parameters_save_path"])

    main(
        source_model_name,
        similarity_evaluator_name,
        train_split_path,
        eval_split_path,
        max_len,
        batch_size,
        n_epochs,
        attacker_lr,
        beta,
        temperature,
        save_dir,
        call_parameters_save_path,
        echo_params,
        n_max_train_samples,
    )

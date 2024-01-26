from functools import partial
from pathlib import Path
from pickle import dump

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.sst2_dataset import SST2Dataset
from src.generation.generative_bart import GenerativeBart
from src.generation.ppo_trainer import PPOTrainer, TRAIN, EVAL
from src.generation.similarity_evaluator import SimilarityEvaluator, get_similarity_scores
from src.generation.value_model import ValueModel
from src.utils import get_available_torch_device


MEM_SNAPSHOT_SAVE_PATH = "snapshot.pickle"


def get_rewards_and_nonstandard_metrics(
        batch: dict,
        batch_prefixes: list[list[str]],
        original_seqs: list[str],
        similarity_evaluator: SimilarityEvaluator,
        device: str
) -> tuple[list[torch.Tensor], dict[str, float]]:
    similarity_scores = get_similarity_scores(batch_prefixes, original_seqs, similarity_evaluator, device)

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
        device: str,
        max_len: int,
        save_dir: Path
):
    save_dir.mkdir(parents=True, exist_ok=True)

    echo_optimizers = {p: AdamW([p], foreach=False, lr=attacker_lr) for p in echo.parameters()}
    value_optimizers = {p: AdamW([p], foreach=False, lr=value_lr) for p in value_model.parameters()}

    def echo_optimizer_hook(parameter) -> None:
        echo_optimizers[parameter].step()
        echo_optimizers[parameter].zero_grad()

    def value_optimizer_hook(parameter) -> None:
        value_optimizers[parameter].step()
        value_optimizers[parameter].zero_grad()

    for p in echo.parameters():
        p.register_post_accumulate_grad_hook(echo_optimizer_hook)

    for p in value_model.parameters():
        p.register_post_accumulate_grad_hook(value_optimizer_hook)

    similarity_evaluator.eval()

    reward_function = partial(
        get_rewards_and_nonstandard_metrics,
        similarity_evaluator=similarity_evaluator,
        device=device
    )
    ppo_trainer = PPOTrainer(
        echo,
        reward_function,
        value_model,
        max_len,
        device,
        save_dir
    )
    best_mean_final_rewards: float | None = None
    best_epoch = -1

    try:
        torch.cuda.memory._record_memory_history(enabled='all')
        for i in tqdm(range(n_epochs), desc="training...", position=0):
            ppo_trainer.iteration(train_dataloader, device, TRAIN)
            with torch.no_grad:
                new_mean_final_rewards = ppo_trainer.iteration(eval_dataloader, device, EVAL)
            if best_mean_final_rewards is None or new_mean_final_rewards > best_mean_final_rewards:
                best_epoch = i
                best_mean_final_rewards = new_mean_final_rewards
                ppo_trainer.save_trained_model()

        ppo_trainer.save_logs()
        ppo_trainer.save_summary(best_epoch)
        ppo_trainer.save_plots()
    except (RuntimeError, KeyboardInterrupt) as e:
        # This is often an OOM error
        #if 'memory' in str(e):
        if True:
            print(f"Caught an error suspected to be OOM error. Saving a memory"
                  f" snapshot to {MEM_SNAPSHOT_SAVE_PATH}\n")
            snapshot = torch.cuda.memory._snapshot()
            with open(MEM_SNAPSHOT_SAVE_PATH, "wb") as f:
                dump(snapshot, f)
        raise e


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
        save_dir: Path
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
        save_dir,
    )


if __name__ == "__main__":
    echo_params = yaml.safe_load(open("params.yaml"))["src.echo.train"]

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

    save_dir = Path(echo_params["save_dir"])

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
    )

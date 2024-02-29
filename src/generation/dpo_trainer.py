import copy
import time
from pathlib import Path
from typing import Callable

from torch.optim import Optimizer

from src.generation.generative_bart import GenerativeBart
from src.utils import get_current_git_commit_id


TRAIN = "train"
EVAL = "eval"
MODES = [TRAIN, EVAL]

POLICY_LOSS_METRIC = "policy_loss"


class DPOTrainer:
    def __init__(
        self,
        trained_model: GenerativeBart,
        rewards_and_metrics_function: Callable,
        trained_model_optimizer: Optimizer,
        max_len: int,
        device: str,
        save_dir: Path,
        call_parameters_save_path: Path,
        params_to_save: dict,
    ):
        self.trained_model = trained_model
        self.rewards_and_metrics_function = rewards_and_metrics_function
        self.reference_model = copy.deepcopy(self.trained_model)
        self.reference_model.eval()
        self.trained_model_optimizer = trained_model_optimizer
        self.max_len = max_len
        self.device = device
        self.save_dir = save_dir
        self.call_parameters_save_path = call_parameters_save_path
        self.standard_metric_names = [
            POLICY_LOSS_METRIC,
        ]
        self.all_data: dict[str, dict[str, list[list[float]]]] = {
            metric_name: {TRAIN: [], EVAL: []} for metric_name in self.standard_metric_names
        }
        self.train_start_time = time.time()
        self.epochs_elapsed = 0
        self.params_to_save = params_to_save
        self.params_to_save.update({"git_commit_id": get_current_git_commit_id()})

    def train(self) -> None:
        self.trained_model.train()
        self.reference_model.eval()

    def eval(self) -> None:
        self.trained_model.eval()
        self.reference_model.eval()

    def initialize_iteration(self, mode: str):
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        if mode == TRAIN:
            self.train()
        else:
            self.eval()


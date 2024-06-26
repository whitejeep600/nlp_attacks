from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt


def sequence_logprob(token_probabilities: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.log(token_probabilities)).reshape(1)


def get_length_without_padding(t: torch.Tensor, stop_token: int) -> int:
    stop_token_positions = (t[1:] == stop_token).nonzero()
    if stop_token_positions.numel() == 0:
        return len(t)
    else:
        return int((t[1:] == stop_token).nonzero()[0][0].item()) + 2


def get_available_torch_devices() -> list[torch.device]:
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        return [torch.device("cpu")]


def all_equal(values) -> bool:
    return len(values) == 0 or all([value == values[0] for value in values])


def get_command_output(command: str, arguments: list[str]) -> str:
    # removing the quotes around the output
    return (
        subprocess.run([command, *arguments], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()[1:-1]
    )


def get_current_git_commit_id() -> str:
    return get_command_output("git", ["log", '--format="%H"', "-n 1"])


def get_ceil_power_of_2(n: int) -> int:
    return 2 ** int(np.ceil(np.log(n) / np.log(2)))


# Just a util to automatically create a target list if it doesn't exist, yo
class ListDict:
    def __init__(self) -> None:
        self.lists: dict[str, list] = {}

    def append(self, list_name: str, item: Any) -> None:
        if list_name not in self.lists.keys():
            self.lists[list_name] = []
        self.lists[list_name].append(item)

    def __getitem__(self, item) -> Any:
        return self.lists[item]


def get_next_run_subdir_name(run_save_dir: Path) -> str:
    subdir_regex = re.compile("run_([0-9]+)")
    existing_run_nos: set[int] = set()
    for run_subdir in run_save_dir.iterdir():
        subdir_name = run_subdir.name
        match = subdir_regex.match(subdir_name)
        if match is not None:
            run_no = match.groups()[0]
            existing_run_nos.add(int(run_no))
    if not existing_run_nos:
        next_no = 0
    else:
        next_no = max(existing_run_nos) + 1
    return f"run_{next_no}"


def round_list(target_list: list[float]) -> list[float]:
    return [round(x, 3) for x in target_list]


def harmonic_mean(numbers: list[float], weights: list[float] | None = None) -> float:
    numbers_array = np.array(numbers)
    if weights is None:
        weights_array = np.ones_like(numbers_array)
    else:
        weights_array = np.array(weights)
    return weights_array.sum() / (weights_array / numbers_array).sum()


def assign_gpu_devices() -> tuple[torch.device, torch.device, torch.device]:
    devices = get_available_torch_devices()
    if len(devices) > 1:
        evaluator_models_device = devices[1]
        reference_model_device = devices[1]
        generator_device = devices[0]
    else:
        generator_device = devices[0]
        evaluator_models_device = devices[0]
        reference_model_device = devices[0]
    return generator_device, reference_model_device, evaluator_models_device


GAN_THRESHOLD = 0.6
BASE_AT_GAN_THRESHOLD = 0.48
LIMIT_AT_GAN_THRESHOLD = 0.5
BASE_AT_1 = 0.5
LIMIT_AT_1 = 1


def get_base(naturalness: float) -> float:
    if naturalness < GAN_THRESHOLD:
        return BASE_AT_GAN_THRESHOLD * (naturalness / GAN_THRESHOLD)
    elif naturalness < 1:
        return BASE_AT_GAN_THRESHOLD + (BASE_AT_1 - BASE_AT_GAN_THRESHOLD) * (
            naturalness - GAN_THRESHOLD
        ) / (1 - GAN_THRESHOLD)
    else:
        return BASE_AT_1


def get_limit(naturalness: float) -> float:
    if naturalness < GAN_THRESHOLD:
        return LIMIT_AT_GAN_THRESHOLD * (naturalness / GAN_THRESHOLD)
    elif naturalness < 1:
        return LIMIT_AT_GAN_THRESHOLD + (LIMIT_AT_1 - LIMIT_AT_GAN_THRESHOLD) * (
            naturalness - GAN_THRESHOLD
        ) / (1 - GAN_THRESHOLD)
    else:
        return LIMIT_AT_1


def plot_reward_base_and_limit() -> None:
    xs = np.linspace(0, 2, 100)
    limits = [get_limit(x) for x in xs]
    bases = [get_base(x) for x in xs]
    plt.plot(xs, limits)
    plt.plot(xs, bases)
    plt.show()

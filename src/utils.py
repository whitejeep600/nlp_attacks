import subprocess
from typing import Any

import numpy as np
import torch


def get_available_torch_devices() -> list[str]:
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        return ["cpu"]


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

import torch


def get_available_torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def all_equal(values) -> bool:
    return len(values) == 0 or all([value == values[0] for value in values])


# Just a util to automatically create a target list if it doesn't exist, yo
class ListDict:
    def __init__(self):
        self.lists: dict[str, list] = {}

    def append(self, list_name: str, item):
        if list_name not in self.lists.keys():
            self.lists[list_name] = []
        self.lists[list_name].append(item)

    def __getitem__(self, item):
        return self.lists[item]
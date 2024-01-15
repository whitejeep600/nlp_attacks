from pathlib import Path

from datasets import load_dataset


def main() -> None:
    target_path = Path("data/sst2")
    target_path.mkdir(exist_ok=True, parents=True)
    for split in ['train', 'validation', 'test']:
        print(f"Downloading split {split}...\n")
        split_data = load_dataset('glue', 'sst2', split=split)
        split_data.to_csv(target_path / f"{split}.csv")
    print("All downloaded.\n")


if __name__ == '__main__':
    main()

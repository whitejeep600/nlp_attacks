import os
from pathlib import Path

from tqdm import tqdm


def main(
        benchmarked_models: list[str],
        algorithms: list[str]
):
    with tqdm(
            total= len(benchmarked_models) * len(algorithms),
            desc='Generating benchmarks...'
    ) as pbar:
        for model_name in benchmarked_models:
            for algorithm_name in algorithms:
                subdir = f"{model_name}/{algorithm_name}"
                target_path = Path("data/") / subdir / "log.csv"
                if target_path.exists():
                    print(f"Skipping the generation of the {subdir} benchmark, since the related"
                          f" data found already present.")
                    pbar.update(1)
                    continue
                else:
                    print(f"Generating the {subdir} benchmark...")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                os.system(
                    f"textattack attack"
                    f" --model {model_name}"
                    f" --recipe {algorithm_name}"
                    f" --num-examples -1"
                    f" --log-to-csv {target_path}"
                    f" --disable-stdout"
                    f" --csv-coloring-style plain"
                )
                pbar.update(1)


if __name__ == '__main__':

    # This refers to pretrained model names accessed via the textattack package.
    # The full list of potential models for the sst2 dataset can be obtained, for
    # example, by running
    #   textattack list models | grep sst
    benchmarked_models = ["cnn-sst2d", "bert-base-uncased-sst2", "roberta-base-sst2"]

    # Again, these are names from textattack. For the full list, run
    #   textattack list attack-recipes
    algorithms = ["hotflip", "pso", "deepwordbug", "textfooler", "bert-attack"]
    main(benchmarked_models, algorithms)

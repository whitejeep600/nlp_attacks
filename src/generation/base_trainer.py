from __future__ import annotations

import copy
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import mean

from src.constants import EVAL, MODES, PLOT_AVG_WINDOW_LENGTH, TRAIN
from src.utils import ListDict, get_current_git_commit_id


class Trainer:
    def __init__(
        self,
        standard_metric_names: list[str],
        save_dir: Path,
        general_training_log_path: Path,
        params_to_save: dict,
        max_len: int,
    ):
        self.standard_metric_names = standard_metric_names
        self.all_data: dict[str, dict[str, list[np.ndarray]]] = {
            metric_name: {TRAIN: [], EVAL: []} for metric_name in self.standard_metric_names
        }
        self.epochs_elapsed = 0
        self.save_dir = save_dir
        self.train_start_time = time.time()
        self.general_training_log_path = general_training_log_path
        self.params_to_save = params_to_save
        self.params_to_save.update({"git_commit_id": get_current_git_commit_id()})
        self.max_len = max_len

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError

    def initialize_iteration(self, mode: str):
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        if mode == TRAIN:
            self.train()
        else:
            self.eval()

    def add_epoch_metrics(
        self,
        epoch_metrics: dict[str, list[float]],
        mode: str,
    ) -> None:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        for metric_name in epoch_metrics.keys():
            if metric_name in self.standard_metric_names:
                if mode == EVAL:
                    new_metrics = [float(mean(epoch_metrics[metric_name]))]
                else:
                    new_metrics = epoch_metrics[metric_name]
                self.all_data[metric_name][mode].append(np.array(new_metrics))
            else:
                warnings.warn(f"Metric '{metric_name}' is not standard, ignoring.")
        for metric_name in self.standard_metric_names:
            if metric_name not in epoch_metrics:
                warnings.warn(f"Metric '{metric_name}' was not passed for this iteration!")

    def add_nonstandard_epoch_metrics(
        self, epoch_metrics: dict[str, list[float]], mode: str
    ) -> None:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        for metric_name in epoch_metrics.keys():
            if metric_name not in self.all_data:
                self.all_data[metric_name] = {TRAIN: [], EVAL: []}
            # For eval iterations, we are not interested in how the metrics change within
            # one iteration, because they don't include changing the model's parameters.
            if mode == EVAL:
                new_metrics = [float(mean(epoch_metrics[metric_name]))]
            else:
                new_metrics = epoch_metrics[metric_name]
            self.all_data[metric_name][mode].append(np.array(new_metrics))

    def save_generated_eval_sentences(
        self,
        original_sentences: list[str],
        generated_sentences: list[str],
        metrics: ListDict | None = None,
    ) -> None:
        generated_sentences_path = self.save_dir / "generated_sentences"
        generated_sentences_path.mkdir(parents=True, exist_ok=True)
        current_save_path = generated_sentences_path / f"epoch_{self.epochs_elapsed}.csv"
        dict_to_save: dict[str, list[str] | list[float]] = {
            "original": original_sentences,
            "generated": generated_sentences,
        }
        if metrics is not None:
            dict_to_save.update(
                {metric_name: metrics[metric_name] for metric_name in metrics.lists.keys()}
            )
        df = pd.DataFrame(dict_to_save)
        df.to_csv(current_save_path)

    def conclude_epoch(self) -> None:
        all_metric_names = list(self.all_data.keys())
        average_metrics = {
            mode: {
                key: round(self.all_data[key][mode][self.epochs_elapsed].mean(), 3)
                for key in all_metric_names
            }
            for mode in MODES
        }
        print(f"\nEpoch {self.epochs_elapsed}.\n")
        for mode in MODES:
            print(f"This epoch's {mode} stats are as follows:\n" f"{average_metrics[mode]}\n")
        self.epochs_elapsed += 1

    def get_epoch_stats(self, epoch_no: int) -> dict:
        return {
            key: round(float(mean(self.all_data[key][EVAL][epoch_no])), 3)
            for key in self.all_data.keys()
        }

    def save_summary(self, best_epoch_no: int) -> None:
        time_now = time.time()
        time_elapsed = time.gmtime(time_now - self.train_start_time)

        summary_path = self.save_dir / "summary.txt"
        best_epoch_stats = self.get_epoch_stats(best_epoch_no)
        summary = (
            f"Training time: {time.strftime('%H:%M:%S', time_elapsed)}"
            f" Number of epochs elapsed: {self.epochs_elapsed}, best stats"
            f" for epoch {best_epoch_no}, as follows: {best_epoch_stats}\n"
        )
        with open(summary_path, "w") as f:
            f.write(summary)

    def save_logs(self) -> None:
        logs_path = self.save_dir / "log.txt"
        with open(logs_path, "w") as f:
            serializable_all_data = {
                metric_name: {
                    mode: [
                        metric_mode_epoch_stats.tolist()
                        for metric_mode_epoch_stats in self.all_data[metric_name][mode]
                    ]
                    for mode in self.all_data[metric_name].keys()
                }
                for metric_name in self.all_data.keys()
            }
            f.write(json.dumps(serializable_all_data, indent=4))

    def save_plots(self) -> None:
        plots_path = self.save_dir / "plots"
        plots_path.mkdir(parents=True, exist_ok=True)
        for variable in self.all_data.keys():
            for mode in MODES:
                plotted_data = self.all_data[variable][mode]
                if not all([len(data) == len(plotted_data[0]) for data in plotted_data]):
                    warnings.warn("Some logged data had inconsistent lengths per epoch.")
                if mode == TRAIN:
                    # Averaging metrics per a small group of consecutive batches. Then plotting
                    # the scores for every group.
                    averaged_data = [
                        epoch_data[
                            np.arange(epoch_data.shape[0] - PLOT_AVG_WINDOW_LENGTH + 1)[:, None]
                            + np.arange(PLOT_AVG_WINDOW_LENGTH)
                        ].mean(axis=1)
                        for epoch_data in plotted_data
                    ]
                    ys = [y for epoch_data in averaged_data for y in epoch_data]
                else:
                    # We want a single, average metric per epoch. This was already averaged
                    # in the add_epoch_metrics function.
                    ys = [y for epoch_data in plotted_data for y in epoch_data]
                xs = np.linspace(0, len(plotted_data), len(ys))
                title = f"{mode}_{variable}"
                plt.title(title)
                plt.plot(xs, ys, linewidth=0.5)
                plt.xlabel("iteration")
                plt.savefig(plots_path / f"{title}.jpg", dpi=256)
                plt.clf()

    def save_call_parameters_and_summary(self, best_epoch_no: int) -> None:
        # For comparison, we likely want to save these parameters together for all
        # model runs, as a general log of the experiment process.
        to_save = copy.deepcopy(self.params_to_save)
        to_save["summary"] = self.get_epoch_stats(best_epoch_no)
        with open(self.general_training_log_path, "a") as f:
            f.write(f"{json.dumps(self.params_to_save, indent=2)}\n\n")

    def save_stuff(self, best_epoch_no: int) -> None:
        print(f"Saving stuff to {self.save_dir}")
        self.save_logs()
        self.save_summary(best_epoch_no)
        self.save_plots()
        self.save_call_parameters_and_summary(best_epoch_no)

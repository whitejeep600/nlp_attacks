import copy
import json
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy import mean
from torch.nn.functional import logsigmoid
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.generation.generative_bart import GenerativeBart
from src.utils import ListDict, get_current_git_commit_id

TRAIN = "train"
EVAL = "eval"
MODES = [TRAIN, EVAL]

POLICY_LOSS_METRIC = "policy_loss"

PLOT_AVG_WINDOW_LENGTH = 16


class SampleGenerations:
    def __init__(
        self,
        prompt: str,
        prompt_tokens: torch.Tensor,
        generation_texts: list[str],
        generation_tokens: list[torch.Tensor],
        generation_probabilities: list[torch.Tensor],
        generation_reference_probabilities: list[torch.Tensor],
        generation_scores: list[float],
        generation_metrics: list[dict[str, float]],
    ):
        # Afterwards it is assumed that the generations for a given prompt (and all their
        # corresponding attributes) are ordered by increasing score (reward).
        ordering = np.argsort(generation_scores)
        self.prompt = prompt
        self.prompt_tokens = prompt_tokens
        self.generation_scores = [generation_scores[i] for i in ordering]
        self.generation_texts = [generation_texts[i] for i in ordering]
        self.generation_tokens = [generation_tokens[i] for i in ordering]
        self.generation_probabilities = [generation_probabilities[i] for i in ordering]
        self.generation_reference_probabilities = [
            generation_reference_probabilities[i] for i in ordering
        ]
        self.generation_metrics = [generation_metrics[i] for i in ordering]


class DPOTrainer:
    def __init__(
        self,
        trained_model: GenerativeBart,
        rewards_and_metrics_function: Callable,
        trained_model_optimizer: Optimizer,
        beta: float,
        max_len: int,
        device: str,
        save_dir: Path,
        call_parameters_save_path: Path,
        params_to_save: dict,
    ):
        # Beta - common notation for the term determining the influence of the penalty
        # for Kullback-Leibler divergence between the trained and reference policy.
        self.trained_model = trained_model
        self.rewards_and_metrics_function = rewards_and_metrics_function
        self.reference_model = copy.deepcopy(self.trained_model)
        self.reference_model.eval()
        self.trained_model_optimizer = trained_model_optimizer
        self.beta = beta
        self.max_len = max_len
        self.device = device
        self.save_dir = save_dir
        self.call_parameters_save_path = call_parameters_save_path
        self.standard_metric_names = [
            POLICY_LOSS_METRIC,
        ]
        self.all_data: dict[str, dict[str, list[np.ndarray]]] = {
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
        average_epoch_metrics = {
            key: float(mean(epoch_metrics[key])) for key in epoch_metrics.keys()
        }
        print(
            f"Epoch {self.epochs_elapsed},"
            f" this epoch's {mode} stats, as follows:\n"
            f"{average_epoch_metrics}\n"
        )

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
        self, original_sentences: list[str], generated_sentences: list[str]
    ) -> None:
        generated_sentences_path = self.save_dir / "generated_sentences"
        generated_sentences_path.mkdir(parents=True, exist_ok=True)
        current_save_path = generated_sentences_path / f"epoch_{self.epochs_elapsed}.csv"
        df = pd.DataFrame({"original": original_sentences, "generated": generated_sentences})
        df.to_csv(current_save_path)

    def sample_two_generations(
        self,
        batch_input_ids: torch.Tensor,
        batch_original_seqs: list[str],
    ) -> list[SampleGenerations]:
        """

        :param batch_input_ids: shape (batch_size, seq_len)
        :param batch_original_seqs: list of strings representing the original batch sequences

        """
        batch_input_ids = batch_input_ids.to(self.device)
        all_generations: list[SampleGenerations] = []
        for sample_original_seq, sample_input_ids in zip(batch_original_seqs, batch_input_ids):
            generation_0, probs_0 = self.trained_model.generate_with_random_sampling(
                sample_input_ids.unsqueeze(0), self.max_len
            )
            generation_1, probs_1 = self.trained_model.generate_with_random_sampling(
                sample_input_ids.unsqueeze(0), self.max_len
            )
            with torch.no_grad():
                reference_probs_0 = self.reference_model.get_reference_probabilities(
                    sample_input_ids,
                    generation_0,
                )
                reference_probs_1 = self.reference_model.get_reference_probabilities(
                    sample_input_ids,
                    generation_1,
                )
            text_0, text_1 = self.trained_model.decode([generation_0, generation_1])
            score_0, metrics_0 = self.rewards_and_metrics_function(sample_original_seq, text_0)
            score_1, metrics_1 = self.rewards_and_metrics_function(sample_original_seq, text_1)
            generations = SampleGenerations(
                prompt=sample_original_seq,
                prompt_tokens=sample_input_ids,
                generation_texts=[text_0, text_1],
                generation_tokens=[generation_0, generation_1],
                generation_probabilities=[probs_0, probs_1],
                generation_reference_probabilities=[reference_probs_0, reference_probs_1],
                generation_scores=[score_0, score_1],
                generation_metrics=[metrics_0, metrics_1],
            )
            all_generations.append(generations)
        return all_generations

    def get_batch_policy_loss(self, batch_generations: list[SampleGenerations]) -> torch.Tensor:
        better_seq_model_probabilities = torch.cat(
            [
                torch.prod(generation.generation_probabilities[1]).reshape(1)
                for generation in batch_generations
            ],
            dim=-1,
        )
        worse_seq_model_probabilities = torch.cat(
            [
                torch.prod(generation.generation_probabilities[0]).reshape(1)
                for generation in batch_generations
            ],
            dim=-1,
        )
        better_seq_reference_probabilities = torch.cat(
            [
                torch.prod(generation.generation_reference_probabilities[1]).reshape(1)
                for generation in batch_generations
            ],
            dim=-1,
        )
        worse_seq_reference_probabilities = torch.cat(
            [
                torch.prod(generation.generation_reference_probabilities[0]).reshape(1)
                for generation in batch_generations
            ],
            dim=-1,
        )
        better_seq_logratios = torch.log(better_seq_model_probabilities) - torch.log(
            better_seq_reference_probabilities
        )
        worse_seq_logratios = torch.log(worse_seq_model_probabilities) - torch.log(
            worse_seq_reference_probabilities
        )
        return -1 * logsigmoid(self.beta * (better_seq_logratios - worse_seq_logratios)).mean()

    def policy_loss_step(self, policy_loss: torch.Tensor):
        self.trained_model_optimizer.zero_grad()
        policy_loss.backward()
        self.trained_model_optimizer.step()

    def save_trained_models(self) -> None:
        torch.save(self.trained_model.bert.state_dict(), self.save_dir / "generator_ckpt.pt")

    def iteration(
        self, dataloader: DataLoader, mode: str, n_max_batches: int | None = None
    ) -> float:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"

        self.initialize_iteration(mode)
        all_original_seqs: list[str] = []
        all_generated_sentences: list[str] = []

        epoch_policy_losses: list[float] = []
        epoch_generation_scores: list[float] = []
        nonstandard_metrics = ListDict()

        for batch_no, batch in tqdm(
            enumerate(dataloader),
            total=n_max_batches if n_max_batches is not None else len(dataloader),
            desc=f"{mode} iteration",
            leave=False,
            position=1,
        ):
            input_ids = batch["input_ids"].to(self.device)
            original_seqs = batch["original_seq"]
            generations = self.sample_two_generations(input_ids, original_seqs)

            # todo length checks maybe

            generation_metric_keys = generations[0].generation_metrics[0].keys()
            for metric_key in generation_metric_keys:
                nonstandard_metrics.append(
                    metric_key,
                    float(
                        mean(
                            [
                                single_generation_metrics[metric_key]
                                for generation in generations
                                for single_generation_metrics in generation.generation_metrics
                            ]
                        )
                    ),
                )

            policy_loss = self.get_batch_policy_loss(generations)
            if mode == TRAIN:
                self.policy_loss_step(policy_loss)

            epoch_policy_losses.append(policy_loss.item())
            epoch_generation_scores.append(
                float(
                    mean(
                        [
                            score
                            for generation in generations
                            for score in generation.generation_scores
                        ]
                    )
                )
            )

            if mode == EVAL:
                all_original_seqs += original_seqs
                all_generated_sentences += [
                    generation.generation_texts[0] for generation in generations
                ]
            if batch_no == n_max_batches:
                break

        mean_generation_score = float(mean(epoch_generation_scores))
        self.add_epoch_metrics(
            {
                POLICY_LOSS_METRIC: epoch_policy_losses,
            },
            mode,
        )
        self.add_nonstandard_epoch_metrics(
            {
                list_name: nonstandard_metrics[list_name]
                for list_name in nonstandard_metrics.lists.keys()
            },
            mode,
        )
        if mode == EVAL:
            self.save_generated_eval_sentences(all_original_seqs, all_generated_sentences)

        return mean_generation_score

    def conclude_epoch(self) -> None:
        self.epochs_elapsed += 1

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

    def save_summary(self, best_epoch_no: int) -> None:
        time_now = time.time()
        time_elapsed = time.gmtime(time_now - self.train_start_time)

        summary_path = self.save_dir / "summary.txt"
        best_epoch_stats = {
            key: float(mean(self.all_data[key][EVAL][best_epoch_no]))
            for key in self.all_data.keys()
        }
        summary = (
            f"Training time: {time.strftime('%H:%M:%S', time_elapsed)}"
            f" Number of epochs elapsed: {self.epochs_elapsed}, best stats (final rewards)"
            f" for epoch {best_epoch_no}, as follows: {best_epoch_stats}\n"
        )
        with open(summary_path, "w") as f:
            f.write(summary)

    def save_plots(self) -> None:
        plots_path = self.save_dir / "plots"
        plots_path.mkdir(parents=True, exist_ok=True)
        for variable in self.all_data.keys():
            for mode in MODES:
                plotted_data = self.all_data[variable][mode]
                if not all([len(data) == len(plotted_data[0]) for data in plotted_data]):
                    warnings.warn("Some logged data had inconsistent lengths per epoch.")
                if mode == TRAIN:
                    averaged_data = [
                        epoch_data[
                            np.arange(epoch_data.shape[0] - PLOT_AVG_WINDOW_LENGTH + 1)[:, None]
                            + np.arange(PLOT_AVG_WINDOW_LENGTH)
                        ].mean(axis=1)
                        for epoch_data in plotted_data
                    ]
                    ys = [y for epoch_data in averaged_data for y in epoch_data]
                else:
                    ys = [y for epoch_data in plotted_data for y in epoch_data]
                xs = np.linspace(0, len(plotted_data), len(ys))
                title = f"{mode}_{variable}"
                plt.title(title)
                plt.plot(xs, ys, linewidth=0.5)
                plt.xlabel("iteration")
                plt.savefig(plots_path / f"{title}.jpg", dpi=256)
                plt.clf()

    def save_call_parameters(self) -> None:
        # For comparison, we likely want to save these parameters together for all
        # model runs, as a general log of the experiment process.
        with open(self.call_parameters_save_path, "a") as f:
            f.write(f"{json.dumps(self.params_to_save, indent=2)}\n\n")

    def save_stuff(self, best_epoch_no: int) -> None:
        print(f"Saving stuff to {self.save_dir}")
        self.save_logs()
        self.save_summary(best_epoch_no)
        self.save_plots()
        self.save_call_parameters()

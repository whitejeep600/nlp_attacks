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
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.generation.generative_bart import GenerativeBart
from src.generation.value_model import ValueModel
from src.utils import ListDict, get_current_git_commit_id

# Discount factor, following the notation from the original PPO paper by Schulman et al.
GAMMA = 0.99

# Generalized Advantage Estimation factor
LAMBDA = 0.95

# Policy update clipping threshold
EPSILON = 0.2

# 2000 years and we're still using Greek if we want something to sound smart.

TRAIN = "train"
EVAL = "eval"
MODES = [TRAIN, EVAL]

REWARD_METRIC = "reward"
POLICY_LOSS_METRIC = "policy_loss"
VALUE_LOSS_METRIC = "value_loss"


class PPOTrainer:
    def __init__(
        self,
        trained_model: GenerativeBart,
        rewards_and_metrics_function: Callable,
        value_model: ValueModel,
        trained_model_optimizer: Optimizer,
        value_model_optimizer: Optimizer,
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
        self.value_model = value_model
        self.trained_model_optimizer = trained_model_optimizer
        self.value_model_optimizer = value_model_optimizer
        self.max_len = max_len
        self.device = device
        self.save_dir = save_dir
        self.call_parameters_save_path = call_parameters_save_path
        self.standard_metric_names = [
            REWARD_METRIC,
            POLICY_LOSS_METRIC,
            VALUE_LOSS_METRIC,
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
        self.value_model.train()
        self.reference_model.eval()

    def eval(self) -> None:
        self.trained_model.eval()
        self.value_model.eval()
        self.reference_model.eval()

    def freeze_reference_model(self):
        self.reference_model = copy.deepcopy(self.trained_model)
        self.reference_model.eval()

    def initialize_iteration(self, mode: str):
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        if mode == TRAIN:
            self.train()
        else:
            self.eval()

    def decode_prefixes(self, generated_ids: list[torch.Tensor]) -> list[list[str]]:
        return self.trained_model.decode_prefixes(generated_ids)

    def decode_tokens_and_get_logits(
        self, batch: torch.Tensor, max_length: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        batch = batch.to(self.device)
        torch.set_grad_enabled(True)
        generated_ids: list[torch.Tensor] = []
        token_probabilities: list[torch.Tensor] = []
        reference_probabilities: list = []
        for seq in batch:
            new_ids, scores = self.trained_model.generate_with_greedy_decoding(
                seq.unsqueeze(0), max_length
            )
            new_ids = new_ids.to(self.device)
            scores = [score.to(self.device) for score in scores]
            generated_ids.append(new_ids)
            new_token_probabilites = torch.stack(
                [torch.softmax(scores[i][0], dim=0)[new_ids[i + 1]] for i in range(len(scores))],
            )
            with torch.no_grad():
                new_reference_probabilites = self.reference_model.get_reference_probabilities(
                    seq,
                    new_ids,
                )
            token_probabilities.append(new_token_probabilites)
            reference_probabilities.append(new_reference_probabilites)
        return generated_ids, token_probabilities, reference_probabilities

    def get_value_function_scores(
        self, batch_prefixes: list[list[str]], original_seqs: list[str]
    ) -> list[torch.Tensor]:
        return [
            torch.cat(
                [self.value_model.get_value(prefix, original_seq) for prefix in sample_prefixes]
            ).to(self.device)
            for sample_prefixes, original_seq in zip(batch_prefixes, original_seqs)
        ]

    def get_clipped_objectives(
        self,
        rewards: list[torch.Tensor],
        values: list[torch.Tensor],
        token_probs: list[torch.Tensor],
        reference_probs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        max_generated_length = max([len(reward_tensor) for reward_tensor in rewards])
        with torch.no_grad():
            discount_exponents = torch.pow(GAMMA * LAMBDA, torch.arange(max_generated_length)).to(
                self.device
            )
            # Following the notation and equations from Schulman et al.
            batch_size = len(rewards)
            gammas = [
                rewards[i][:-1] + GAMMA * values[i][1:] - values[i][:-1] for i in range(batch_size)
            ]
            advantages = [
                torch.stack(
                    [
                        torch.sum(
                            gammas[batch_index][t:]
                            * discount_exponents[: len(gammas[batch_index][t:])]
                        )
                        for t in range(len(rewards[batch_index]) - 1)
                    ],
                    dim=0,
                )
                for batch_index in range(batch_size)
            ]
        ratios = [token_probs[i][:-1] / reference_probs[i][:-1] for i in range(batch_size)]
        clipped_ratios = [torch.clip(ratio, 1 - EPSILON, 1 + EPSILON) for ratio in ratios]
        clipped_objectives = [
            torch.minimum(ratios[i] * advantages[i], clipped_ratios[i] * advantages[i])
            for i in range(batch_size)
        ]
        return clipped_objectives

    def get_policy_loss(self, clipped_objectives: list[torch.Tensor]) -> torch.Tensor:
        # gradient ascent
        policy_loss = -1 * torch.mean(torch.concat(clipped_objectives))
        return policy_loss

    def policy_loss_backprop(self, policy_loss: torch.Tensor, step: bool = False) -> None:
        if step:
            self.trained_model_optimizer.zero_grad()
            policy_loss.backward()
            self.trained_model_optimizer.step()
        else:
            policy_loss.backward()

    def get_value_loss(
        self, rewards: list[torch.Tensor], values: list[torch.Tensor]
    ) -> torch.Tensor:
        value_loss = torch.mean(
            torch.abs(
                torch.concat([values[i] - rewards[i][-1].detach() for i in range(len(values))])
            )
        )
        return value_loss

    def value_loss_backprop(self, value_loss: torch.Tensor, step: bool = False) -> None:
        if step:
            self.value_model_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.value_model_optimizer.step()
        else:
            value_loss.backward(retain_graph=True)

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
                self.all_data[metric_name][mode].append(new_metrics)
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
            self.all_data[metric_name][mode].append(new_metrics)

    def save_logs(self) -> None:
        logs_path = self.save_dir / "log.txt"
        with open(logs_path, "w") as f:
            f.write(json.dumps(self.all_data, indent=4))

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
                ys = [y for epoch_data in plotted_data for y in epoch_data]
                xs = np.linspace(0, len(plotted_data), len(ys))
                title = f"{mode}_{variable}"
                plt.title(title)
                plt.plot(xs, ys, linewidth=0.5)
                plt.xlabel("iteration")
                plt.savefig(plots_path / f"{title}.jpg", dpi=256)
                plt.clf()

    def save_trained_model(self) -> None:
        torch.save(self.trained_model.bert.state_dict(), self.save_dir / "generator_ckpt.pt")
        torch.save(self.value_model.model.state_dict(), self.save_dir / "value_ckpt.pt")

    def save_generated_eval_sentences(
        self, original_sentences: list[str], generated_sentences: list[str]
    ) -> None:
        generated_sentences_path = self.save_dir / "generated_sentences"
        generated_sentences_path.mkdir(parents=True, exist_ok=True)
        current_save_path = generated_sentences_path / f"epoch_{self.epochs_elapsed}.csv"
        df = pd.DataFrame({"original": original_sentences, "generated": generated_sentences})
        df.to_csv(current_save_path)

    def save_call_parameters(self) -> None:
        # For comparison, we likely want to save these parameters together for all
        # model runs, as a general log of the experiment process.
        with open(self.call_parameters_save_path, "a") as f:
            f.write(f"{json.dumps(self.params_to_save, indent=2)}\n\n")

    # The three most common programming errors are infinite recursion and off-by-one.
    def run_length_checks(
        self,
        batch_prefixes: list[list[str]],
        generated_ids: list[torch.Tensor],
        token_probs: list[torch.Tensor],
        reference_probs: list[torch.Tensor],
    ) -> None:
        for sample_no in range(len(batch_prefixes)):
            if not len(generated_ids[sample_no]) - 1 == len(batch_prefixes[sample_no]):
                warnings.warn(
                    f"Unexpected generation length, generated ids {generated_ids[sample_no]}, "
                    f" batch prefixes {batch_prefixes}"
                )

            if not len(token_probs[sample_no]) == len(reference_probs[sample_no]):
                warnings.warn(
                    f"Token generated and reference probs differ in length,"
                    f" generated {token_probs[sample_no]}, reference {reference_probs[sample_no]}\n"
                )
            if not len(token_probs[sample_no]) == len(generated_ids[sample_no]) - 1:
                warnings.warn(
                    f"Expected token generation probabilities for all generated tokens"
                    f" except the start token, got generated token ids {generated_ids[sample_no]},"
                    f" probs {token_probs[sample_no]}\n"
                )

    def iteration(
        self, dataloader: DataLoader, mode: str, n_max_batches: int | None = None
    ) -> float:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"

        self.initialize_iteration(mode)
        if mode == TRAIN:
            self.freeze_reference_model()
        all_original_seqs: list[str] = []
        all_generated_sentences: list[str] = []

        epoch_policy_losses: list[float] = []
        epoch_value_losses: list[float] = []
        epoch_rewards: list[float] = []
        nonstandard_metrics = ListDict()

        for batch_no, batch in tqdm(
            enumerate(dataloader),
            total=n_max_batches if n_max_batches is not None else len(dataloader),
            desc=f"{mode} iteration",
            leave=False,
            position=1,
        ):
            input_ids = batch["input_ids"].to(self.device)
            generated_ids, token_probs, reference_probs = self.decode_tokens_and_get_logits(
                input_ids, self.max_len
            )
            batch_prefixes = self.decode_prefixes(generated_ids)
            self.run_length_checks(batch_prefixes, generated_ids, token_probs, reference_probs)

            original_seqs = batch["original_seq"]

            with torch.no_grad():
                rewards, stats = self.rewards_and_metrics_function(
                    batch, batch_prefixes, original_seqs
                )

            for stat_name in stats.keys():
                nonstandard_metrics.append(stat_name, stats[stat_name])

            values = self.get_value_function_scores(batch_prefixes, original_seqs)

            clipped_objectives = self.get_clipped_objectives(
                rewards, values, token_probs, reference_probs
            )

            policy_loss = self.get_policy_loss(clipped_objectives)
            value_loss = self.get_value_loss(rewards, values)

            if mode == TRAIN:
                step = batch_no % 4 == 0
                self.policy_loss_backprop(policy_loss, step)
                self.value_loss_backprop(value_loss, step)

            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())

            final_rewards = [reward[-1].item() for reward in rewards]

            epoch_rewards.append(float(mean(final_rewards)))
            if mode == EVAL:
                all_original_seqs += original_seqs
                all_generated_sentences += [prefixes[-1] for prefixes in batch_prefixes]
            if batch_no == n_max_batches:
                break

        mean_final_reward = float(mean(epoch_rewards))
        self.add_epoch_metrics(
            {
                REWARD_METRIC: epoch_rewards,
                POLICY_LOSS_METRIC: epoch_policy_losses,
                VALUE_LOSS_METRIC: epoch_value_losses,
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

        return mean_final_reward

    def conclude_epoch(self) -> None:
        self.epochs_elapsed += 1

    def save_stuff(self, best_epoch_no: int) -> None:
        print(f"Saving stuff to {self.save_dir}")
        self.save_call_parameters()
        self.save_summary(best_epoch_no)
        self.save_plots()
        self.save_call_parameters()

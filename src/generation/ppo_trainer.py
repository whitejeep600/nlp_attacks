import copy
import json
import time
import warnings
from pathlib import Path
from typing import Callable

import torch
from matplotlib import pyplot as plt
from numpy import mean
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.generation.generative_bart import GenerativeBart
from src.generation.value_model import ValueModel
from src.utils import ListDict, all_equal

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
        value_optimizer: Optimizer,
        max_len: int,
        device: str,
        stats_save_dir: Path
    ):
        self.trained_model = trained_model
        self.rewards_and_metrics_function = rewards_and_metrics_function
        self.reference_model = copy.deepcopy(self.trained_model)
        self.reference_model.eval()
        self.trained_model.bert.to(device)
        self.reference_model.bert.to(device)
        self.value_model = value_model
        self.trained_model_optimizer = trained_model_optimizer
        self.value_optimizer = value_optimizer
        self.max_len = max_len
        self.device = device
        self.save_dir = stats_save_dir
        self.standard_metric_names = [
            REWARD_METRIC,
            POLICY_LOSS_METRIC,
            VALUE_LOSS_METRIC,
        ]
        self.all_data: dict[str, dict[str, list[float]]] = {
            metric_name: {
                TRAIN: [],
                EVAL: []
            }
            for metric_name in self.standard_metric_names
        }
        self.train_start_time = time.time()
        self.epochs_elapsed = 0

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
        self.epochs_elapsed += 1

    def decode_prefixes(self, generated_ids: list[torch.Tensor]) -> list[list[str]]:
        return self.trained_model.decode_prefixes(generated_ids)

    def decode_tokens_and_get_logits(
        self, batch: torch.Tensor, max_length: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        self.trained_model_optimizer.zero_grad()
        batch = batch.to(self.device)
        torch.set_grad_enabled(True)
        generated_ids: list[torch.Tensor] = []
        token_probabilities: list[torch.Tensor] = []
        reference_probabilities: list = []
        for seq in batch:
            new_ids, scores = self.trained_model.generate_with_greedy_decoding(
                seq.unsqueeze(0), max_length
            )
            generated_ids.append(new_ids)
            new_token_probabilites = torch.stack(
                    [
                        torch.softmax(scores[i][0], dim=0)[new_ids[i + 1]]
                        for i in range(len(scores))
                    ],
                )
            new_reference_probabilites = torch.exp(
                    self.reference_model.bert.compute_transition_scores(
                        new_ids.unsqueeze(0), scores, normalize_logits=True
                    ).squeeze(0)
                )
            # We don't want to include the generation logit of the EOS token.
            if new_ids[-1] == self.trained_model.bert.generation_config.eos_token_id:
                new_token_probabilites = new_token_probabilites[:-1]
                new_reference_probabilites = new_reference_probabilites[:-1]
                generated_ids = generated_ids[:-1]
            token_probabilities.append(
                new_token_probabilites
            )
            reference_probabilities.append(
                new_reference_probabilites
            )
        return generated_ids, token_probabilities, reference_probabilities

    def get_value_function_scores(
        self, batch_prefixes: list[list[str]], original_seqs: list[str]
    ) -> list[torch.Tensor]:
        self.value_optimizer.zero_grad()
        return [
            torch.concat(
                [self.value_model.get_value(prefix, original_seq) for prefix in sample_prefixes]
            )
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
            discount_exponents = torch.pow(GAMMA * LAMBDA, torch.arange(max_generated_length)).to(self.device)
            # Following the notation and equations from Schulman et al.
            batch_size = len(rewards)
            gammas = [
                rewards[i][:-1] + GAMMA * values[i][1:] - values[i][:-1] for i in range(batch_size)
            ]
            advantages = [
                torch.stack(
                    [
                        torch.sum(
                            gammas[batch_index][t:] * discount_exponents[: len(gammas[batch_index][t:])]
                        )
                        for t in range(len(rewards[batch_index]))
                    ],
                    dim=0
                )
                for batch_index in range(batch_size)
            ]
        ratios = [token_probs[i] / reference_probs[i] for i in range(batch_size)]
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

    def policy_loss_step(self, policy_loss: torch.Tensor) -> None:
        policy_loss.backward()
        self.trained_model_optimizer.step()

    def get_value_loss(
        self, rewards: list[torch.Tensor], values: list[torch.Tensor]
    ) -> torch.Tensor:
        value_loss = torch.mean(
            torch.concat([values[i] - rewards[i][-1].detach() for i in range(len(values))])
        )
        return value_loss

    def value_loss_step(self, value_loss: torch.Tensor) -> None:
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

    def add_epoch_metrics(
        self,
        epoch_metrics: dict[str, float],
        mode: str,
    ) -> None:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        for metric_name in epoch_metrics.keys():
            if metric_name in self.standard_metric_names:
                self.all_data[metric_name][mode].append(epoch_metrics[metric_name])
            else:
                warnings.warn(f"Metric '{metric_name}' is not standard, ignoring.")
        for metric_name in self.standard_metric_names:
            if metric_name not in epoch_metrics:
                warnings.warn(f"Metric '{metric_name}' was not passed for this iteration!")
        print(
            f"Epoch {self.epochs_elapsed},"
            f" this epoch's {mode} stats, as follows:\n"
            f"{epoch_metrics}\n"
        )
        if mode == EVAL:
            self.epochs_elapsed += 1

    def add_nonstandard_epoch_metrics(self, epoch_metrics: dict[str, float], mode: str) -> None:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        for metric_name in epoch_metrics.keys():
            if metric_name not in self.all_data:
                self.all_data[metric_name] = {
                    TRAIN: [],
                    EVAL: []
                }
            self.all_data[metric_name][mode].append(epoch_metrics[metric_name])

    def iteration(
            self,
            dataloader: DataLoader,
            device: str,
            mode: str,
    ) -> float:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"

        self.initialize_iteration(mode)
        if mode == TRAIN:
            self.freeze_reference_model()

        epoch_policy_losses: list[float] = []
        epoch_value_losses: list[float] = []
        epoch_rewards: list[float] = []
        nonstandard_metrics = ListDict()

        for batch in tqdm(
                dataloader, total=len(dataloader), desc=f"{mode} iteration", leave=False, position=1
        ):
            input_ids = batch["input_ids"].to(device)
            generated_ids, token_probs, reference_probs = self.decode_tokens_and_get_logits(
                input_ids, self.max_len
            )
            batch_prefixes = self.decode_prefixes(generated_ids)

            for i in range(len(batch_prefixes)):
                assert all_equal(
                    [len(token_probs[i]), len(reference_probs[i]), len(batch_prefixes[i])]
                )

            original_seqs = batch["original_seq"]

            with torch.no_grad():
                rewards, stats = self.rewards_and_metrics_function(batch, batch_prefixes, original_seqs)

            for stat_name in stats.keys():
                nonstandard_metrics.append(stat_name, stats[stat_name])

            values = self.get_value_function_scores(batch_prefixes, original_seqs)

            clipped_objectives = self.get_clipped_objectives(
                rewards, values, token_probs, reference_probs
            )

            policy_loss = self.get_policy_loss(clipped_objectives)
            value_loss = self.get_value_loss(rewards, values)

            if mode == TRAIN:
                self.policy_loss_step(policy_loss)
                self.value_loss_step(value_loss)

            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())

            final_rewards = [reward[-1].item() for reward in rewards]

            epoch_rewards.append(float(mean(final_rewards)))

        mean_final_reward = float(mean(epoch_rewards))
        self.add_epoch_metrics(
            {
                REWARD_METRIC: mean_final_reward,
                POLICY_LOSS_METRIC: float(mean(epoch_policy_losses)),
                VALUE_LOSS_METRIC: float(mean(epoch_value_losses)),
            },
            mode
        )
        self.add_nonstandard_epoch_metrics(
            {
                list_name: float(mean(nonstandard_metrics[list_name])) for list_name in nonstandard_metrics.lists.keys()
            },
            mode
        )
        return mean_final_reward

    def save_logs(self) -> None:
        logs_path = self.save_dir / "log.txt"
        with open(logs_path, "w") as f:
            f.write(json.dumps(self.all_data, indent=4))

    def save_summary(self, best_epoch_no: int) -> None:
        time_now = time.time()
        time_elapsed = time.gmtime(time_now - self.train_start_time)

        summary_path = self.save_dir / "summary.txt"
        best_epoch_stats = {
            key: self.all_data[key][EVAL][best_epoch_no] for key in self.all_data.keys()
        }
        summary = (
            f"Training time: {time.strftime('%H:%M:%S', time_elapsed)}"
            f" Number of epochs elapsed: {self.epochs_elapsed}, best stats (final rewards)"
            f" for epoch {best_epoch_no}, as follows: {best_epoch_stats}"
        )
        with open(summary_path, "w") as f:
            f.write(summary)

    def save_plots(self) -> None:
        plots_path = self.save_dir / "plots"
        plots_path.mkdir(parents=True, exist_ok=True)
        for variable in self.all_data.keys():
            for mode in MODES:
                title = f"{mode}_{variable}"
                plt.title(title)
                plt.plot(self.all_data[variable][mode])
                plt.xlabel("iteration")
                plt.savefig(plots_path / f"{title}.jpg")

    def save_trained_model(self) -> None:
        torch.save(self.trained_model.bert.state_dict(), self.save_dir / "ckpt.pt")



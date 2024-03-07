import copy
import warnings
from pathlib import Path
from typing import Callable

import torch
from numpy import mean
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.generation.base_trainer import EVAL, MODES, POLICY_LOSS_METRIC, TRAIN, Trainer
from src.generation.generative_bart import GenerativeBart
from src.generation.value_model import ValueModel
from src.utils import ListDict

# Discount factor, following the notation from the original PPO paper by Schulman et al.
GAMMA = 0.99

# Generalized Advantage Estimation factor
LAMBDA = 0.95

# Policy update clipping threshold
EPSILON = 0.2

# 2000 years and we're still using Greek if we want something to sound smart.


REWARD_METRIC = "reward"
VALUE_LOSS_METRIC = "value_loss"


class PPOTrainer(Trainer):
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
        super().__init__(
            standard_metric_names=[
                REWARD_METRIC,
                POLICY_LOSS_METRIC,
                VALUE_LOSS_METRIC,
            ],
            save_dir=save_dir,
            call_parameters_save_path=call_parameters_save_path,
            params_to_save=params_to_save,
            max_len=max_len,
        )
        self.trained_model = trained_model
        self.rewards_and_metrics_function = rewards_and_metrics_function
        self.reference_model = copy.deepcopy(self.trained_model)
        self.reference_model.eval()
        self.value_model = value_model
        self.trained_model_optimizer = trained_model_optimizer
        self.value_model_optimizer = value_model_optimizer
        self.device = device

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
            new_ids, new_token_probabilites = self.trained_model.generate(
                seq.unsqueeze(0),
                method="greedy",
                max_length=max_length,
            )
            new_ids = new_ids.to(self.device)
            generated_ids.append(new_ids)
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
        policy_loss.backward()
        if step:
            self.trained_model_optimizer.step()
            self.trained_model_optimizer.zero_grad()

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
        value_loss.backward(retain_graph=True)
        if step:
            self.value_model_optimizer.step()
            self.value_model_optimizer.zero_grad()

    def save_trained_models(self) -> None:
        torch.save(self.trained_model.bert.state_dict(), self.save_dir / "generator_ckpt.pt")
        torch.save(self.value_model.state_dict(), self.save_dir / "value_ckpt.pt")

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

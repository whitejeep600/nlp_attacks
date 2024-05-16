from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy import isclose, mean
from torch.nn.functional import logsigmoid
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import EVAL, GENERATIONS_EQUAL, MODES, POLICY_LOSS, TRAIN
from src.generation.base_trainer import Trainer
from src.generation.generative_bart import GenerativeBart
from src.utils import get_length_without_padding, sequence_logprob


class SampleGenerations:
    def __init__(
        self,
        prompt: str,
        prompt_tokens: torch.Tensor,
        generation_texts: list[str],
        generation_tokens: list[torch.Tensor],
        generation_probabilities: list[torch.Tensor],
        generation_reference_probabilities: list[torch.Tensor],
        rewards: list[float],
        generation_metrics: list[dict[str, float]],
    ):
        # Afterwards it is assumed that the generations for a given prompt (and all their
        # corresponding attributes) are ordered by increasing score (reward).
        ordering = np.argsort(rewards)
        self.prompt = prompt
        self.prompt_tokens = prompt_tokens
        self.rewards = [rewards[i] for i in ordering]
        self.generation_texts = [generation_texts[i] for i in ordering]
        self.generation_tokens = [generation_tokens[i] for i in ordering]
        self.generation_probabilities = [generation_probabilities[i] for i in ordering]
        self.generation_reference_probabilities = [
            generation_reference_probabilities[i] for i in ordering
        ]
        self.generation_metrics = [generation_metrics[i] for i in ordering]


class RewardCalculator:
    def __init__(self):
        pass

    def get_rewards_and_metrics(
        self, prompt: str, generations: list[str]
    ) -> tuple[list[float], list[dict[str, float]]]:
        raise NotImplementedError

    def get_metric_names(self) -> list[str]:
        raise NotImplementedError

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


class WarmupScheduler:
    def __init__(self, initial_lr: float, final_lr: float, n_steps: int):
        self.current_lr = initial_lr
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_n_steps = n_steps
        self.steps_performed = 0

    def get(self):
        result = self.current_lr
        if self.steps_performed < self.total_n_steps:
            self.steps_performed += 1
            self.current_lr = (
                self.initial_lr
                + (self.final_lr - self.initial_lr)
                * (self.steps_performed / self.total_n_steps) ** 2
            )
        elif self.steps_performed == self.total_n_steps:
            assert isclose(result, self.final_lr)
        return result


class DPOTrainer(Trainer):
    def __init__(
        self,
        trained_model: GenerativeBart,
        metric_calculator: RewardCalculator,
        trained_model_optimizer: Optimizer,
        beta: float,
        temperature: float,
        attacker_lr: float,
        max_len: int,
        reference_model: GenerativeBart,
        save_dir: Path,
        general_training_log_path: Path,
        params_to_save: dict,
        gradient_accumulation_batches: int = 4,
    ):
        super().__init__(
            standard_metric_names=[POLICY_LOSS, GENERATIONS_EQUAL],
            save_dir=save_dir,
            general_training_log_path=general_training_log_path,
            params_to_save=params_to_save,
            max_len=max_len,
        )
        # Beta - common notation for the term determining the influence of the penalty
        # for Kullback-Leibler divergence between the trained and reference policy.
        self.trained_model = trained_model
        self.metric_calculator = metric_calculator
        self.reference_model = reference_model
        self.reference_model.eval()
        self.trained_model_optimizer = trained_model_optimizer
        self.beta = beta
        self.temperature = temperature
        self.trained_model_lr_scheduler = WarmupScheduler(0, attacker_lr, 128)
        self.gradient_accumulation_batches = gradient_accumulation_batches
        self.temperatures: list[float] = []

    def train(self) -> None:
        self.trained_model.train()
        self.reference_model.eval()  # sic!
        self.metric_calculator.train()

    def eval(self) -> None:
        self.trained_model.eval()
        self.reference_model.eval()
        self.metric_calculator.eval()

    def batch_generate(
        self,
        batch_inputs: torch.Tensor,
        method: str = "sampling",
        max_length: int | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if method not in ["sampling", "greedy"]:
            raise ValueError(f"Invalid generation method: {method}")
        if max_length is None:
            max_length = self.trained_model.max_length
        batch_inputs = batch_inputs.to(self.trained_model.device)
        all_decoded_ids = (
            torch.Tensor([[self.trained_model.start_token] for _ in range(len(batch_inputs))])
            .int()
            .to(self.trained_model.device)
        )
        all_probabilities: list[list[torch.Tensor]] = [[] for _ in batch_inputs]
        all_reference_probabilities: list[list[torch.Tensor]] = [[] for _ in batch_inputs]
        for _ in range(max_length - 1):
            new_logits = self.trained_model.bert(
                input_ids=batch_inputs,
                decoder_input_ids=all_decoded_ids,
            ).logits[:, -1, :]
            with torch.no_grad():
                new_reference_logits = (
                    self.reference_model.bert(
                        input_ids=batch_inputs.to(self.reference_model.device),
                        decoder_input_ids=all_decoded_ids.to(self.reference_model.device),
                    )
                    .logits[:, -1, :]
                    .to(self.trained_model.device)
                )
            new_probabilities = torch.softmax(new_logits / self.temperature, dim=-1)
            new_reference_probabilities = torch.softmax(
                new_reference_logits / self.temperature, dim=-1
            )
            if method == "greedy":
                next_ids = torch.argmax(new_logits, dim=-1)[:, None]
            else:  # method == "sampling":
                next_ids = torch.multinomial(new_probabilities, 1, replacement=True)
            for i in range(len(new_probabilities)):
                all_probabilities[i].append(new_probabilities[i][next_ids[i]].reshape(1))
                all_reference_probabilities[i].append(
                    new_reference_probabilities[i][next_ids[i]].reshape(1)
                )
            all_decoded_ids = torch.cat((all_decoded_ids, next_ids), dim=-1)
            if (next_ids == self.trained_model.stop_token).all():
                break
        decoded_id_single_tensors = [decoded_tensor for decoded_tensor in all_decoded_ids]
        probability_single_tensors = [
            torch.cat(probability_list, dim=-1) for probability_list in all_probabilities
        ]
        reference_probability_single_tensors = [
            torch.cat(probability_list, dim=-1) for probability_list in all_reference_probabilities
        ]
        real_lengths = [
            get_length_without_padding(decoded_id_single_tensor, self.trained_model.stop_token)
            for decoded_id_single_tensor in decoded_id_single_tensors
        ]
        truncated_decoded_id_single_tensors = [
            ids[:length] for (ids, length) in zip(decoded_id_single_tensors, real_lengths)
        ]
        truncated_probs = [
            probability_single_tensor[:length]
            for (probability_single_tensor, length) in zip(probability_single_tensors, real_lengths)
        ]
        truncated_reference_probs = [
            probability_single_tensor[:length]
            for (probability_single_tensor, length) in zip(
                reference_probability_single_tensors, real_lengths
            )
        ]
        return truncated_decoded_id_single_tensors, truncated_probs, truncated_reference_probs

    def sample_two_generations_per_sample(
        self,
        batch_input_ids: torch.Tensor,
        batch_original_seqs: list[str],
    ) -> list[SampleGenerations]:
        """

        :param batch_input_ids: shape (batch_size, seq_len)
        :param batch_original_seqs: list of strings representing the original batch sequences
        :return: list of SampleGenerations, one SampleGenerations object per sample. Note that
            each SampleGeneration itself contains multiple generations for a single samples

        """
        batch_input_ids = batch_input_ids.to(self.trained_model.device)
        all_generations: list[SampleGenerations] = []
        generation_ids, probs, reference_probs = self.batch_generate(
            torch.repeat_interleave(batch_input_ids, 2, dim=0),
            max_length=self.max_len,
            method="sampling",
        )
        for batch_index in range(len(generation_ids) // 2):
            ids_0 = generation_ids[batch_index * 2]
            ids_1 = generation_ids[batch_index * 2 + 1]
            probs_0 = probs[batch_index * 2]
            probs_1 = probs[batch_index * 2 + 1]
            reference_probs_0 = reference_probs[batch_index * 2]
            reference_probs_1 = reference_probs[batch_index * 2 + 1]
            sample_original_seq = batch_original_seqs[batch_index]
            text_0, text_1 = self.trained_model.decode([ids_0, ids_1])
            rewards, metrics = self.metric_calculator.get_rewards_and_metrics(
                sample_original_seq, [text_0, text_1]
            )
            generations = SampleGenerations(
                prompt=sample_original_seq,
                prompt_tokens=batch_input_ids[batch_index],
                generation_texts=[text_0, text_1],
                generation_tokens=[ids_0, ids_1],
                generation_probabilities=[probs_0, probs_1],
                generation_reference_probabilities=[reference_probs_0, reference_probs_1],
                rewards=rewards,
                generation_metrics=metrics,
            )
            all_generations.append(generations)
        return all_generations

    def get_batch_policy_loss(self, batch_generations: list[SampleGenerations]) -> torch.Tensor:
        better_seq_logprobs = torch.cat(
            [
                sequence_logprob(generation.generation_probabilities[1])
                for generation in batch_generations
            ],
            dim=-1,
        )
        better_seq_reference_logprobs = torch.cat(
            [
                sequence_logprob(generation.generation_reference_probabilities[1])
                for generation in batch_generations
            ],
            dim=-1,
        )

        better_seq_logratios = better_seq_logprobs - better_seq_reference_logprobs

        worse_seq_logprobs = torch.cat(
            [
                sequence_logprob(generation.generation_probabilities[0])
                for generation in batch_generations
            ],
            dim=-1,
        )
        worse_seq_reference_logprobs = torch.cat(
            [
                sequence_logprob(generation.generation_reference_probabilities[0])
                for generation in batch_generations
            ],
            dim=-1,
        )

        worse_seq_logratios = worse_seq_logprobs - worse_seq_reference_logprobs

        return -1 * logsigmoid(self.beta * (better_seq_logratios - worse_seq_logratios)).mean()

    def update_learning_rate(self) -> None:
        new_lr = self.trained_model_lr_scheduler.get()
        for g in self.trained_model_optimizer.param_groups:
            g["lr"] = new_lr

    def adjust_temperature(self, generations_equal_ratio: float) -> None:
        if generations_equal_ratio < 0.1:
            new_temperature = 1 + (self.temperature - 1) * 0.9
        else:
            new_temperature = min(self.temperature + 0.1, 2.5)
        self.temperature = new_temperature

    def policy_loss_step(self, policy_loss: torch.Tensor, batch_no: int):
        self.update_learning_rate()
        policy_loss.backward()
        if batch_no % self.gradient_accumulation_batches == self.gradient_accumulation_batches - 1:
            self.trained_model_optimizer.step()
            self.trained_model_optimizer.zero_grad()

    def save_trained_model(self, filename: str = "generator_ckpt.pt") -> None:
        torch.save(self.trained_model.bert.state_dict(), self.save_dir / filename)

    def save_epoch_generations_and_metrics(
        self, all_batch_generations: list[list[SampleGenerations]]
    ) -> None:
        flattened_generations = [
            sample_generation
            for batch_generations in all_batch_generations
            for sample_generation in batch_generations
        ]
        all_generation_info_dicts: list[dict[str, Any]] = []
        for sample_generations in flattened_generations:
            for i in range(len(sample_generations.generation_texts)):
                generation_info = {
                    "prompt": sample_generations.prompt,
                    "text": sample_generations.generation_texts[i],
                    "reward": sample_generations.rewards[i],
                }
                generation_info.update(sample_generations.generation_metrics[i])
                all_generation_info_dicts.append(generation_info)
        df_to_save = pd.DataFrame.from_records(all_generation_info_dicts)
        generated_sentences_path = self.save_dir / "generated_sentences"
        generated_sentences_path.mkdir(parents=True, exist_ok=True)
        current_save_path = generated_sentences_path / f"epoch_{self.epochs_elapsed}.csv"
        df_to_save.to_csv(current_save_path, index=False)

    def plot_temperatures(self) -> None:
        plots_path = self.save_dir / "plots"

        plots_path.mkdir(parents=True, exist_ok=True)
        xs = self.temperatures
        title = "epoch_temperature"
        plt.title(title)
        plt.plot(xs, linewidth=0.5)
        plt.xlabel("iteration")
        plt.savefig(plots_path / f"{title}.jpg", dpi=256)
        plt.clf()

    def iteration(
        self, dataloader: DataLoader, mode: str, n_max_batches: int | None = None
    ) -> float:
        assert mode in MODES, f"unsupported mode, expected one of {MODES}"
        self.initialize_iteration(mode)

        epoch_policy_losses: list[float] = []
        all_epoch_batch_generations: list[list[SampleGenerations]] = []

        for batch_no, batch in tqdm(
            enumerate(dataloader),
            total=n_max_batches if n_max_batches is not None else len(dataloader),
            desc=f"{mode} iteration",
            leave=False,
            position=1,
        ):
            input_ids = batch["input_ids"].to(self.trained_model.device)
            original_seqs = batch["original_seq"]
            generations = self.sample_two_generations_per_sample(input_ids, original_seqs)
            all_epoch_batch_generations.append(generations)

            policy_loss = self.get_batch_policy_loss(generations)
            if mode == TRAIN:
                self.policy_loss_step(policy_loss, batch_no)

            epoch_policy_losses.append(policy_loss.item())

            if batch_no == n_max_batches:
                break

        batch_nonstandard_metrics = {
            metric_name: [
                float(
                    mean(
                        [
                            single_generation_metrics[metric_name]
                            for sample_generations in batch_generations
                            for single_generation_metrics in sample_generations.generation_metrics
                        ]
                    )
                )
                for batch_generations in all_epoch_batch_generations
            ]
            for metric_name in self.metric_calculator.get_metric_names()
        }

        mean_generation_score = float(
            mean(
                [
                    single_generation_score
                    for batch_generations in all_epoch_batch_generations
                    for sample_generations in batch_generations
                    for single_generation_score in sample_generations.rewards
                ]
            )
        )
        generations_equal_ratio = [
            float(
                mean(
                    [
                        int(
                            sample_generations.generation_texts[0]
                            == sample_generations.generation_texts[1]
                        )
                        for sample_generations in batch_generations
                    ]
                )
            )
            for batch_generations in all_epoch_batch_generations
        ]
        self.add_epoch_metrics(
            {POLICY_LOSS: epoch_policy_losses, GENERATIONS_EQUAL: generations_equal_ratio},
            mode,
        )
        self.add_nonstandard_epoch_metrics(
            batch_nonstandard_metrics,
            mode,
        )
        if mode == EVAL:
            self.save_epoch_generations_and_metrics(all_epoch_batch_generations)
            self.temperatures.append(self.temperature)
            self.adjust_temperature(float(mean(generations_equal_ratio)))

        return mean_generation_score

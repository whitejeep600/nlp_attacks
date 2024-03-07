import copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from numpy import mean
from torch.nn.functional import logsigmoid
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.generation.base_trainer import EVAL, MODES, POLICY_LOSS_METRIC, TRAIN, Trainer
from src.generation.generative_bart import GenerativeBart
from src.utils import ListDict


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


class DPOTrainer(Trainer):
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
        super().__init__(
            standard_metric_names=[POLICY_LOSS_METRIC],
            save_dir=save_dir,
            call_parameters_save_path=call_parameters_save_path,
            params_to_save=params_to_save,
            max_len=max_len,
            device=device,
        )
        # Beta - common notation for the term determining the influence of the penalty
        # for Kullback-Leibler divergence between the trained and reference policy.
        self.trained_model = trained_model
        self.rewards_and_metrics_function = rewards_and_metrics_function
        self.reference_model = copy.deepcopy(self.trained_model)
        self.reference_model.eval()
        self.trained_model_optimizer = trained_model_optimizer
        self.beta = beta

    def train(self) -> None:
        self.trained_model.train()
        self.reference_model.eval()

    def eval(self) -> None:
        self.trained_model.eval()
        self.reference_model.eval()

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
            generation_ids, probs = self.trained_model.batch_generate(
                torch.stack((sample_input_ids, sample_input_ids)),
                max_length=self.max_len,
                method="sampling",
            )
            ids_0, ids_1 = generation_ids
            probs_0, probs_1 = probs
            with torch.no_grad():
                reference_probs_0 = self.reference_model.get_reference_probabilities(
                    sample_input_ids,
                    ids_0,
                )
                reference_probs_1 = self.reference_model.get_reference_probabilities(
                    sample_input_ids,
                    ids_1,
                )
            text_0, text_1 = self.trained_model.decode([ids_0, ids_1])
            score_0, metrics_0 = self.rewards_and_metrics_function(sample_original_seq, text_0)
            score_1, metrics_1 = self.rewards_and_metrics_function(sample_original_seq, text_1)
            generations = SampleGenerations(
                prompt=sample_original_seq,
                prompt_tokens=sample_input_ids,
                generation_texts=[text_0, text_1],
                generation_tokens=[ids_0, ids_1],
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

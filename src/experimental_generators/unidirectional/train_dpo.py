from __future__ import annotations

from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classifiers.entailment_evaluator import EntailmentEvaluator
from src.classifiers.sentiment_classifier import SentimentClassifier
from src.constants import (
    ENTAILMENT,
    EVAL,
    GAN_ACCURACY,
    GAN_GENERATED_LABEL,
    NATURALNESS,
    NEGATIVE,
    POSITIVE,
    REWARD,
    TARGET_LABEL_PROB,
    TRAIN,
)
from src.datasets.sst2_dataset import SST2Dataset
from src.gan.gan_discriminator import GANDiscriminator
from src.generation.dpo_trainer import DPOTrainer, RewardCalculator
from src.generation.generative_bart import GenerativeBart
from src.utils import assign_gpu_devices, get_next_run_subdir_name, harmonic_mean, round_list

GAN_THRESHOLD = 0.6


def get_base(n: float) -> float:
    if n < GAN_THRESHOLD:
        return 0.48 * (n / GAN_THRESHOLD)
    elif n < 1:
        return 0.48 + 0.02 * (n - GAN_THRESHOLD) / (1 - GAN_THRESHOLD)
    else:
        return 0.5


def get_limit(n: float) -> float:
    if n < GAN_THRESHOLD:
        return 0.1 * (n / GAN_THRESHOLD)
    elif n < 1:
        return 0.1 + 0.4 * (n - GAN_THRESHOLD) / (1 - GAN_THRESHOLD)
    else:
        return 0.5


class UnidirectionalMetricCalculator(RewardCalculator):
    def __init__(
        self,
        entailment_classifier: EntailmentEvaluator,
        sentiment_classifier: SentimentClassifier,
        gan_discriminator: GANDiscriminator,
        gan_weight_decay: float,
        target_label: int,
    ):
        super().__init__()
        self.entailment_classifier = entailment_classifier
        self.sentiment_classifier = sentiment_classifier
        self.gan_discriminator = gan_discriminator
        self.precomputed_prompt_target_label_probs: dict[str, float] = {}
        self.mode = TRAIN
        self.gan_weight_decay = gan_weight_decay
        self.target_label = target_label

        # Weighted loss to balance the fact that in training batches there are two generations
        # for one sampled sentence. This will stop working if GAN_GENERATED_LABEL is not 1.
        self.gan_loss = nn.CrossEntropyLoss(
            reduction="mean", weight=torch.FloatTensor([2, 1]).to(gan_discriminator.device)
        )

    @classmethod
    def calculate_rewards(
        cls,
        entailment_scores: list[float],
        target_label_prob_gains: list[float],
        gan_naturalness_scores: list[float],
    ) -> list[float]:
        rewards: list[float] = []
        for i in range(len(gan_naturalness_scores)):
            gan_naturalness_score = gan_naturalness_scores[i]
            entailment_score = entailment_scores[i] + 0.01
            target_label_prob_gain = 0.5 + target_label_prob_gains[i] / 2  # in [0, 1]
            base = get_base(gan_naturalness_score)
            limit = get_limit(gan_naturalness_score)
            from_other_goals = harmonic_mean([entailment_score, target_label_prob_gain])
            reward = round(base + limit * from_other_goals, 3)
            rewards.append(reward)
        return rewards

    # This is for evaluation purposes only and does not influence the rewards.
    def get_prompt_original_target_label_prob(self, prompt: str) -> float:
        if prompt not in self.precomputed_prompt_target_label_probs:
            original_target_label_prob = self.sentiment_classifier.evaluate_texts(
                [prompt], return_probs=True
            )[0][self.target_label].item()
            self.precomputed_prompt_target_label_probs[prompt] = round(
                original_target_label_prob, 2
            )
        return self.precomputed_prompt_target_label_probs[prompt]

    @staticmethod
    def get_sentence_length_difference_penalties(
        prompt: str, generations: list[str]
    ) -> list[float]:
        prompt_length = len(prompt.split())
        generation_lengths = [len(g.split()) for g in generations]
        return [
            (min(g_length, prompt_length) / max(g_length, prompt_length)) ** 2
            for g_length in generation_lengths
        ]

    def get_entailment(self, prompt: str, generations: list[str]) -> list[float]:
        model_evaluated_entailment_scores = self.entailment_classifier.evaluate_text_pairs(
            [(prompt, generation) for generation in generations], return_probs=True
        )[:, self.entailment_classifier.entailment_code].tolist()
        length_difference_penalties = self.get_sentence_length_difference_penalties(
            prompt, generations
        )
        final_scores = np.multiply(
            model_evaluated_entailment_scores, length_difference_penalties
        ).tolist()
        return round_list(final_scores)

    def get_target_label_prob(self, generations: list[str]) -> list[float]:
        target_label_probs = [
            float(score.item())
            for score in self.sentiment_classifier.evaluate_texts(generations, return_probs=True)[
                :, self.target_label
            ]
        ]
        return round_list(target_label_probs)

    def get_gan_naturalness(self, prompt: str, generations: list[str]) -> tuple[list[float], float]:
        all_sentences = generations + [prompt]
        all_labels = torch.LongTensor(
            [GAN_GENERATED_LABEL for _ in generations] + [1 - GAN_GENERATED_LABEL]
        ).to(self.gan_discriminator.device)
        batch = self.gan_discriminator.prepare_batch(all_sentences)
        gan_logits = self.gan_discriminator.forward(batch)
        discriminator_accuracy = float(
            (torch.argmax(gan_logits, dim=1) == all_labels).float().mean()
        )

        # Multiplying by 2 because these scores will be close to 0.5 for a good generator and
        # discriminator - discriminator's accuracy will be close to random guessing. However,
        # scores around 0.5 will have an unduly large influence on the reward expressed
        # as a harmonic mean of 3 goals, if the other two are - and this is the goal - close to 1.
        # Also not taking the last one, because the last element of the batch was the prompt.

        gan_fooling_factors = (
            2 * torch.softmax(gan_logits, dim=1)[:, 1 - GAN_GENERATED_LABEL][:-1]
        ).tolist()

        if self.mode == TRAIN:
            gan_weight_norm_sum = self.gan_discriminator.weight_norm_sum()
            loss = (
                self.gan_loss(gan_logits, all_labels) + self.gan_weight_decay * gan_weight_norm_sum
            )
            self.gan_discriminator.optimizer.zero_grad()
            loss.backward()
            self.gan_discriminator.optimizer.step()

        return round_list(gan_fooling_factors), round(discriminator_accuracy, 3)

    def get_rewards_and_metrics(
        self, prompt: str, generations: list[str]
    ) -> tuple[list[float], list[dict[str, float]]]:
        with ThreadPoolExecutor(max_workers=4) as executor:
            entailment_calculation = executor.submit(
                partial(self.get_entailment, prompt=prompt, generations=generations)
            )
            target_label_prob_calculation = executor.submit(
                partial(self.get_target_label_prob, generations=generations)
            )
            gan_naturalness_calculation = executor.submit(
                partial(self.get_gan_naturalness, prompt=prompt, generations=generations)
            )
            prompt_target_label_prob_calculation = executor.submit(
                partial(self.get_prompt_original_target_label_prob, prompt=prompt)
            )

        prompt_target_label_prob = prompt_target_label_prob_calculation.result()
        target_label_probs = target_label_prob_calculation.result()
        target_label_prob_gains = [
            target_label_prob - prompt_target_label_prob for target_label_prob in target_label_probs
        ]
        entailment_scores = entailment_calculation.result()
        gan_naturalness_scores, discriminator_accuracy = gan_naturalness_calculation.result()

        rewards = self.calculate_rewards(
            entailment_scores, target_label_prob_gains, gan_naturalness_scores
        )

        prompts_equal_generation = [
            int(generation.lower() == prompt.lower()) for generation in generations
        ]

        stats = [
            {
                ENTAILMENT: entailment_score,
                TARGET_LABEL_PROB: target_label_prob,
                NATURALNESS: gan_naturalness_score,
                "prompt_equals_generation": prompt_equals_generation,
                REWARD: reward,
                GAN_ACCURACY: discriminator_accuracy,
                "prompt_target_label_prob": prompt_target_label_prob,
                "target_label_prob_gain": round(target_label_prob - prompt_target_label_prob, 2),
            }
            for (
                entailment_score,
                target_label_prob,
                gan_naturalness_score,
                prompt_equals_generation,
                reward,
            ) in zip(
                entailment_scores,
                target_label_probs,
                gan_naturalness_scores,
                prompts_equal_generation,
                rewards,
            )
        ]
        return rewards, stats

    def get_metric_names(self) -> list[str]:
        return [
            ENTAILMENT,
            TARGET_LABEL_PROB,
            NATURALNESS,
            GAN_ACCURACY,
            REWARD,
            "prompt_equals_generation",
            "target_label_prob_gain",
            "prompt_target_label_prob",
        ]

    def train(self) -> None:
        self.gan_discriminator.train()
        self.mode = TRAIN

    def eval(self) -> None:
        self.gan_discriminator.eval()
        self.mode = EVAL


def main(
    source_model_name: str,
    train_split_path: Path,
    eval_split_path: Path,
    target_label_name: str,
    max_len: int,
    batch_size: int,
    n_epochs: int,
    attacker_lr: float,
    gan_lr: float,
    beta: float,
    temperature,
    gan_weight_decay,
    save_dir: Path,
    general_training_log_path: Path,
    params_to_save: dict,
    n_max_train_samples: int | None = None,
    source_model_weights_path: Path | None = None,
    gan_discriminator_weights_path: Path | None = None,
):
    if target_label_name == "positive":
        target_label = POSITIVE
    elif target_label_name == "negative":
        target_label = NEGATIVE
    else:
        raise ValueError(f"Unexpected target label {target_label_name}")
    generator_device, reference_model_device, evaluator_models_device = assign_gpu_devices()

    trained_model = GenerativeBart(
        source_model_name, max_len, generator_device, source_model_weights_path
    )
    reference_model = GenerativeBart(
        source_model_name, max_len, reference_model_device, source_model_weights_path
    )

    entailment_classifier = EntailmentEvaluator(evaluator_models_device)
    sentiment_classifier = SentimentClassifier(evaluator_models_device, raw_name="cnn-sst2")
    gan_discriminator = GANDiscriminator(
        evaluator_models_device, max_len, gan_lr, gan_discriminator_weights_path
    )

    opposite_label = 1-target_label
    train_dataset = SST2Dataset(
        train_split_path, trained_model.tokenizer, max_len, min_length=8, filter_label=opposite_label
    )
    eval_dataset = SST2Dataset(
        eval_split_path, trained_model.tokenizer, max_len, min_length=8, filter_label=opposite_label
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    if n_max_train_samples is not None:
        n_max_train_batches = n_max_train_samples // batch_size
    else:
        n_max_train_batches = None

    run_subdir_name = save_dir / get_next_run_subdir_name(save_dir)
    params_to_save.update({"run_subdir_name": str(run_subdir_name)})
    run_subdir_name.mkdir(parents=True, exist_ok=True)
    general_training_log_path.parent.mkdir(parents=True, exist_ok=True)

    unidirectional_optimizer = AdamW(trained_model.parameters(), lr=0)

    metric_calculator = UnidirectionalMetricCalculator(
        entailment_classifier=entailment_classifier,
        sentiment_classifier=sentiment_classifier,
        gan_discriminator=gan_discriminator,
        gan_weight_decay=gan_weight_decay,
        target_label=target_label,
    )

    dpo_trainer = DPOTrainer(
        trained_model=trained_model,
        metric_calculator=metric_calculator,
        trained_model_optimizer=unidirectional_optimizer,
        beta=beta,
        temperature=temperature,
        attacker_lr=attacker_lr,
        max_len=max_len,
        reference_model=reference_model,
        save_dir=run_subdir_name,
        general_training_log_path=general_training_log_path,
        params_to_save=params_to_save,
    )

    best_mean_final_rewards: float | None = None
    best_epoch = -1

    for i in tqdm(range(n_epochs), desc="training...", position=0):
        dpo_trainer.iteration(train_dataloader, TRAIN, n_max_batches=n_max_train_batches)
        with torch.no_grad():
            new_mean_final_rewards = dpo_trainer.iteration(eval_dataloader, EVAL)
        dpo_trainer.conclude_epoch()
        if best_mean_final_rewards is None or new_mean_final_rewards > best_mean_final_rewards:
            best_epoch = i
            best_mean_final_rewards = new_mean_final_rewards
            dpo_trainer.save_trained_model(filename="best_generator_ckpt.pt")
            torch.save(gan_discriminator.module.state_dict(), run_subdir_name / "best_gan_ckpt.pt")

    dpo_trainer.save_trained_model(filename="last_generator_ckpt.pt")
    torch.save(gan_discriminator.module.state_dict(), run_subdir_name / "last_gan_ckpt.pt")
    dpo_trainer.save_stuff(best_epoch)
    dpo_trainer.plot_temperatures()


if __name__ == "__main__":
    train_params = yaml.safe_load(open("params.yaml"))[
        "src.experimental_generators.unidirectional.train_dpo"
    ]

    source_model_name = train_params["source_model_name"]
    source_model_weights_path = (
        Path(train_params["source_model_weights_path"])
        if train_params["source_model_weights_path"] is not None
        else None
    )
    gan_discriminator_weights_path = (
        Path(train_params["gan_discriminator_weights_path"])
        if train_params["gan_discriminator_weights_path"] is not None
        else None
    )

    train_split_path = Path(train_params["train_split_path"])
    eval_split_path = Path(train_params["eval_split_path"])

    target_label_name = train_params["target_label_name"]
    max_len = int(train_params["max_len"])
    batch_size = int(train_params["batch_size"])
    n_epochs = int(train_params["n_epochs"])
    attacker_lr = float(train_params["attacker_lr"])
    gan_lr = float(train_params["gan_lr"])
    beta = float(train_params["beta"])
    temperature = float(train_params["temperature"])
    gan_weight_decay = float(train_params["gan_weight_decay"])
    n_max_train_samples: int | None = train_params["n_max_train_samples"]

    save_dir = Path(train_params["save_dir"])
    general_training_log_path = Path(train_params["general_training_log_path"])

    main(
        source_model_name,
        train_split_path,
        eval_split_path,
        target_label_name,
        max_len,
        batch_size,
        n_epochs,
        attacker_lr,
        gan_lr,
        beta,
        temperature,
        gan_weight_decay,
        save_dir,
        general_training_log_path,
        train_params,
        n_max_train_samples,
        source_model_weights_path,
        gan_discriminator_weights_path,
    )

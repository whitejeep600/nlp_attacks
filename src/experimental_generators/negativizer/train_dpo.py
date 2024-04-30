from __future__ import annotations

from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.modules.loss import _Loss
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
    NEGATIVITY,
    POSITIVE,
    REWARD,
    TRAIN,
)
from src.datasets.sst2_dataset import SST2Dataset
from src.gan.gan_discriminator import GANDiscriminator
from src.generation.dpo_trainer import DPOTrainer, RewardCalculator
from src.generation.generative_bart import GenerativeBart
from src.utils import assign_gpu_devices, get_next_run_subdir_name, round_list


class NegativizerMetricCalculator(RewardCalculator):
    def __init__(
        self,
        entailment_classifier: EntailmentEvaluator,
        sentiment_classifier: SentimentClassifier,
        gan_discriminator: GANDiscriminator,
        gan_loss: _Loss,
    ):
        super().__init__()
        self.entailment_classifier = entailment_classifier
        self.sentiment_classifier = sentiment_classifier
        self.gan_discriminator = gan_discriminator
        self.gan_loss = gan_loss
        self.precomputed_prompt_negativities: dict[str, float] = {}
        self.mode = TRAIN

    @classmethod
    def calculate_rewards(
        cls,
        entailment_scores: list[float],
        negativity_scores: list[float],
        gan_naturalness_scores: list[float],
    ) -> list[float]:
        allowed_naturalness_threshold = 0.8
        naturalness_penalties = np.minimum(
            np.array(gan_naturalness_scores), allowed_naturalness_threshold
        ) * (1 / allowed_naturalness_threshold)
        score_lists = [entailment_scores, negativity_scores]
        min_scores_per_list = [min(scores) for scores in score_lists]
        worse_score_list = score_lists[np.argmin(min_scores_per_list)]
        return round_list(np.multiply(worse_score_list, naturalness_penalties).tolist())
        # The logic here is that if the naturalness scores are high enough, we shouldn't pay too
        # much attention to them, and allow the model some margin. But if they are low, we should
        # try to improve them before anything else, because in that case the model is probably
        # generating gibberish and the other metrics are unreliable anyway.

    # This is for evaluation purposes only and does not influence the rewards.
    def get_prompt_original_negativity(self, prompt: str) -> float:
        if prompt not in self.precomputed_prompt_negativities:
            negativity = self.sentiment_classifier.evaluate_texts([prompt], return_probs=True)[0][
                NEGATIVE
            ].item()
            self.precomputed_prompt_negativities[prompt] = round(negativity, 2)
        return self.precomputed_prompt_negativities[prompt]

    @staticmethod
    def get_sentence_length_difference_penalties(
        prompt: str, generations: list[str]
    ) -> list[float]:
        prompt_length = len(prompt.split())
        generation_lengths = [len(g.split()) for g in generations]
        return [
            min(g_length, prompt_length) / max(g_length, prompt_length)
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

    def get_negativity(self, generations: list[str]) -> list[float]:
        negativity_scores = [
            float(score.item())
            for score in self.sentiment_classifier.evaluate_texts(generations, return_probs=True)[
                :, NEGATIVE
            ]
        ]
        return round_list(negativity_scores)

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
            loss = self.gan_loss(gan_logits, all_labels)
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
            negativity_calculation = executor.submit(
                partial(self.get_negativity, generations=generations)
            )
            gan_naturalness_calculation = executor.submit(
                partial(self.get_gan_naturalness, prompt=prompt, generations=generations)
            )
            prompt_negativity_calculation = executor.submit(
                partial(self.get_prompt_original_negativity, prompt=prompt)
            )

        entailment_scores = entailment_calculation.result()
        negativity_scores = negativity_calculation.result()
        gan_naturalness_scores, discriminator_accuracy = gan_naturalness_calculation.result()
        prompt_negativity = prompt_negativity_calculation.result()

        rewards = self.calculate_rewards(
            entailment_scores, negativity_scores, gan_naturalness_scores
        )

        prompts_equal_generation = [
            int(generation.lower() == prompt.lower()) for generation in generations
        ]
        generations_equal = int(generations[0] == generations[1])

        stats = [
            {
                ENTAILMENT: entailment_score,
                NEGATIVITY: negativity_score,
                NATURALNESS: gan_naturalness_score,
                "prompt_equals_generation": prompt_equals_generation,
                REWARD: reward,
                GAN_ACCURACY: discriminator_accuracy,
                "generations_equal": generations_equal,
                "prompt_negativity": prompt_negativity,
                "negativity_gain": round(negativity_score - prompt_negativity, 2),
            }
            for (
                entailment_score,
                negativity_score,
                gan_naturalness_score,
                prompt_equals_generation,
                reward,
            ) in zip(
                entailment_scores,
                negativity_scores,
                gan_naturalness_scores,
                prompts_equal_generation,
                rewards,
            )
        ]
        return rewards, stats

    def get_metric_names(self) -> list[str]:
        return [
            ENTAILMENT,
            NEGATIVITY,
            NATURALNESS,
            GAN_ACCURACY,
            REWARD,
            "prompt_equals_generation",
            "generations_equal",
            "negativity_gain",
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
    max_len: int,
    batch_size: int,
    n_epochs: int,
    attacker_lr: float,
    gan_lr: float,
    beta: float,
    temperature,
    save_dir: Path,
    general_training_log_path: Path,
    params_to_save: dict,
    n_max_train_samples: int | None = None,
    source_model_weights_path: Path | None = None,
    gan_discriminator_weights_path: Path | None = None,
):
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

    train_dataset = SST2Dataset(
        train_split_path, trained_model.tokenizer, max_len, min_length=8, filter_label=POSITIVE
    )
    eval_dataset = SST2Dataset(
        eval_split_path, trained_model.tokenizer, max_len, min_length=8, filter_label=POSITIVE
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

    negativizer_optimizer = AdamW(trained_model.parameters(), lr=0)

    # Weighted loss to balance the fact that in training batches there are two generations
    # for one sampled sentence. This will stop working if GAN_GENERATED_LABEL is not 1.
    gan_loss = nn.CrossEntropyLoss(
        reduction="mean", weight=torch.FloatTensor([2, 1]).to(gan_discriminator.device)
    )
    metric_calculator = NegativizerMetricCalculator(
        entailment_classifier=entailment_classifier,
        sentiment_classifier=sentiment_classifier,
        gan_discriminator=gan_discriminator,
        gan_loss=gan_loss,
    )

    dpo_trainer = DPOTrainer(
        trained_model=trained_model,
        metric_calculator=metric_calculator,
        trained_model_optimizer=negativizer_optimizer,
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


if __name__ == "__main__":
    train_params = yaml.safe_load(open("params.yaml"))[
        "src.experimental_generators.negativizer.train_dpo"
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

    max_len = int(train_params["max_len"])
    batch_size = int(train_params["batch_size"])
    n_epochs = int(train_params["n_epochs"])
    attacker_lr = float(train_params["attacker_lr"])
    gan_lr = float(train_params["gan_lr"])
    beta = float(train_params["beta"])
    temperature = float(train_params["temperature"])
    n_max_train_samples: int | None = train_params["n_max_train_samples"]

    save_dir = Path(train_params["save_dir"])
    general_training_log_path = Path(train_params["general_training_log_path"])

    main(
        source_model_name,
        train_split_path,
        eval_split_path,
        max_len,
        batch_size,
        n_epochs,
        attacker_lr,
        gan_lr,
        beta,
        temperature,
        save_dir,
        general_training_log_path,
        train_params,
        n_max_train_samples,
        source_model_weights_path,
        gan_discriminator_weights_path,
    )

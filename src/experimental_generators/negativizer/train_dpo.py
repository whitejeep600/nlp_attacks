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
from src.classifiers.grammaticality_evaluator import GrammaticalityEvaluator
from src.classifiers.sentiment_classifier import SentimentClassifier
from src.datasets.sst2_dataset import SST2Dataset
from src.gan.gan_discriminator import GANDiscriminator
from src.generation.dpo_trainer import EVAL, TRAIN, DPOTrainer
from src.generation.generative_bart import GenerativeBart
from src.utils import get_available_torch_devices

GAN_GENERATED_LABEL = 1


def harmonic_mean(numbers: list[float], weights: list[float] | None = None) -> float:
    numbers_array = np.array(numbers)
    if weights is None:
        weights_array = np.ones_like(numbers_array)
    else:
        weights_array = np.array(weights)
    return weights_array.sum() / (weights_array / numbers_array).sum()


def calculate_reward(
    entailment_score: float,
    negativity_score: float,
    gan_naturalness_score: float,
    grammaticality_score: float,
) -> float:
    gan_naturalness_score = min(1.0, gan_naturalness_score)
    return gan_naturalness_score * harmonic_mean(
        [entailment_score, negativity_score, grammaticality_score]
    )


def get_similarity_scores_and_nonstandard_metrics(
    prompt: str,
    generations: list[str],
    entailment_classifier: EntailmentEvaluator,
    sentiment_classifier: SentimentClassifier,
    grammaticality_evaluator: GrammaticalityEvaluator,
    gan_discriminator: GANDiscriminator,
    gan_loss: _Loss,
) -> tuple[list[float], list[dict[str, float]]]:

    def get_entailment():
        entailment_scores = entailment_classifier.evaluate_text_pairs(
            [(prompt, generation) for generation in generations], return_probs=True
        )[:, entailment_classifier.entailment_code].tolist()
        return entailment_scores

    def get_negativity():
        negativity_scores = [
            float(score.item())
            for score in sentiment_classifier.evaluate_texts(generations, return_probs=True)[:, 0]
        ]
        return negativity_scores

    def get_grammaticality():
        grammaticality_scores = [
            float(score.item())
            for score in grammaticality_evaluator.evaluate_texts(generations, return_probs=True)[
                :, 1
            ]
        ]
        return grammaticality_scores

    def get_gan_naturalness():
        all_sentences = generations + [prompt]
        all_labels = torch.LongTensor(
            [GAN_GENERATED_LABEL for _ in generations] + [1 - GAN_GENERATED_LABEL]
        ).to(gan_discriminator.device)
        batch = gan_discriminator.prepare_batch(all_sentences)
        gan_logits = gan_discriminator.forward(batch)
        discriminator_accuracy = float(
            (torch.argmax(gan_logits, dim=1) == all_labels).float().mean()
        )
        loss = gan_loss(gan_logits, all_labels)
        gan_discriminator.optimizer.zero_grad()
        loss.backward()
        gan_discriminator.optimizer.step()

        # Multiplying by 2 because these scores will be close to 0.5 for a good generator and
        # discriminator - discriminator's accuracy will be close to random guessing. However,
        # scores around 0.5 will have an unduly large influence on the reward expressed
        # as a harmonic mean of 3 goals, if the other two are - and this is the goal - close to 1.
        # Also not taking the last one, because the last element of the batch was the prompt.

        gan_fooling_factors = (
            2 * torch.softmax(gan_logits, dim=1)[:, 1 - GAN_GENERATED_LABEL][:-1]
        ).tolist()

        return gan_fooling_factors, discriminator_accuracy

    with ThreadPoolExecutor(max_workers=4) as executor:
        entailment_calculation = executor.submit(get_entailment)
        negativity_calculation = executor.submit(get_negativity)
        gan_naturalness_calculation = executor.submit(get_gan_naturalness)
        grammaticality_calculation = executor.submit(get_grammaticality)

    entailment_scores = entailment_calculation.result()
    negativity_scores = negativity_calculation.result()
    gan_naturalness_scores, discriminator_accuracy = gan_naturalness_calculation.result()
    grammaticality_scores = grammaticality_calculation.result()

    stats = [
        {
            "entailment_score": entailment_score,
            "negativity_score": negativity_score,
            "gan_naturalness_score": gan_naturalness_scores,
            "gan_discriminator_accuracy": discriminator_accuracy,
            "grammaticality_score": grammaticality_scores,
        }
        for (entailment_score, negativity_score, grammaticality_score, grammaticality_score) in zip(
            entailment_scores, negativity_scores, gan_naturalness_scores, grammaticality_scores
        )
    ]
    target_metrics = [
        calculate_reward(
            entailment_score, negativity_score, gan_naturalness_score, grammaticality_score
        )
        for (
            entailment_score,
            negativity_score,
            gan_naturalness_score,
            grammaticality_score,
        ) in zip(
            entailment_scores, negativity_scores, gan_naturalness_scores, grammaticality_scores
        )
    ]

    return target_metrics, stats


def train(
    echo: GenerativeBart,
    entailment_classifier: EntailmentEvaluator,
    sentiment_classifier: SentimentClassifier,
    grammaticality_evaluator: GrammaticalityEvaluator,
    gan_discriminator: GANDiscriminator,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    n_epochs: int,
    attacker_lr: float,
    beta: float,
    trained_model_device: str,
    reference_model_device: str,
    max_len: int,
    save_dir: Path,
    call_parameters_save_path: Path,
    params_to_save: dict,
    n_max_train_batches: int | None = None,
):
    run_no = 0
    while (save_dir / f"run_{run_no}").exists():
        run_no += 1
    save_dir = save_dir / f"run_{run_no}"
    params_to_save.update({"save_dir": str(save_dir)})
    save_dir.mkdir(parents=True, exist_ok=True)
    call_parameters_save_path.parent.mkdir(parents=True, exist_ok=True)

    echo_optimizer = AdamW(echo.parameters(), lr=attacker_lr)

    entailment_classifier.eval()

    # Weighted loss to balance the fact that in training batches there are two generations
    # for one sampled sentence. This will stop working if GAN_GENERATED_LABEL is not 1.
    gan_loss = nn.CrossEntropyLoss(
        reduction="mean", weight=torch.FloatTensor([2, 1]).to(gan_discriminator.device)
    )
    reward_function = partial(
        get_similarity_scores_and_nonstandard_metrics,
        entailment_classifier=entailment_classifier,
        sentiment_classifier=sentiment_classifier,
        grammaticality_evaluator=grammaticality_evaluator,
        gan_discriminator=gan_discriminator,
        gan_loss=gan_loss,
    )
    dpo_trainer = DPOTrainer(
        trained_model=echo,
        rewards_and_metrics_function=reward_function,
        trained_model_optimizer=echo_optimizer,
        beta=beta,
        max_len=max_len,
        trained_model_device=trained_model_device,
        reference_model_device=reference_model_device,
        save_dir=save_dir,
        call_parameters_save_path=call_parameters_save_path,
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
            dpo_trainer.save_trained_models()

    dpo_trainer.save_stuff(best_epoch)
    torch.save(gan_discriminator.module.state_dict(), save_dir / "gan_ckpt.pt")


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
    save_dir: Path,
    call_parameters_save_path: Path,
    params_to_save: dict,
    n_max_train_samples: int | None = None,
    source_model_weights_path: Path | None = None,
):
    devices = get_available_torch_devices()
    if len(devices) > 1:
        trained_model_device = devices[0]
        entailment_classifier_device = devices[1]
        reference_model_device = devices[1]
    else:
        trained_model_device = devices[0]
        entailment_classifier_device = devices[0]
        reference_model_device = devices[0]

    entailment_classifier = EntailmentEvaluator(entailment_classifier_device)

    sentiment_classifier = SentimentClassifier(entailment_classifier_device)
    grammaticality_evaluator = GrammaticalityEvaluator(entailment_classifier_device)
    gan_discriminator = GANDiscriminator(entailment_classifier_device, max_len, gan_lr)

    trained_model = GenerativeBart(source_model_name, max_len, trained_model_device)
    if source_model_weights_path is not None:
        trained_model.bert.load_state_dict(
            torch.load(source_model_weights_path, map_location=torch.device(trained_model_device))
        )

    train_dataset = SST2Dataset(
        train_split_path,
        trained_model.tokenizer,
        max_len,
    )
    eval_dataset = SST2Dataset(
        eval_split_path,
        trained_model.tokenizer,
        max_len,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    if n_max_train_samples is not None:
        n_max_train_batches = n_max_train_samples // batch_size
    else:
        n_max_train_batches = None

    train(
        trained_model,
        entailment_classifier,
        sentiment_classifier,
        grammaticality_evaluator,
        gan_discriminator,
        train_dataloader,
        eval_dataloader,
        n_epochs,
        attacker_lr,
        beta,
        trained_model_device,
        reference_model_device,
        max_len,
        save_dir,
        call_parameters_save_path,
        params_to_save,
        n_max_train_batches,
    )


if __name__ == "__main__":
    echo_params = yaml.safe_load(open("params.yaml"))[
        "src.experimental_generators.negativizer.train_dpo"
    ]

    source_model_name = echo_params["source_model_name"]
    source_model_weights_path = Path(echo_params["source_model_weights_path"])

    train_split_path = Path(echo_params["train_split_path"])
    eval_split_path = Path(echo_params["eval_split_path"])

    max_len = int(echo_params["max_len"])
    batch_size = int(echo_params["batch_size"])
    n_epochs = int(echo_params["n_epochs"])
    attacker_lr = float(echo_params["attacker_lr"])
    gan_lr = float(echo_params["gan_lr"])
    beta = float(echo_params["beta"])
    n_max_train_samples: int | None = echo_params["n_max_train_samples"]

    save_dir = Path(echo_params["save_dir"])
    call_parameters_save_path = Path(echo_params["call_parameters_save_path"])

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
        save_dir,
        call_parameters_save_path,
        echo_params,
        n_max_train_samples,
        source_model_weights_path,
    )

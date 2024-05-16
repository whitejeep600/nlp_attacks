from __future__ import annotations

# Metrics
ENTAILMENT = "entailment_score"
TARGET_LABEL_PROB = "target_label_prob"
NATURALNESS = "gan_naturalness_score"
GRAMMATICALITY = "grammaticality_score"
GAN_ACCURACY = "gan_discriminator_accuracy"
REWARD = "reward"
POLICY_LOSS = "policy_loss"
GENERATIONS_EQUAL = "generations_equal"

# Labels
GAN_GENERATED_LABEL = 1
POSITIVE = 1
NEGATIVE = 0

# Dataframe columns
LABEL = "label"
SENTENCE = "sentence"
ID = "idx"

# Other
TRAIN = "train"
EVAL = "eval"
MODES = [TRAIN, EVAL]

# Config constant(s)
PLOT_AVG_WINDOW_LENGTH = 16

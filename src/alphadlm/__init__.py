"""Sequence-level alpha loss for factorized masked diffusion denoisers."""

import os

# Train and tests run on CPU. Set before JAX is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from alphadlm.corrupt import alpha_bar, corrupt, sample_masked
from alphadlm.losses import (
    alpha_loss,
    ell_k,
    mean_token_cross_entropy,
    sequence_alpha_loss,
    token_cross_entropy,
    tokenwise_alpha_loss,
)
from alphadlm.optimum import (
    figure2_joint,
    lower_temperature_target,
    marginals,
    minimize_factorized_sequence,
    minimize_factorized_tokenwise,
    minimize_j,
)
from alphadlm.sampler import confidence_sample

__all__ = [
    "alpha_bar",
    "alpha_loss",
    "confidence_sample",
    "corrupt",
    "ell_k",
    "figure2_joint",
    "lower_temperature_target",
    "marginals",
    "mean_token_cross_entropy",
    "minimize_factorized_sequence",
    "minimize_factorized_tokenwise",
    "minimize_j",
    "sample_masked",
    "sequence_alpha_loss",
    "token_cross_entropy",
    "tokenwise_alpha_loss",
]

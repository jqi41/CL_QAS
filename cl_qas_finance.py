#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finance_clqas_revised.py

Financial CL-QAS experiment aligned with the revised manuscript.

Pipeline
--------
Financial price series
    -> causal technical features
    -> 32 timesteps x 8 features = 256-D input
    -> training-only robust normalization
    -> rank-r TT-SVD approximation
    -> normalized 256-dimensional amplitude state
    -> 8-qubit VQC

Methods
-------
1. Naive-VQC
   Fixed depth-4 ring-entangled VQC.

2. QAS-No-CL
   Transformer architecture policy with genuine bi-loop QAS.

3. CL-QAS
   Same QAS framework with:
       - EWC regularization
       - KL(pi_phi || pi_ref)
       - no replay buffer

Architecture search
-------------------
The policy selects:
    - depth: {2, 3, 4}
    - entangling pattern per layer:
        ring, linear, brick_even, brick_odd

All candidate circuits use RX-RY-RZ rotations on every qubit.

Reward
------
    R_m = c_m - lambda_hw * C(A_m)

where
    c_m = 0.70 * balanced_accuracy + 0.30 * F1.

If no valid financial CSV is provided, the script automatically
uses a synthetic regime-switching financial series for debugging.

Dependencies
------------
pip install torch numpy pandas scikit-learn
"""

import argparse
import copy
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REAL = torch.float32
COMPLEX = torch.complex64

LOOKBACK = 32
BASE_FEATURES = 8
INPUT_DIM = LOOKBACK * BASE_FEATURES  # 256

NUM_QUBITS = 8
NUM_CLASSES = 2

TT_MODES = (4, 16, 4)
TT_RANK = 3

DEPTH_CHOICES = (2, 3, 4)
ENT_PATTERNS = ("ring", "linear", "brick_even", "brick_odd")

SEARCH_STEPS = 15
CANDIDATES_PER_STEP = 4
SEARCH_INNER_EPOCHS = 10
FINAL_EPOCHS = 40

BATCH_SIZE = 64

VQC_LR = 3e-3
POLICY_LR = 1e-3
WEIGHT_DECAY = 1e-5

MU_EWC = 0.5
ETA_KL = 0.01
ENTROPY_COEF = 0.002

LAMBDA_HW = 0.01

DEFAULT_SEEDS = (11, 22, 33, 44, 55)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Financial indicators
# ============================================================

def _ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)

    y = np.zeros_like(x, dtype=np.float64)
    y[0] = x[0]

    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1.0 - alpha) * y[t - 1]

    return y


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))

    for t in range(len(x)):
        start = max(0, t - window + 1)
        length = t - start + 1
        out[t] = (cumsum[t + 1] - cumsum[start]) / length

    return out


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)

    for t in range(len(x)):
        start = max(0, t - window + 1)
        segment = x[start:t + 1]
        out[t] = np.std(segment)

    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])

    up = np.maximum(delta, 0.0)
    down = np.maximum(-delta, 0.0)

    ema_up = _ema(up, period)
    ema_down = _ema(down, period)

    rs = ema_up / np.maximum(ema_down, 1e-12)

    return 100.0 - 100.0 / (1.0 + rs)


def _macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)

    macd = ema_fast - ema_slow
    signal_line = _ema(macd, signal)
    hist = macd - signal_line

    return macd, signal_line, hist


def base_features(
    close: np.ndarray,
    lookback: int = LOOKBACK,
) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)

    ret = np.zeros_like(close, dtype=np.float64)
    ret[1:] = (
        close[1:] - close[:-1]
    ) / np.maximum(close[:-1], 1e-8)

    r_mean = _rolling_mean(ret, lookback)
    r_std = _rolling_std(ret, lookback)

    rsi = _rsi(close, period=14) / 100.0

    _, _, macd_hist = _macd(close)

    ema_price = _ema(close, lookback)
    momentum = close / np.maximum(ema_price, 1e-8) - 1.0

    rolling_price_mean = _rolling_mean(close, lookback)
    rolling_price_std = _rolling_std(close, lookback)

    bb_z = (
        close - rolling_price_mean
    ) / np.maximum(rolling_price_std, 1e-8)

    short_vol = _rolling_std(ret, 5)

    features = np.stack(
        [
            ret,
            r_mean,
            r_std,
            rsi,
            macd_hist,
            momentum,
            bb_z,
            short_vol,
        ],
        axis=1,
    )

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


# ============================================================
# Future-return labels
# ============================================================

def future_direction_labels(
    close: np.ndarray,
    horizon: int = 1,
) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    close = np.asarray(close, dtype=np.float64)

    labels = np.full(
        len(close),
        -1,
        dtype=np.int64,
    )

    future_return = (
        close[horizon:] - close[:-horizon]
    ) / np.maximum(close[:-horizon], 1e-8)

    labels[:-horizon] = (
        future_return > 0.0
    ).astype(np.int64)

    return labels


# ============================================================
# 256-D temporal windows
# ============================================================

def construct_windows(
    features: np.ndarray,
    labels: np.ndarray,
    lookback: int = LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for t in range(lookback - 1, len(features)):
        if labels[t] < 0:
            continue

        window = features[
            t - lookback + 1:t + 1
        ].reshape(-1)

        if window.size != INPUT_DIM:
            raise RuntimeError(
                f"Expected {INPUT_DIM}-D input, got {window.size}."
            )

        X.append(window)
        y.append(labels[t])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
    )


# ============================================================
# Data loading
# ============================================================

def load_close_from_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)

    candidate_columns = (
        "Adj Close",
        "Adj_Close",
        "adj_close",
        "Close",
        "close",
    )

    for column in candidate_columns:
        if column not in df.columns:
            continue

        values = df[column].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]

        if len(values) > 300:
            return values

    raise RuntimeError(
        f"No usable Close/Adj Close column found in {path}."
    )


def make_synthetic_close(
    n: int = 8000,
    regimes: int = 8,
    seed: int = 777,
) -> np.ndarray:
    rng = np.random.RandomState(seed)

    prices = np.zeros(n, dtype=np.float64)

    price = 100.0
    segment_length = n // regimes
    index = 0

    for regime in range(regimes):
        drift = rng.uniform(-0.0004, 0.0006)
        volatility = rng.uniform(0.004, 0.018)
        phi = rng.uniform(-0.10, 0.55)

        length = (
            segment_length
            if regime < regimes - 1
            else n - index
        )

        innovations = rng.normal(
            0.0,
            volatility,
            size=length,
        )

        returns = np.zeros(length)

        for t in range(length):
            previous = returns[t - 1] if t > 0 else 0.0
            returns[t] = drift + phi * previous + innovations[t]

        for r in returns:
            price = max(
                1e-3,
                price * (1.0 + r),
            )

            prices[index] = price
            index += 1

    return prices


# ============================================================
# Sequential financial tasks
# ============================================================

def build_finance_tasks(
    close: np.ndarray,
    n_tasks: int = 8,
    lookback: int = LOOKBACK,
    horizon: int = 1,
    min_per_task: int = 400,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    features = base_features(
        close,
        lookback=lookback,
    )

    labels = future_direction_labels(
        close,
        horizon=horizon,
    )

    X, y = construct_windows(
        features,
        labels,
        lookback=lookback,
    )

    cuts = np.linspace(
        0,
        len(X),
        n_tasks + 1,
        dtype=int,
    )

    tasks = []

    for task_id in range(n_tasks):
        start = cuts[task_id]
        end = cuts[task_id + 1]

        Xi = X[start:end]
        yi = y[start:end]

        if len(Xi) < min_per_task:
            print(
                f"[task-build] Skip task {task_id + 1}: "
                f"only {len(Xi)} samples."
            )
            continue

        counts = np.bincount(
            yi,
            minlength=2,
        )

        print(
            f"[task-build] Task {task_id + 1}: "
            f"n={len(Xi)}, "
            f"class0={counts[0]}, "
            f"class1={counts[1]}"
        )

        tasks.append(
            (
                torch.tensor(Xi, dtype=REAL),
                torch.tensor(yi, dtype=torch.long),
            )
        )

    return tasks


# ============================================================
# Chronological train / validation / test split
# ============================================================

def time_split(
    X: torch.Tensor,
    y: torch.Tensor,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    n = len(X)

    n_train = max(
        int(train_frac * n),
        64,
    )

    n_val = max(
        int(val_frac * n),
        32,
    )

    if n_train + n_val >= n:
        raise RuntimeError(
            "Task is too small for chronological "
            "train/validation/test splitting."
        )

    i1 = n_train
    i2 = n_train + n_val

    return (
        X[:i1],
        y[:i1],
        X[i1:i2],
        y[i1:i2],
        X[i2:],
        y[i2:],
    )


# ============================================================
# Train-only robust normalization
# ============================================================

@dataclass
class RobustScalerState:
    median: torch.Tensor
    scale: torch.Tensor


def fit_robust_scaler(
    X_train: torch.Tensor,
) -> RobustScalerState:
    median = torch.median(
        X_train,
        dim=0,
    ).values

    absolute_deviation = torch.abs(
        X_train - median
    )

    mad = torch.median(
        absolute_deviation,
        dim=0,
    ).values

    scale = 1.4826 * mad + 1e-6

    return RobustScalerState(
        median=median,
        scale=scale,
    )


def transform_robust(
    X: torch.Tensor,
    scaler: RobustScalerState,
) -> torch.Tensor:
    Z = (
        X - scaler.median
    ) / scaler.scale

    return torch.tanh(Z)


# ============================================================
# TT-SVD approximation
# ============================================================

def normalize_state(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return x / (
        torch.linalg.vector_norm(x)
        + eps
    )


def tt_svd_vector(
    x: torch.Tensor,
    modes: Tuple[int, ...] = TT_MODES,
    max_rank: int = TT_RANK,
) -> torch.Tensor:
    if int(np.prod(modes)) != x.numel():
        raise ValueError(
            f"TT modes {modes} do not match "
            f"vector size {x.numel()}."
        )

    tensor = x.reshape(*modes)

    cores = []
    r_prev = 1
    remainder = tensor

    for k in range(len(modes) - 1):
        n_k = modes[k]

        matrix = remainder.reshape(
            r_prev * n_k,
            -1,
        )

        U, S, Vh = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )

        rank = min(
            max_rank,
            U.shape[1],
        )

        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        core = U.reshape(
            r_prev,
            n_k,
            rank,
        )

        cores.append(core)

        remainder = S.unsqueeze(1) * Vh
        r_prev = rank

        remainder = remainder.reshape(
            r_prev,
            *modes[k + 1:],
        )

    cores.append(
        remainder.reshape(
            r_prev,
            modes[-1],
            1,
        )
    )

    reconstructed = cores[0]

    for core in cores[1:]:
        reconstructed = torch.einsum(
            "...a,aib->...ib",
            reconstructed,
            core,
        )

    return (
        reconstructed
        .squeeze(0)
        .squeeze(-1)
        .reshape(-1)
    )


def tt_encode_dataset(
    X: torch.Tensor,
    rank: int = TT_RANK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = []
    fidelities = []

    with torch.no_grad():
        for x in X:
            exact = normalize_state(x)

            approximation = tt_svd_vector(
                x,
                max_rank=rank,
            )

            approximation = normalize_state(
                approximation
            )

            fidelity = torch.abs(
                torch.dot(
                    exact,
                    approximation,
                )
            ) ** 2

            encoded.append(
                approximation
            )

            fidelities.append(
                fidelity
            )

    return (
        torch.stack(encoded),
        torch.stack(fidelities),
    )


# ============================================================
# Differentiable quantum gates
# ============================================================

def rx(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)

    return torch.stack(
        [
            torch.stack([c, -1j * s]),
            torch.stack([-1j * s, c]),
        ]
    ).to(COMPLEX)


def ry(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)

    return torch.stack(
        [
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ]
    ).to(COMPLEX)


def rz(theta: torch.Tensor) -> torch.Tensor:
    a = torch.exp(-0.5j * theta)
    b = torch.exp(0.5j * theta)
    z = torch.zeros_like(a)

    return torch.stack(
        [
            torch.stack([a, z]),
            torch.stack([z, b]),
        ]
    ).to(COMPLEX)


def apply_1q(
    state: torch.Tensor,
    gate: torch.Tensor,
    wire: int,
) -> torch.Tensor:
    batch_size = state.shape[0]

    psi = state.reshape(
        batch_size,
        *([2] * NUM_QUBITS),
    )

    axis = wire + 1

    permutation = (
        [0]
        + [
            i
            for i in range(
                1,
                NUM_QUBITS + 1,
            )
            if i != axis
        ]
        + [axis]
    )

    psi = psi.permute(
        *permutation
    ).contiguous()

    original_shape = psi.shape

    psi = psi.reshape(
        -1,
        2,
    )

    psi = torch.einsum(
        "bi,ji->bj",
        psi,
        gate,
    )

    psi = psi.reshape(
        original_shape
    )

    inverse = np.argsort(
        permutation
    )

    psi = psi.permute(
        *inverse
    ).contiguous()

    return psi.reshape(
        batch_size,
        -1,
    )


def apply_cnot(
    state: torch.Tensor,
    control: int,
    target: int,
) -> torch.Tensor:
    dim = 2 ** NUM_QUBITS

    indices = torch.arange(
        dim,
        device=state.device,
    )

    control_bit = (
        indices
        >> (
            NUM_QUBITS
            - 1
            - control
        )
    ) & 1

    target_mask = (
        1
        << (
            NUM_QUBITS
            - 1
            - target
        )
    )

    mapped = torch.where(
        control_bit.bool(),
        indices ^ target_mask,
        indices,
    )

    return state[:, mapped]


def z_expectations(
    state: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.abs(
        state
    ) ** 2

    basis_indices = torch.arange(
        2 ** NUM_QUBITS,
        device=state.device,
    )

    outputs = []

    for qubit in range(
        NUM_QUBITS
    ):
        bit = (
            basis_indices
            >> (
                NUM_QUBITS
                - 1
                - qubit
            )
        ) & 1

        sign = (
            1.0
            - 2.0
            * bit.float()
        )

        expectation = torch.sum(
            probabilities
            * sign.unsqueeze(0),
            dim=1,
        )

        outputs.append(
            expectation
        )

    return torch.stack(
        outputs,
        dim=1,
    )


# ============================================================
# Architecture representation
# ============================================================

@dataclass
class Architecture:
    depth: int
    patterns: Tuple[str, ...]


def edges_for_pattern(
    pattern: str,
) -> List[Tuple[int, int]]:
    if pattern == "ring":
        return [
            (
                q,
                (q + 1) % NUM_QUBITS,
            )
            for q in range(
                NUM_QUBITS
            )
        ]

    if pattern == "linear":
        return [
            (q, q + 1)
            for q in range(
                NUM_QUBITS - 1
            )
        ]

    if pattern == "brick_even":
        return [
            (q, q + 1)
            for q in range(
                0,
                NUM_QUBITS - 1,
                2,
            )
        ]

    if pattern == "brick_odd":
        edges = [
            (q, q + 1)
            for q in range(
                1,
                NUM_QUBITS - 1,
                2,
            )
        ]

        edges.append(
            (
                NUM_QUBITS - 1,
                0,
            )
        )

        return edges

    raise ValueError(
        f"Unknown entangling pattern: {pattern}"
    )


def architecture_stats(
    architecture: Architecture,
) -> Dict[str, int]:
    n1 = (
        architecture.depth
        * NUM_QUBITS
        * 3
    )

    n2 = sum(
        len(
            edges_for_pattern(
                pattern
            )
        )
        for pattern in architecture.patterns
    )

    return {
        "depth": architecture.depth,
        "n1": n1,
        "n2": n2,
    }


def naive_architecture() -> Architecture:
    depth = max(
        DEPTH_CHOICES
    )

    return Architecture(
        depth=depth,
        patterns=tuple(
            ["ring"] * depth
        ),
    )


# ============================================================
# VQC
# ============================================================

class VQC(nn.Module):
    def __init__(
        self,
        architecture: Architecture,
    ):
        super().__init__()

        self.architecture = architecture

        self.theta = nn.Parameter(
            0.05
            * torch.randn(
                architecture.depth,
                NUM_QUBITS,
                3,
            )
        )

    def forward(
        self,
        amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        state = amplitudes.to(
            DEVICE,
            dtype=COMPLEX,
        )

        for layer in range(
            self.architecture.depth
        ):
            for qubit in range(
                NUM_QUBITS
            ):
                state = apply_1q(
                    state,
                    rx(
                        self.theta[
                            layer,
                            qubit,
                            0,
                        ]
                    ),
                    qubit,
                )

                state = apply_1q(
                    state,
                    ry(
                        self.theta[
                            layer,
                            qubit,
                            1,
                        ]
                    ),
                    qubit,
                )

                state = apply_1q(
                    state,
                    rz(
                        self.theta[
                            layer,
                            qubit,
                            2,
                        ]
                    ),
                    qubit,
                )

            pattern = (
                self.architecture.patterns[
                    layer
                ]
            )

            for control, target in edges_for_pattern(
                pattern
            ):
                state = apply_cnot(
                    state,
                    control,
                    target,
                )

        observables = z_expectations(
            state
        )

        return observables[
            :,
            :NUM_CLASSES,
        ]


# ============================================================
# Transformer architecture policy
# ============================================================

class QASPolicy(nn.Module):
    def __init__(
        self,
        feature_dim: int = INPUT_DIM,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()

        max_depth = max(
            DEPTH_CHOICES
        )

        self.feature_proj = nn.Linear(
            feature_dim,
            d_model,
        )

        self.layer_tokens = nn.Parameter(
            0.02
            * torch.randn(
                1,
                max_depth,
                d_model,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.depth_head = nn.Linear(
            d_model,
            len(DEPTH_CHOICES),
        )

        self.pattern_head = nn.Linear(
            d_model,
            len(ENT_PATTERNS),
        )

        self._initialize_policy()

    def _initialize_policy(
        self,
    ) -> None:
        with torch.no_grad():
            self.depth_head.bias.zero_()
            self.depth_head.bias[-1] = 0.4

            self.pattern_head.bias.zero_()

            ring_index = ENT_PATTERNS.index(
                "ring"
            )

            self.pattern_head.bias[
                ring_index
            ] = 0.4

    def forward(
        self,
        context: torch.Tensor,
    ):
        task_embedding = (
            self.feature_proj(
                context
            )
            .mean(
                dim=0,
                keepdim=True,
            )
        )

        hidden = (
            self.layer_tokens
            + task_embedding.unsqueeze(1)
        )

        hidden = self.encoder(
            hidden
        )

        pooled = hidden.mean(
            dim=1
        )

        depth_logits = (
            self.depth_head(
                pooled
            )
            .squeeze(0)
        )

        pattern_logits = (
            self.pattern_head(
                hidden
            )
            .squeeze(0)
        )

        return (
            depth_logits,
            pattern_logits,
        )


# ============================================================
# Architecture sampling
# ============================================================

@dataclass
class PolicySample:
    architecture: Architecture
    log_prob: torch.Tensor
    entropy: torch.Tensor


def sample_architecture(
    policy: QASPolicy,
    context: torch.Tensor,
) -> PolicySample:
    (
        depth_logits,
        pattern_logits,
    ) = policy(
        context
    )

    depth_distribution = (
        torch.distributions.Categorical(
            logits=depth_logits
        )
    )

    depth_index = (
        depth_distribution.sample()
    )

    depth = DEPTH_CHOICES[
        int(
            depth_index.item()
        )
    ]

    log_probability = (
        depth_distribution.log_prob(
            depth_index
        )
    )

    entropy = (
        depth_distribution.entropy()
    )

    patterns = []

    for layer in range(
        depth
    ):
        pattern_distribution = (
            torch.distributions.Categorical(
                logits=pattern_logits[
                    layer
                ]
            )
        )

        pattern_index = (
            pattern_distribution.sample()
        )

        patterns.append(
            ENT_PATTERNS[
                int(
                    pattern_index.item()
                )
            ]
        )

        log_probability += (
            pattern_distribution.log_prob(
                pattern_index
            )
        )

        entropy += (
            pattern_distribution.entropy()
        )

    entropy /= (
        depth + 1
    )

    return PolicySample(
        architecture=Architecture(
            depth=depth,
            patterns=tuple(
                patterns
            ),
        ),
        log_prob=log_probability,
        entropy=entropy,
    )


def greedy_architecture(
    policy: QASPolicy,
    context: torch.Tensor,
) -> Architecture:
    with torch.no_grad():
        (
            depth_logits,
            pattern_logits,
        ) = policy(
            context
        )

        depth_index = int(
            depth_logits.argmax().item()
        )

        depth = DEPTH_CHOICES[
            depth_index
        ]

        patterns = []

        for layer in range(
            depth
        ):
            pattern_index = int(
                pattern_logits[
                    layer
                ]
                .argmax()
                .item()
            )

            patterns.append(
                ENT_PATTERNS[
                    pattern_index
                ]
            )

    return Architecture(
        depth=depth,
        patterns=tuple(
            patterns
        ),
    )


# ============================================================
# EWC
# ============================================================

@dataclass
class EWCState:
    means: Dict[str, torch.Tensor]
    fisher: Dict[str, torch.Tensor]


def estimate_fisher(
    policy: QASPolicy,
    context: torch.Tensor,
    num_samples: int = 24,
) -> EWCState:
    fisher = {
        name: torch.zeros_like(
            parameter
        )
        for name, parameter
        in policy.named_parameters()
        if parameter.requires_grad
    }

    policy.train()

    for _ in range(
        num_samples
    ):
        policy.zero_grad(
            set_to_none=True
        )

        sample = sample_architecture(
            policy,
            context,
        )

        (
            -sample.log_prob
        ).backward()

        for name, parameter in (
            policy.named_parameters()
        ):
            if parameter.grad is not None:
                fisher[name] += (
                    parameter.grad.detach()
                    ** 2
                ) / num_samples

    means = {
        name: parameter.detach().clone()
        for name, parameter
        in policy.named_parameters()
        if parameter.requires_grad
    }

    return EWCState(
        means=means,
        fisher=fisher,
    )


def ewc_penalty(
    policy: QASPolicy,
    state: Optional[EWCState],
) -> torch.Tensor:
    if state is None:
        return torch.tensor(
            0.0,
            device=DEVICE,
        )

    penalty = torch.tensor(
        0.0,
        device=DEVICE,
    )

    for name, parameter in (
        policy.named_parameters()
    ):
        if name not in state.fisher:
            continue

        penalty += torch.sum(
            state.fisher[name]
            * (
                parameter
                - state.means[name]
            ) ** 2
        )

    return 0.5 * penalty


# ============================================================
# KL(pi_phi || pi_ref)
# ============================================================

def categorical_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    probability = torch.softmax(
        current_logits,
        dim=-1,
    )

    log_probability = torch.log_softmax(
        current_logits,
        dim=-1,
    )

    log_reference = torch.log_softmax(
        reference_logits,
        dim=-1,
    )

    return torch.sum(
        probability
        * (
            log_probability
            - log_reference
        ),
        dim=-1,
    )


def policy_kl(
    policy: QASPolicy,
    reference_policy: Optional[QASPolicy],
    context: torch.Tensor,
) -> torch.Tensor:
    if reference_policy is None:
        return torch.tensor(
            0.0,
            device=DEVICE,
        )

    (
        depth_logits,
        pattern_logits,
    ) = policy(
        context
    )

    with torch.no_grad():
        (
            reference_depth,
            reference_pattern,
        ) = reference_policy(
            context
        )

    depth_kl = categorical_kl(
        depth_logits,
        reference_depth,
    )

    pattern_kl = (
        categorical_kl(
            pattern_logits,
            reference_pattern,
        )
        .mean()
    )

    return (
        depth_kl
        + pattern_kl
    )


# ============================================================
# Loss and metrics
# ============================================================

def make_class_weighted_ce(
    y_train: torch.Tensor,
) -> nn.Module:
    counts = torch.bincount(
        y_train,
        minlength=2,
    ).float()

    weights = (
        counts.sum()
        / (
            2.0
            * counts.clamp_min(1)
        )
    )

    weights = torch.clamp(
        weights,
        max=4.0,
    )

    return nn.CrossEntropyLoss(
        weight=weights.to(
            DEVICE
        )
    )


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    model.eval()

    truth = []
    prediction = []

    with torch.no_grad():
        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):
            xb = X[
                start:start + BATCH_SIZE
            ].to(
                DEVICE
            )

            logits = model(
                xb
            )

            pred = (
                logits
                .argmax(
                    dim=1
                )
                .cpu()
            )

            prediction.extend(
                pred.tolist()
            )

            truth.extend(
                y[
                    start:start + BATCH_SIZE
                ].tolist()
            )

    return {
        "acc": accuracy_score(
            truth,
            prediction,
        ),
        "bAcc": balanced_accuracy_score(
            truth,
            prediction,
        ),
        "F1": f1_score(
            truth,
            prediction,
            zero_division=0,
        ),
    }


# ============================================================
# Inner-loop VQC optimization
# ============================================================

def train_vqc(
    architecture: Architecture,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float = VQC_LR,
    initial_state: Optional[
        Dict[str, torch.Tensor]
    ] = None,
) -> VQC:
    model = VQC(
        architecture
    ).to(
        DEVICE
    )

    if initial_state is not None:
        try:
            model.load_state_dict(
                initial_state
            )
        except RuntimeError:
            pass

    criterion = make_class_weighted_ce(
        y_train
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    best_loss = np.inf
    best_state = None

    for _ in range(
        epochs
    ):
        model.train()

        permutation = torch.randperm(
            len(X_train)
        )

        total_loss = 0.0

        for start in range(
            0,
            len(X_train),
            BATCH_SIZE,
        ):
            ids = permutation[
                start:start + BATCH_SIZE
            ]

            xb = X_train[
                ids
            ].to(
                DEVICE
            )

            yb = y_train[
                ids
            ].to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                xb
            )

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * len(ids)
            )

        epoch_loss = (
            total_loss
            / len(X_train)
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# Reward
# ============================================================

def predictive_score(
    metrics: Dict[str, float],
) -> float:
    return (
        0.70
        * metrics["bAcc"]
        + 0.30
        * metrics["F1"]
    )


def hardware_cost(
    architecture: Architecture,
) -> float:
    stats = architecture_stats(
        architecture
    )

    reference_n2 = (
        max(
            DEPTH_CHOICES
        )
        * NUM_QUBITS
    )

    return (
        stats["n2"]
        / reference_n2
    )


def architecture_reward(
    metrics: Dict[str, float],
    architecture: Architecture,
) -> float:
    return (
        predictive_score(
            metrics
        )
        - LAMBDA_HW
        * hardware_cost(
            architecture
        )
    )


# ============================================================
# Candidate evaluation
# ============================================================

@dataclass
class CandidateResult:
    architecture: Architecture
    reward: float
    val_metrics: Dict[str, float]
    state_dict: Dict[str, torch.Tensor]
    log_prob: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None


def evaluate_candidate(
    architecture: Architecture,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
) -> CandidateResult:
    model = train_vqc(
        architecture,
        X_train,
        y_train,
        epochs=SEARCH_INNER_EPOCHS,
    )

    metrics = evaluate(
        model,
        X_val,
        y_val,
    )

    reward = architecture_reward(
        metrics,
        architecture,
    )

    return CandidateResult(
        architecture=architecture,
        reward=reward,
        val_metrics=metrics,
        state_dict=copy.deepcopy(
            model.state_dict()
        ),
    )


# ============================================================
# Bi-loop QAS
# ============================================================

def search_task(
    policy: QASPolicy,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    ewc_state: Optional[EWCState] = None,
    reference_policy: Optional[QASPolicy] = None,
    use_cl: bool = False,
):
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=POLICY_LR,
    )

    context = X_train[
        :min(
            192,
            len(X_train),
        )
    ].to(
        DEVICE
    )

    reward_baseline = None

    anchor = evaluate_candidate(
        naive_architecture(),
        X_train,
        y_train,
        X_val,
        y_val,
    )

    best_candidate = anchor

    anchor_stats = architecture_stats(
        anchor.architecture
    )

    print(
        "      Anchor ring | "
        f"R={anchor.reward:.4f} | "
        f"bAcc={anchor.val_metrics['bAcc']:.4f} | "
        f"F1={anchor.val_metrics['F1']:.4f} | "
        f"G2={anchor_stats['n2']}"
    )

    for outer in range(
        SEARCH_STEPS
    ):
        candidate_results = []

        for _ in range(
            CANDIDATES_PER_STEP
        ):
            policy_sample = sample_architecture(
                policy,
                context,
            )

            result = evaluate_candidate(
                policy_sample.architecture,
                X_train,
                y_train,
                X_val,
                y_val,
            )

            result.log_prob = (
                policy_sample.log_prob
            )

            result.entropy = (
                policy_sample.entropy
            )

            candidate_results.append(
                result
            )

            if (
                result.reward
                > best_candidate.reward
            ):
                best_candidate = result

        rewards = np.asarray(
            [
                result.reward
                for result
                in candidate_results
            ],
            dtype=np.float32,
        )

        mean_reward = float(
            rewards.mean()
        )

        if reward_baseline is None:
            reward_baseline = mean_reward

        advantages = torch.tensor(
            rewards - reward_baseline,
            dtype=REAL,
            device=DEVICE,
        )

        if (
            advantages.numel() > 1
            and advantages.std() > 1e-6
        ):
            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std()
                + 1e-6
            )

        log_probs = torch.stack(
            [
                result.log_prob
                for result
                in candidate_results
            ]
        )

        entropy = torch.stack(
            [
                result.entropy
                for result
                in candidate_results
            ]
        ).mean()

        reinforce_loss = -torch.mean(
            advantages.detach()
            * log_probs
        )

        L_ewc = (
            ewc_penalty(
                policy,
                ewc_state,
            )
            if (
                use_cl
                and ewc_state is not None
            )
            else torch.tensor(
                0.0,
                device=DEVICE,
            )
        )

        L_kl = (
            policy_kl(
                policy,
                reference_policy,
                context,
            )
            if (
                use_cl
                and reference_policy is not None
            )
            else torch.tensor(
                0.0,
                device=DEVICE,
            )
        )

        policy_loss = (
            reinforce_loss
            + MU_EWC * L_ewc
            + ETA_KL * L_kl
            - ENTROPY_COEF * entropy
        )

        policy_optimizer.zero_grad(
            set_to_none=True
        )

        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            1.0,
        )

        policy_optimizer.step()

        reward_baseline = (
            0.9 * reward_baseline
            + 0.1 * mean_reward
        )

        if (
            outer == 0
            or (outer + 1) % 5 == 0
        ):
            best_stats = architecture_stats(
                best_candidate.architecture
            )

            print(
                f"      outer={outer + 1:02d} | "
                f"meanR={mean_reward:.4f} | "
                f"bestR={best_candidate.reward:.4f} | "
                f"G2={best_stats['n2']} | "
                f"EWC={float(L_ewc.detach()):.6f} | "
                f"KL={float(L_kl.detach()):.6f}"
            )

    greedy_arch = greedy_architecture(
        policy,
        context,
    )

    greedy_result = evaluate_candidate(
        greedy_arch,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    if (
        greedy_result.reward
        > best_candidate.reward
    ):
        best_candidate = greedy_result

    return (
        best_candidate,
        context,
    )


# ============================================================
# Final test evaluation
# ============================================================

def final_evaluation(
    candidate: CandidateResult,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> Dict:
    model = train_vqc(
        candidate.architecture,
        X_train,
        y_train,
        epochs=FINAL_EPOCHS,
        initial_state=candidate.state_dict,
    )

    metrics = evaluate(
        model,
        X_test,
        y_test,
    )

    stats = architecture_stats(
        candidate.architecture
    )

    return {
        **metrics,
        "reward": candidate.reward,
        "depth": stats["depth"],
        "n1": stats["n1"],
        "n2": stats["n2"],
        "patterns": "|".join(
            candidate.architecture.patterns
        ),
    }


# ============================================================
# Prepare task without leakage
# ============================================================

def prepare_task(
    X_raw: torch.Tensor,
    y: torch.Tensor,
):
    (
        X_train_raw,
        y_train,
        X_val_raw,
        y_val,
        X_test_raw,
        y_test,
    ) = time_split(
        X_raw,
        y,
    )

    scaler = fit_robust_scaler(
        X_train_raw
    )

    X_train_scaled = transform_robust(
        X_train_raw,
        scaler,
    )

    X_val_scaled = transform_robust(
        X_val_raw,
        scaler,
    )

    X_test_scaled = transform_robust(
        X_test_raw,
        scaler,
    )

    X_train, F_train = tt_encode_dataset(
        X_train_scaled,
        rank=TT_RANK,
    )

    X_val, F_val = tt_encode_dataset(
        X_val_scaled,
        rank=TT_RANK,
    )

    X_test, F_test = tt_encode_dataset(
        X_test_scaled,
        rank=TT_RANK,
    )

    fidelity = torch.cat(
        [
            F_train,
            F_val,
            F_test,
        ]
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        float(
            fidelity.mean()
        ),
    )


# ============================================================
# Sequential experiment
# ============================================================

def run_seed(
    tasks,
    seed: int,
) -> List[Dict]:
    set_seed(seed)

    policy_nocl = (
        QASPolicy()
        .to(DEVICE)
    )

    policy_cl = (
        QASPolicy()
        .to(DEVICE)
    )

    ewc_state = None
    reference_policy = None

    results = []

    for task_id, (
        X_raw,
        y,
    ) in enumerate(
        tasks,
        start=1,
    ):
        print(
            "\n"
            + "=" * 78
        )

        print(
            f"Seed {seed} | "
            f"Financial Task {task_id}"
        )

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            mean_fidelity,
        ) = prepare_task(
            X_raw,
            y,
        )

        print(
            f"  TT rank={TT_RANK}, "
            f"mean fidelity="
            f"{mean_fidelity:.6f}"
        )

        # ----------------------------------------------------
        # Naive-VQC
        # ----------------------------------------------------

        print("\n  [Naive-VQC]")

        naive_arch = naive_architecture()

        naive_model = train_vqc(
            naive_arch,
            X_train,
            y_train,
            epochs=FINAL_EPOCHS,
        )

        naive_metrics = evaluate(
            naive_model,
            X_test,
            y_test,
        )

        naive_stats = architecture_stats(
            naive_arch
        )

        results.append(
            {
                "seed": seed,
                "task": task_id,
                "method": "Naive-VQC",
                **naive_metrics,
                "reward": np.nan,
                "depth": naive_stats["depth"],
                "n1": naive_stats["n1"],
                "n2": naive_stats["n2"],
                "patterns": "|".join(
                    naive_arch.patterns
                ),
                "tt_fidelity": mean_fidelity,
            }
        )

        print(
            f"      Test | "
            f"acc={naive_metrics['acc']:.4f} | "
            f"bAcc={naive_metrics['bAcc']:.4f} | "
            f"F1={naive_metrics['F1']:.4f} | "
            f"G2={naive_stats['n2']}"
        )

        # ----------------------------------------------------
        # QAS-No-CL
        # ----------------------------------------------------

        print("\n  [QAS-No-CL]")

        qas_candidate, _ = search_task(
            policy_nocl,
            X_train,
            y_train,
            X_val,
            y_val,
            use_cl=False,
        )

        qas_result = final_evaluation(
            qas_candidate,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        results.append(
            {
                "seed": seed,
                "task": task_id,
                "method": "QAS-No-CL",
                **qas_result,
                "tt_fidelity": mean_fidelity,
            }
        )

        print(
            f"      Test | "
            f"acc={qas_result['acc']:.4f} | "
            f"bAcc={qas_result['bAcc']:.4f} | "
            f"F1={qas_result['F1']:.4f} | "
            f"G2={qas_result['n2']}"
        )

        # ----------------------------------------------------
        # CL-QAS
        # ----------------------------------------------------

        print("\n  [CL-QAS]")

        cl_candidate, context = search_task(
            policy_cl,
            X_train,
            y_train,
            X_val,
            y_val,
            ewc_state=ewc_state,
            reference_policy=reference_policy,
            use_cl=True,
        )

        cl_result = final_evaluation(
            cl_candidate,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        results.append(
            {
                "seed": seed,
                "task": task_id,
                "method": "CL-QAS",
                **cl_result,
                "tt_fidelity": mean_fidelity,
            }
        )

        print(
            f"      Test | "
            f"acc={cl_result['acc']:.4f} | "
            f"bAcc={cl_result['bAcc']:.4f} | "
            f"F1={cl_result['F1']:.4f} | "
            f"G2={cl_result['n2']}"
        )

        ewc_state = estimate_fisher(
            policy_cl,
            context,
        )

        reference_policy = (
            copy.deepcopy(
                policy_cl
            )
            .eval()
        )

        for parameter in (
            reference_policy.parameters()
        ):
            parameter.requires_grad_(
                False
            )

    return results


# ============================================================
# Reporting
# ============================================================

def summarize_results(
    results: List[Dict],
) -> None:
    print(
        "\n"
        + "=" * 82
    )

    print(
        "OVERALL FINANCIAL PREDICTIVE PERFORMANCE"
    )

    print(
        "=" * 82
    )

    methods = (
        "Naive-VQC",
        "QAS-No-CL",
        "CL-QAS",
    )

    for method in methods:
        rows = [
            row
            for row in results
            if row["method"] == method
        ]

        print(
            f"\n{method}"
        )

        for metric in (
            "acc",
            "bAcc",
            "F1",
            "n2",
        ):
            values = np.asarray(
                [
                    row[metric]
                    for row in rows
                ],
                dtype=float,
            )

            print(
                f"  {metric:>5}: "
                f"{values.mean():.4f} "
                f"± "
                f"{values.std(ddof=1):.4f}"
            )

        if method != "Naive-VQC":
            rewards = np.asarray(
                [
                    row["reward"]
                    for row in rows
                ],
                dtype=float,
            )

            print(
                f" reward: "
                f"{rewards.mean():.4f} "
                f"± "
                f"{rewards.std(ddof=1):.4f}"
            )

    fidelity = np.asarray(
        [
            row["tt_fidelity"]
            for row in results
            if row["method"] == "CL-QAS"
        ],
        dtype=float,
    )

    print(
        "\nTT representation fidelity: "
        f"{fidelity.mean():.6f} "
        f"± "
        f"{fidelity.std(ddof=1):.6f}"
    )


# ============================================================
# CSV export
# ============================================================

def save_results(
    results: List[Dict],
    path: str,
) -> None:
    fields = (
        "seed",
        "task",
        "method",
        "acc",
        "bAcc",
        "F1",
        "reward",
        "depth",
        "n1",
        "n2",
        "patterns",
        "tt_fidelity",
    )

    with open(
        path,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in fields
                }
            )

    print(
        f"\nSaved results to:\n"
        f"    {path}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Revised CL-QAS financial experiment."
        )
    )

    parser.add_argument(
        "--csv",
        nargs="*",
        default=[],
        help=(
            "Financial CSV files containing Close or Adj Close. "
            "If no valid CSV is supplied, synthetic data are used."
        ),
    )

    parser.add_argument(
        "--tasks",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help=(
            "Future prediction horizon."
        ),
    )

    parser.add_argument(
        "--min-per-task",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run reduced search/training settings for debugging."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "revised_finance_clqas_results.csv"
        ),
    )

    args = parser.parse_args()

    global SEARCH_STEPS
    global CANDIDATES_PER_STEP
    global SEARCH_INNER_EPOCHS
    global FINAL_EPOCHS

    if args.quick:
        SEARCH_STEPS = 3
        CANDIDATES_PER_STEP = 2
        SEARCH_INNER_EPOCHS = 3
        FINAL_EPOCHS = 10

        seeds = (11,)

        print(
            "[Quick mode] Reduced search/training settings."
        )
    else:
        seeds = DEFAULT_SEEDS

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load real financial data if provided
    # --------------------------------------------------------

    closes = []

    for path in args.csv:
        if not os.path.exists(
            path
        ):
            print(
                f"[WARN] File not found: {path}"
            )
            continue

        try:
            close_i = load_close_from_csv(
                path
            )

            closes.append(
                close_i
            )

            print(
                f"[INFO] Loaded {path} "
                f"({len(close_i)} observations)"
            )

        except Exception as exc:
            print(
                f"[WARN] Skip {path}: {exc}"
            )

    # --------------------------------------------------------
    # Automatic synthetic fallback
    # --------------------------------------------------------

    if closes:
        close = np.concatenate(
            closes
        )

        print(
            f"[INFO] Using real financial data: "
            f"{len(close)} total observations."
        )

    else:
        print(
            "[INFO] No valid CSV supplied. "
            "Using synthetic financial data."
        )

        close = make_synthetic_close(
            n=8000,
            regimes=8,
            seed=777,
        )

        print(
            f"[INFO] Synthetic series generated: "
            f"{len(close)} observations."
        )

    # --------------------------------------------------------
    # Build sequential tasks
    # --------------------------------------------------------

    tasks = build_finance_tasks(
        close,
        n_tasks=args.tasks,
        lookback=LOOKBACK,
        horizon=args.horizon,
        min_per_task=args.min_per_task,
    )

    if len(tasks) < 2:
        raise RuntimeError(
            "Too few valid financial tasks."
        )

    print(
        f"\nPrepared {len(tasks)} "
        f"sequential financial tasks."
    )

    # --------------------------------------------------------
    # Run independent seeds
    # --------------------------------------------------------

    all_results: List[Dict] = []

    for seed in seeds:
        seed_results = run_seed(
            tasks,
            seed,
        )

        all_results.extend(
            seed_results
        )

    # --------------------------------------------------------
    # Report and save
    # --------------------------------------------------------

    summarize_results(
        all_results
    )

    save_results(
        all_results,
        args.output,
    )


if __name__ == "__main__":
    main()

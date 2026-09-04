#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ecg_clqas_ablation_table1_consistent.py

TT-rank and component ablations for CL-QAS using a Table-1-consistent
experimental protocol.

Key design principles
---------------------
1. Every ablation uses exactly the same ECG partitions for a given seed.
2. Architecture search uses:
       80% search-training
       10% validation
       10% untouched test
3. After architecture selection, the selected architecture is re-trained
   from scratch on train + validation = 90%, then tested on the remaining 10%.
4. ECG metrics are calculated from pooled held-out predictions across all
   sequential ECG records for each random seed.
5. Mean +/- std in the manuscript are then computed across random seeds.
6. Component ablations differ ONLY in EWC/KL regularization.
7. TT-rank ablations differ ONLY in TT rank.
8. Rank 6 is omitted because rank 4 is already the maximum useful TT rank
   for the adopted (4, 16, 4) tensorization.

Component ablations
-------------------
CL-QAS:
    EWC + KL

CL-QAS-no-EWC:
    KL only

CL-QAS-no-KL:
    EWC only

QAS-No-CL:
    neither EWC nor KL

TT-rank ablation
----------------
r in {1, 2, 3, 4}

Dataset
-------
MIT-BIH ECG, binary AAMI N vs V classification.

Sequential tasks
----------------
Records:
    105, 106, 109, 114, 116, 119, 200, 201

Input
-----
Each ECG beat is resampled to a 256-dimensional vector.

Quantum representation
----------------------
The normalized 256-D vector is approximated by TT-SVD using modes
(4, 16, 4). The reconstructed vector is normalized and used as an
8-qubit amplitude state.

Architecture search
-------------------
The Transformer policy selects:
    - circuit depth from {2, 3, 4}
    - one entangling pattern per layer:
        ring
        linear
        brick_even
        brick_odd

Each VQC layer contains:
    RX -> RY -> RZ
on every qubit, followed by the selected CNOT entangling pattern.

Continual regularization
------------------------
EWC:
    L_EWC = 1/2 sum_i F_i (phi_i - phi_i_old)^2

KL:
    L_KL = D_KL(pi_phi || pi_ref)

Policy objective:
    L_policy =
        L_REINFORCE
        + mu * L_EWC
        + eta * L_KL
        - beta_ent * entropy

Dependencies
------------
pip install wfdb torch numpy pandas scikit-learn
"""

# ============================================================
# Imports
# ============================================================

import copy
import os
import random

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

try:
    import wfdb
except ImportError as exc:
    raise ImportError(
        "Please install wfdb with:\n"
        "    pip install wfdb"
    ) from exc


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

REAL = torch.float32
COMPLEX = torch.complex64


# ------------------------------------------------------------
# ECG dataset
# ------------------------------------------------------------

ECG_RECORDS = [
    105,
    106,
    109,
    114,
    116,
    119,
    200,
    201,
]

AAMI_N = {
    "N",
    "L",
    "R",
    "e",
    "j",
}

AAMI_V = {
    "V",
    "E",
}

INPUT_DIM = 256

MAX_BEATS_PER_RECORD = 800

ECG_WINDOW_SEC = 0.6


# ------------------------------------------------------------
# TT
# ------------------------------------------------------------

TT_MODES = (
    4,
    16,
    4,
)

TT_RANKS_TO_TEST = (
    1,
    2,
    3,
    4,
)

DEFAULT_TT_RANK = 3


# ------------------------------------------------------------
# Quantum circuit
# ------------------------------------------------------------

NUM_QUBITS = 8

NUM_CLASSES = 2

DEPTH_CHOICES = (
    2,
    3,
    4,
)

ENT_PATTERNS = (
    "ring",
    "linear",
    "brick_even",
    "brick_odd",
)


# ------------------------------------------------------------
# Architecture search
# ------------------------------------------------------------

SEARCH_STEPS = 10

CANDIDATES_PER_STEP = 3

SEARCH_INNER_EPOCHS = 8


# ------------------------------------------------------------
# Final model training
# ------------------------------------------------------------

FINAL_REFIT_EPOCHS = 40

BATCH_SIZE = 64


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

VQC_LR = 3e-3

POLICY_LR = 1e-3

WEIGHT_DECAY = 1e-5


# ------------------------------------------------------------
# Continual regularization
# ------------------------------------------------------------

MU_EWC = 0.5

ETA_KL = 0.01

ENTROPY_COEF = 0.002


# ------------------------------------------------------------
# Hardware-aware reward
# ------------------------------------------------------------

LAMBDA_HW = 0.01


# ------------------------------------------------------------
# Experimental split
# ------------------------------------------------------------

SEARCH_TRAIN_FRAC = 0.80

SEARCH_VAL_FRAC = 0.10

# Remaining 10% is test.

SPLIT_SEED_BASE = 5000


# ------------------------------------------------------------
# Random seeds
# ------------------------------------------------------------

SEEDS = (
    11,
    22,
    33,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# ECG preprocessing
# ============================================================

def map_symbol_to_binary(
    symbol: str,
) -> Optional[int]:

    if symbol in AAMI_N:
        return 0

    if symbol in AAMI_V:
        return 1

    return None


def choose_channel(
    signal_names: List[str],
) -> int:

    upper = [
        name.upper()
        for name in signal_names
    ]

    if "MLII" in upper:

        return upper.index(
            "MLII"
        )

    return 0


def fft_bandpass(
    signal: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 40.0,
) -> np.ndarray:

    n = len(signal)

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / fs,
    )

    spectrum = np.fft.rfft(
        signal
    )

    mask = (
        (frequencies >= low)
        & (frequencies <= high)
    )

    spectrum *= mask

    filtered = np.fft.irfft(
        spectrum,
        n=n,
    )

    return filtered.astype(
        np.float32
    )


def beat_vector_256(
    segment: np.ndarray,
    target_len: int = INPUT_DIM,
) -> np.ndarray:

    x = np.asarray(
        segment,
        dtype=np.float32,
    )

    source_grid = np.linspace(
        0.0,
        1.0,
        len(x),
        endpoint=False,
        dtype=np.float32,
    )

    target_grid = np.linspace(
        0.0,
        1.0,
        target_len,
        endpoint=False,
        dtype=np.float32,
    )

    x_resampled = np.interp(
        target_grid,
        source_grid,
        x,
    ).astype(
        np.float32
    )

    # Per-beat normalization.
    mean = float(
        x_resampled.mean()
    )

    std = float(
        x_resampled.std()
        + 1e-6
    )

    x_resampled = (
        x_resampled
        - mean
    ) / std

    x_resampled = np.clip(
        x_resampled,
        -5.0,
        5.0,
    )

    return x_resampled.astype(
        np.float32
    )


# ============================================================
# Robust MIT-BIH loader
# ============================================================

def _try_load_wfdb_record(
    record: int,
    pn_dir: str,
):

    signal, fields = wfdb.rdsamp(
        str(record),
        pn_dir=pn_dir,
    )

    annotation = wfdb.rdann(
        str(record),
        "atr",
        pn_dir=pn_dir,
    )

    return (
        signal,
        fields,
        annotation,
    )


def load_record(
    record: int,
    max_beats: int = MAX_BEATS_PER_RECORD,
    window_sec: float = ECG_WINDOW_SEC,
    min_per_class: int = 10,
):

    """
    Loads one MIT-BIH record.

    Tries the version-specific PhysioNet directory first to avoid
    unnecessary version lookup behavior in some wfdb installations.
    """

    errors = []

    signal = None
    fields = None
    annotation = None

    for pn_dir in (
        "mitdb/1.0.0",
        "mitdb",
    ):

        try:

            (
                signal,
                fields,
                annotation,
            ) = _try_load_wfdb_record(
                record,
                pn_dir,
            )

            break

        except Exception as exc:

            errors.append(
                f"{pn_dir}: {exc}"
            )

    if signal is None:

        raise RuntimeError(
            f"Could not load MIT-BIH record {record}.\n"
            + "\n".join(errors)
        )

    channel = choose_channel(
        fields["sig_name"]
    )

    x = signal[
        :,
        channel,
    ]

    fs = float(
        fields["fs"]
    )

    x = fft_bandpass(
        x,
        fs,
    )

    half_window = int(
        window_sec * fs
    )

    X = []

    y = []

    for (
        sample,
        symbol,
    ) in zip(
        annotation.sample,
        annotation.symbol,
    ):

        label = map_symbol_to_binary(
            symbol
        )

        if label is None:
            continue

        start = (
            sample
            - half_window
        )

        end = (
            sample
            + half_window
        )

        if (
            start < 0
            or end >= len(x)
        ):
            continue

        segment = x[
            start:end
        ]

        X.append(
            beat_vector_256(
                segment
            )
        )

        y.append(
            label
        )

        if (
            max_beats is not None
            and len(X) >= max_beats
        ):
            break

    if not X:

        raise RuntimeError(
            f"No valid N/V beats "
            f"for record {record}."
        )

    X = np.stack(
        X
    ).astype(
        np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    counts = np.bincount(
        y,
        minlength=2,
    )

    if (
        counts[0] < min_per_class
        or counts[1] < min_per_class
    ):

        raise RuntimeError(
            f"Record {record}: "
            f"insufficient class counts "
            f"{counts.tolist()}."
        )

    return (
        torch.tensor(
            X,
            dtype=REAL,
        ),
        torch.tensor(
            y,
            dtype=torch.long,
        ),
    )


def load_tasks():

    tasks = []

    for record in ECG_RECORDS:

        X, y = load_record(
            record
        )

        counts = torch.bincount(
            y,
            minlength=2,
        )

        print(
            f"Record {record}: "
            f"N={counts[0].item()}, "
            f"V={counts[1].item()}, "
            f"total={len(y)}"
        )

        tasks.append(
            (
                record,
                X,
                y,
            )
        )

    return tasks


# ============================================================
# Table-1-consistent stratified split
# ============================================================

def stratified_split_table1(
    y: torch.Tensor,
    train_frac: float = SEARCH_TRAIN_FRAC,
    val_frac: float = SEARCH_VAL_FRAC,
    seed: int = 1234,
):

    """
    Search split:
        80% training
        10% validation
        10% test

    After architecture selection, training + validation are merged,
    giving a final 90/10 train/test evaluation.
    """

    rng = np.random.RandomState(
        seed
    )

    y_np = y.cpu().numpy()

    train_indices = []

    val_indices = []

    test_indices = []

    for class_id in (
        0,
        1,
    ):

        indices = np.where(
            y_np == class_id
        )[0].copy()

        rng.shuffle(
            indices
        )

        n = len(
            indices
        )

        n_train = int(
            np.floor(
                train_frac * n
            )
        )

        n_val = int(
            np.floor(
                val_frac * n
            )
        )

        if n >= 3:

            n_train = min(
                n_train,
                n - 2,
            )

            n_val = min(
                n_val,
                n - n_train - 1,
            )

        train_indices.extend(
            indices[
                :n_train
            ].tolist()
        )

        val_indices.extend(
            indices[
                n_train:
                n_train + n_val
            ].tolist()
        )

        test_indices.extend(
            indices[
                n_train + n_val:
            ].tolist()
        )

    rng.shuffle(
        train_indices
    )

    rng.shuffle(
        val_indices
    )

    rng.shuffle(
        test_indices
    )

    return (
        torch.tensor(
            train_indices,
            dtype=torch.long,
        ),
        torch.tensor(
            val_indices,
            dtype=torch.long,
        ),
        torch.tensor(
            test_indices,
            dtype=torch.long,
        ),
    )


# ============================================================
# Shared split cache
# ============================================================

def build_split_cache(
    tasks,
    seed: int,
):

    """
    Builds one fixed partition for each record.

    All component variants and all TT ranks use these exact
    same raw data partitions for a given seed.
    """

    cache = {}

    for task_id, (
        record,
        _,
        y,
    ) in enumerate(
        tasks,
        start=1,
    ):

        split_seed = (
            SPLIT_SEED_BASE
            + 100 * seed
            + task_id
        )

        (
            train_idx,
            val_idx,
            test_idx,
        ) = stratified_split_table1(
            y,
            seed=split_seed,
        )

        cache[
            record
        ] = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }

    return cache


# ============================================================
# Train-only feature normalization
# ============================================================

@dataclass
class StandardizationState:

    mean: torch.Tensor

    std: torch.Tensor


def fit_standardizer(
    X_train: torch.Tensor,
) -> StandardizationState:

    mean = X_train.mean(
        dim=0
    )

    std = X_train.std(
        dim=0
    ) + 1e-6

    return StandardizationState(
        mean=mean,
        std=std,
    )


def standardize(
    X: torch.Tensor,
    state: StandardizationState,
) -> torch.Tensor:

    return (
        X
        - state.mean
    ) / state.std


# ============================================================
# TT-SVD
# ============================================================

def normalize_state(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:

    norm = torch.linalg.vector_norm(
        x
    )

    return x / (
        norm + eps
    )


def tt_svd_vector(
    x: torch.Tensor,
    modes: Tuple[int, ...] = TT_MODES,
    max_rank: int = 3,
) -> torch.Tensor:

    if (
        int(np.prod(modes))
        != x.numel()
    ):

        raise ValueError(
            "TT modes do not match "
            "the vector dimension."
        )

    tensor = x.reshape(
        *modes
    )

    cores = []

    rank_prev = 1

    remainder = tensor

    for k in range(
        len(modes) - 1
    ):

        n_k = modes[
            k
        ]

        matrix = remainder.reshape(
            rank_prev * n_k,
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

        U = U[
            :,
            :rank,
        ]

        S = S[
            :rank
        ]

        Vh = Vh[
            :rank,
            :,
        ]

        core = U.reshape(
            rank_prev,
            n_k,
            rank,
        )

        cores.append(
            core
        )

        remainder = (
            S.unsqueeze(1)
            * Vh
        )

        rank_prev = rank

        remainder = remainder.reshape(
            rank_prev,
            *modes[
                k + 1:
            ],
        )

    cores.append(
        remainder.reshape(
            rank_prev,
            modes[-1],
            1,
        )
    )

    reconstructed = cores[
        0
    ]

    for core in cores[
        1:
    ]:

        reconstructed = torch.einsum(
            "...a,aib->...ib",
            reconstructed,
            core,
        )

    reconstructed = (
        reconstructed
        .squeeze(0)
        .squeeze(-1)
        .reshape(-1)
    )

    return reconstructed


def tt_encode_dataset(
    X: torch.Tensor,
    rank: int,
):

    encoded = []

    fidelities = []

    with torch.no_grad():

        for x in X:

            exact = normalize_state(
                x
            )

            approximation = tt_svd_vector(
                x,
                max_rank=rank,
            )

            approximation = normalize_state(
                approximation
            )

            overlap = torch.dot(
                exact,
                approximation,
            )

            fidelity = (
                torch.abs(
                    overlap
                )
                ** 2
            )

            encoded.append(
                approximation
            )

            fidelities.append(
                fidelity
            )

    return (
        torch.stack(
            encoded
        ),
        torch.stack(
            fidelities
        ),
    )


# ============================================================
# Task preparation
# ============================================================

def prepare_task_table1(
    X_raw,
    y,
    rank,
    split_info,
):

    train_idx = split_info[
        "train_idx"
    ]

    val_idx = split_info[
        "val_idx"
    ]

    test_idx = split_info[
        "test_idx"
    ]

    X_train_raw = X_raw[
        train_idx
    ]

    y_train = y[
        train_idx
    ]

    X_val_raw = X_raw[
        val_idx
    ]

    y_val = y[
        val_idx
    ]

    X_test_raw = X_raw[
        test_idx
    ]

    y_test = y[
        test_idx
    ]

    # Fit normalization on training data only.
    standardizer = fit_standardizer(
        X_train_raw
    )

    X_train_std = standardize(
        X_train_raw,
        standardizer,
    )

    X_val_std = standardize(
        X_val_raw,
        standardizer,
    )

    X_test_std = standardize(
        X_test_raw,
        standardizer,
    )

    X_train, F_train = tt_encode_dataset(
        X_train_std,
        rank,
    )

    X_val, F_val = tt_encode_dataset(
        X_val_std,
        rank,
    )

    X_test, F_test = tt_encode_dataset(
        X_test_std,
        rank,
    )

    fidelity = torch.cat(
        [
            F_train,
            F_val,
            F_test,
        ]
    )

    return {
        "X_train": X_train,
        "y_train": y_train,

        "X_val": X_val,
        "y_val": y_val,

        "X_test": X_test,
        "y_test": y_test,

        "fidelity": float(
            fidelity.mean().item()
        ),
    }


# ============================================================
# Quantum gates
# ============================================================

def rx(theta):

    c = torch.cos(
        theta / 2
    )

    s = torch.sin(
        theta / 2
    )

    return torch.stack(
        [
            torch.stack(
                [
                    c,
                    -1j * s,
                ]
            ),
            torch.stack(
                [
                    -1j * s,
                    c,
                ]
            ),
        ]
    ).to(
        COMPLEX
    )


def ry(theta):

    c = torch.cos(
        theta / 2
    )

    s = torch.sin(
        theta / 2
    )

    return torch.stack(
        [
            torch.stack(
                [
                    c,
                    -s,
                ]
            ),
            torch.stack(
                [
                    s,
                    c,
                ]
            ),
        ]
    ).to(
        COMPLEX
    )


def rz(theta):

    a = torch.exp(
        -0.5j * theta
    )

    b = torch.exp(
        0.5j * theta
    )

    zero = torch.zeros_like(
        a
    )

    return torch.stack(
        [
            torch.stack(
                [
                    a,
                    zero,
                ]
            ),
            torch.stack(
                [
                    zero,
                    b,
                ]
            ),
        ]
    ).to(
        COMPLEX
    )


def apply_1q(
    state,
    gate,
    wire,
):

    batch_size = state.shape[
        0
    ]

    psi = state.reshape(
        batch_size,
        *(
            [2]
            * NUM_QUBITS
        ),
    )

    axis = (
        wire + 1
    )

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
        + [
            axis
        ]
    )

    psi = psi.permute(
        *permutation
    ).contiguous()

    old_shape = psi.shape

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
        old_shape
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
    state,
    control,
    target,
):

    dimension = (
        2 ** NUM_QUBITS
    )

    indices = torch.arange(
        dimension,
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

    return state[
        :,
        mapped,
    ]


def z_expectations(
    state,
):

    probabilities = (
        state.abs()
        ** 2
    )

    indices = torch.arange(
        2 ** NUM_QUBITS,
        device=state.device,
    )

    outputs = []

    for qubit in range(
        NUM_QUBITS
    ):

        bit = (
            indices
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
# Architecture definition
# ============================================================

@dataclass
class Architecture:

    depth: int

    patterns: Tuple[
        str,
        ...
    ]


def edges_for_pattern(
    pattern: str,
):

    if pattern == "ring":

        return [
            (
                q,
                (q + 1)
                % NUM_QUBITS,
            )
            for q in range(
                NUM_QUBITS
            )
        ]

    if pattern == "linear":

        return [
            (
                q,
                q + 1,
            )
            for q in range(
                NUM_QUBITS - 1
            )
        ]

    if pattern == "brick_even":

        return [
            (
                q,
                q + 1,
            )
            for q in range(
                0,
                NUM_QUBITS - 1,
                2,
            )
        ]

    if pattern == "brick_odd":

        edges = [
            (
                q,
                q + 1,
            )
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
        f"Unknown pattern: "
        f"{pattern}"
    )


def architecture_stats(
    architecture: Architecture,
):

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


def naive_architecture():

    depth = 4

    return Architecture(
        depth=depth,
        patterns=tuple(
            ["ring"]
            * depth
        ),
    )


# ============================================================
# VQC
# ============================================================

class VQC(
    nn.Module
):

    def __init__(
        self,
        architecture,
    ):

        super().__init__()

        self.architecture = (
            architecture
        )

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
        amplitudes,
    ):

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
                self.architecture
                .patterns[
                    layer
                ]
            )

            for (
                control,
                target,
            ) in edges_for_pattern(
                pattern
            ):

                state = apply_cnot(
                    state,
                    control,
                    target,
                )

        z = z_expectations(
            state
        )

        return z[
            :,
            :NUM_CLASSES,
        ]


# ============================================================
# QAS policy
# ============================================================

class QASPolicy(
    nn.Module
):

    def __init__(
        self,
        feature_dim=INPUT_DIM,
        d_model=64,
        nhead=8,
        num_layers=2,
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
            len(
                DEPTH_CHOICES
            ),
        )

        self.pattern_head = nn.Linear(
            d_model,
            len(
                ENT_PATTERNS
            ),
        )

    def forward(
        self,
        context,
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


@dataclass
class PolicySample:

    architecture: Architecture

    log_prob: torch.Tensor

    entropy: torch.Tensor


def sample_architecture(
    policy,
    context,
):

    (
        depth_logits,
        pattern_logits,
    ) = policy(
        context
    )

    depth_dist = (
        torch.distributions
        .Categorical(
            logits=depth_logits
        )
    )

    depth_index = (
        depth_dist.sample()
    )

    depth = DEPTH_CHOICES[
        int(
            depth_index.item()
        )
    ]

    log_prob = (
        depth_dist.log_prob(
            depth_index
        )
    )

    entropy = (
        depth_dist.entropy()
    )

    patterns = []

    for layer in range(
        depth
    ):

        pattern_dist = (
            torch.distributions
            .Categorical(
                logits=pattern_logits[
                    layer
                ]
            )
        )

        pattern_index = (
            pattern_dist.sample()
        )

        patterns.append(
            ENT_PATTERNS[
                int(
                    pattern_index.item()
                )
            ]
        )

        log_prob = (
            log_prob
            + pattern_dist.log_prob(
                pattern_index
            )
        )

        entropy = (
            entropy
            + pattern_dist.entropy()
        )

    entropy = (
        entropy
        / (
            depth + 1
        )
    )

    return PolicySample(
        architecture=Architecture(
            depth=depth,
            patterns=tuple(
                patterns
            ),
        ),
        log_prob=log_prob,
        entropy=entropy,
    )


# ============================================================
# EWC
# ============================================================

@dataclass
class EWCState:

    means: Dict[
        str,
        torch.Tensor,
    ]

    fisher: Dict[
        str,
        torch.Tensor,
    ]


def estimate_fisher(
    policy,
    context,
    num_samples=24,
):

    fisher = {
        name: torch.zeros_like(
            parameter
        )
        for (
            name,
            parameter,
        ) in policy.named_parameters()
        if parameter.requires_grad
    }

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

        for (
            name,
            parameter,
        ) in policy.named_parameters():

            if parameter.grad is not None:

                fisher[
                    name
                ] += (
                    parameter.grad.detach()
                    ** 2
                ) / num_samples

    means = {
        name: parameter.detach().clone()
        for (
            name,
            parameter,
        ) in policy.named_parameters()
        if parameter.requires_grad
    }

    return EWCState(
        means=means,
        fisher=fisher,
    )


def ewc_penalty(
    policy,
    state,
):

    if state is None:

        return torch.tensor(
            0.0,
            device=DEVICE,
        )

    penalty = torch.tensor(
        0.0,
        device=DEVICE,
    )

    for (
        name,
        parameter,
    ) in policy.named_parameters():

        if name not in state.fisher:
            continue

        penalty = (
            penalty
            + torch.sum(
                state.fisher[
                    name
                ]
                * (
                    parameter
                    - state.means[
                        name
                    ]
                ) ** 2
            )
        )

    return (
        0.5
        * penalty
    )


# ============================================================
# KL regularization
# ============================================================

def categorical_kl(
    current_logits,
    reference_logits,
):

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
    policy,
    reference_policy,
    context,
):

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

    depth_term = categorical_kl(
        depth_logits,
        reference_depth,
    )

    pattern_term = categorical_kl(
        pattern_logits,
        reference_pattern,
    ).mean()

    return (
        depth_term
        + pattern_term
    )


# ============================================================
# VQC training
# ============================================================

def class_weighted_loss(
    y_train,
):

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


def train_vqc(
    architecture,
    X_train,
    y_train,
    epochs,
    initial_state=None,
):

    model = VQC(
        architecture
    ).to(
        DEVICE
    )

    if initial_state is not None:

        model.load_state_dict(
            initial_state
        )

    criterion = class_weighted_loss(
        y_train
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=VQC_LR,
        weight_decay=WEIGHT_DECAY,
    )

    for _ in range(
        epochs
    ):

        permutation = torch.randperm(
            len(
                X_train
            )
        )

        model.train()

        for start in range(
            0,
            len(
                X_train
            ),
            BATCH_SIZE,
        ):

            indices = permutation[
                start:
                start + BATCH_SIZE
            ]

            xb = X_train[
                indices
            ].to(
                DEVICE
            )

            yb = y_train[
                indices
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

    return model


# ============================================================
# Evaluation
# ============================================================

def predict_vqc(
    model,
    X,
    y,
):

    model.eval()

    y_true = []

    y_pred = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):

            xb = X[
                start:
                start + BATCH_SIZE
            ].to(
                DEVICE
            )

            logits = model(
                xb
            )

            pred = logits.argmax(
                dim=1
            ).cpu()

            y_pred.extend(
                pred.tolist()
            )

            y_true.extend(
                y[
                    start:
                    start + BATCH_SIZE
                ].tolist()
            )

    return (
        np.asarray(
            y_true,
            dtype=np.int64,
        ),
        np.asarray(
            y_pred,
            dtype=np.int64,
        ),
    )


def metrics_from_predictions(
    y_true,
    y_pred,
):

    return {
        "acc": accuracy_score(
            y_true,
            y_pred,
        ),

        "bAcc": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def evaluate(
    model,
    X,
    y,
):

    (
        y_true,
        y_pred,
    ) = predict_vqc(
        model,
        X,
        y,
    )

    return metrics_from_predictions(
        y_true,
        y_pred,
    )


# ============================================================
# Reward
# ============================================================

def predictive_score(
    metrics,
):

    return (
        0.7
        * metrics["bAcc"]
        + 0.3
        * metrics["F1"]
    )


def hardware_cost(
    architecture,
):

    stats = architecture_stats(
        architecture
    )

    # Four-layer ring circuit:
    # 4 layers x 8 CNOTs = 32.
    reference_n2 = (
        4
        * NUM_QUBITS
    )

    return (
        stats["n2"]
        / reference_n2
    )


def reward_value(
    metrics,
    architecture,
):

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

    val_metrics: Dict[
        str,
        float,
    ]

    state_dict: Dict[
        str,
        torch.Tensor,
    ]

    log_prob: Optional[
        torch.Tensor
    ] = None

    entropy: Optional[
        torch.Tensor
    ] = None


def evaluate_candidate(
    architecture,
    X_train,
    y_train,
    X_val,
    y_val,
):

    model = train_vqc(
        architecture,
        X_train,
        y_train,
        SEARCH_INNER_EPOCHS,
    )

    metrics = evaluate(
        model,
        X_val,
        y_val,
    )

    reward = reward_value(
        metrics,
        architecture,
    )

    return CandidateResult(
        architecture=architecture,
        reward=float(
            reward
        ),
        val_metrics=metrics,
        state_dict=copy.deepcopy(
            model.state_dict()
        ),
    )


# ============================================================
# Bi-loop architecture search
# ============================================================

def search_task(
    policy,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    ewc_state=None,
    reference_policy=None,
    use_ewc=True,
    use_kl=True,
):

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=POLICY_LR,
    )

    context = X_train[
        :min(
            192,
            len(
                X_train
            ),
        )
    ].to(
        DEVICE
    )

    baseline = None

    best_candidate = None

    for search_step in range(
        SEARCH_STEPS
    ):

        candidates = []

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

            candidates.append(
                result
            )

            if (
                best_candidate is None
                or result.reward
                > best_candidate.reward
            ):

                best_candidate = result

        rewards = np.asarray(
            [
                result.reward
                for result
                in candidates
            ],
            dtype=np.float32,
        )

        mean_reward = float(
            rewards.mean()
        )

        if baseline is None:

            baseline = (
                mean_reward
            )

        advantages = torch.tensor(
            rewards - baseline,
            dtype=REAL,
            device=DEVICE,
        )

        log_probs = torch.stack(
            [
                result.log_prob
                for result
                in candidates
            ]
        )

        entropy = torch.stack(
            [
                result.entropy
                for result
                in candidates
            ]
        ).mean()

        reinforce_loss = -torch.mean(
            advantages.detach()
            * log_probs
        )

        if (
            use_ewc
            and ewc_state is not None
        ):

            L_ewc = ewc_penalty(
                policy,
                ewc_state,
            )

        else:

            L_ewc = torch.tensor(
                0.0,
                device=DEVICE,
            )

        if (
            use_kl
            and reference_policy is not None
        ):

            L_kl = policy_kl(
                policy,
                reference_policy,
                context,
            )

        else:

            L_kl = torch.tensor(
                0.0,
                device=DEVICE,
            )

        policy_loss = (
            reinforce_loss
            + MU_EWC
            * L_ewc
            + ETA_KL
            * L_kl
            - ENTROPY_COEF
            * entropy
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            1.0,
        )

        optimizer.step()

        baseline = (
            0.9
            * baseline
            + 0.1
            * mean_reward
        )

    return (
        best_candidate,
        context,
    )


# ============================================================
# Final Table-1-consistent refit
# ============================================================

def final_refit_and_predict(
    candidate,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):

    """
    The architecture is selected using train/validation.

    The final VQC is then reinitialized and trained from scratch
    using train + validation = 90% of the available data.

    The test set remains untouched.
    """

    X_final = torch.cat(
        [
            X_train,
            X_val,
        ],
        dim=0,
    )

    y_final = torch.cat(
        [
            y_train,
            y_val,
        ],
        dim=0,
    )

    model = train_vqc(
        candidate.architecture,
        X_final,
        y_final,
        epochs=FINAL_REFIT_EPOCHS,
        initial_state=None,
    )

    (
        y_true,
        y_pred,
    ) = predict_vqc(
        model,
        X_test,
        y_test,
    )

    stats = architecture_stats(
        candidate.architecture
    )

    return {
        "y_true": y_true,
        "y_pred": y_pred,

        "reward": float(
            candidate.reward
        ),

        "depth": stats[
            "depth"
        ],

        "n2": stats[
            "n2"
        ],

        "patterns": "|".join(
            candidate
            .architecture
            .patterns
        ),
    }


# ============================================================
# Component ablation definitions
# ============================================================

ABLATION_CONFIGS = {

    "CL-QAS": {
        "use_ewc": True,
        "use_kl": True,
    },

    "CL-QAS-no-EWC": {
        "use_ewc": False,
        "use_kl": True,
    },

    "CL-QAS-no-KL": {
        "use_ewc": True,
        "use_kl": False,
    },

    "QAS-No-CL": {
        "use_ewc": False,
        "use_kl": False,
    },
}


# ============================================================
# Component ablation
# ============================================================

def run_component_ablation(
    tasks,
    rank=DEFAULT_TT_RANK,
):

    seed_rows = []

    task_rows = []

    for seed in SEEDS:

        split_cache = build_split_cache(
            tasks,
            seed,
        )

        for (
            method,
            config,
        ) in ABLATION_CONFIGS.items():

            set_seed(
                seed
            )

            policy = QASPolicy().to(
                DEVICE
            )

            ewc_state = None

            reference_policy = None

            pooled_true = []

            pooled_pred = []

            task_rewards = []

            task_n2 = []

            task_depth = []

            task_fidelity = []

            print(
                "\n"
                + "=" * 90
            )

            print(
                f"COMPONENT ABLATION | "
                f"seed={seed} | "
                f"method={method} | "
                f"TT rank={rank}"
            )

            print(
                "=" * 90
            )

            for task_id, (
                record,
                X_raw,
                y,
            ) in enumerate(
                tasks,
                start=1,
            ):

                data = prepare_task_table1(
                    X_raw,
                    y,
                    rank,
                    split_cache[
                        record
                    ],
                )

                candidate, context = search_task(
                    policy,

                    data[
                        "X_train"
                    ],
                    data[
                        "y_train"
                    ],

                    data[
                        "X_val"
                    ],
                    data[
                        "y_val"
                    ],

                    ewc_state=ewc_state,

                    reference_policy=(
                        reference_policy
                    ),

                    use_ewc=config[
                        "use_ewc"
                    ],

                    use_kl=config[
                        "use_kl"
                    ],
                )

                result = final_refit_and_predict(
                    candidate,

                    data[
                        "X_train"
                    ],
                    data[
                        "y_train"
                    ],

                    data[
                        "X_val"
                    ],
                    data[
                        "y_val"
                    ],

                    data[
                        "X_test"
                    ],
                    data[
                        "y_test"
                    ],
                )

                pooled_true.extend(
                    result[
                        "y_true"
                    ].tolist()
                )

                pooled_pred.extend(
                    result[
                        "y_pred"
                    ].tolist()
                )

                task_rewards.append(
                    result[
                        "reward"
                    ]
                )

                task_n2.append(
                    result[
                        "n2"
                    ]
                )

                task_depth.append(
                    result[
                        "depth"
                    ]
                )

                task_fidelity.append(
                    data[
                        "fidelity"
                    ]
                )

                task_metrics = (
                    metrics_from_predictions(
                        result[
                            "y_true"
                        ],
                        result[
                            "y_pred"
                        ],
                    )
                )

                print(
                    f"Task {task_id:02d} "
                    f"(record {record}) | "
                    f"Acc={task_metrics['acc']:.4f} | "
                    f"BAcc={task_metrics['bAcc']:.4f} | "
                    f"F1={task_metrics['F1']:.4f} | "
                    f"Fid={data['fidelity']:.4f} | "
                    f"#2Q={result['n2']:2d} | "
                    f"R={result['reward']:.4f}"
                )

                task_rows.append(
                    {
                        "experiment": (
                            "component_task"
                        ),

                        "seed": seed,

                        "method": method,

                        "rank": rank,

                        "task": task_id,

                        "record": record,

                        "acc": task_metrics[
                            "acc"
                        ],

                        "bAcc": task_metrics[
                            "bAcc"
                        ],

                        "F1": task_metrics[
                            "F1"
                        ],

                        "reward": result[
                            "reward"
                        ],

                        "n2": result[
                            "n2"
                        ],

                        "depth": result[
                            "depth"
                        ],

                        "patterns": result[
                            "patterns"
                        ],

                        "fidelity": data[
                            "fidelity"
                        ],
                    }
                )

                # --------------------------------------------
                # Continual-learning state update
                # --------------------------------------------

                if config[
                    "use_ewc"
                ]:

                    ewc_state = estimate_fisher(
                        policy,
                        context,
                    )

                if config[
                    "use_kl"
                ]:

                    reference_policy = (
                        copy.deepcopy(
                            policy
                        )
                        .eval()
                    )

                    for parameter in (
                        reference_policy
                        .parameters()
                    ):

                        parameter.requires_grad_(
                            False
                        )

            # ------------------------------------------------
            # Pooled ECG evaluation for this seed
            # ------------------------------------------------

            pooled_true = np.asarray(
                pooled_true,
                dtype=np.int64,
            )

            pooled_pred = np.asarray(
                pooled_pred,
                dtype=np.int64,
            )

            metrics = metrics_from_predictions(
                pooled_true,
                pooled_pred,
            )

            row = {
                "experiment": (
                    "component"
                ),

                "seed": seed,

                "method": method,

                "rank": rank,

                "acc": metrics[
                    "acc"
                ],

                "bAcc": metrics[
                    "bAcc"
                ],

                "F1": metrics[
                    "F1"
                ],

                "reward": float(
                    np.mean(
                        task_rewards
                    )
                ),

                "n2": float(
                    np.mean(
                        task_n2
                    )
                ),

                "depth": float(
                    np.mean(
                        task_depth
                    )
                ),

                "fidelity": float(
                    np.mean(
                        task_fidelity
                    )
                ),

                "n_test": int(
                    len(
                        pooled_true
                    )
                ),
            }

            seed_rows.append(
                row
            )

            print(
                "-" * 90
            )

            print(
                f"POOLED ECG | "
                f"{method} | "
                f"seed={seed} | "
                f"Acc={metrics['acc']:.4f} | "
                f"BAcc={metrics['bAcc']:.4f} | "
                f"F1={metrics['F1']:.4f} | "
                f"#2Q={np.mean(task_n2):.2f} | "
                f"Fid={np.mean(task_fidelity):.4f}"
            )

    return (
        seed_rows,
        task_rows,
    )


# ============================================================
# TT-rank ablation
# ============================================================

def run_rank_ablation(
    tasks,
):

    seed_rows = []

    task_rows = []

    for seed in SEEDS:

        split_cache = build_split_cache(
            tasks,
            seed,
        )

        for rank in TT_RANKS_TO_TEST:

            set_seed(
                seed
            )

            policy = QASPolicy().to(
                DEVICE
            )

            ewc_state = None

            reference_policy = None

            pooled_true = []

            pooled_pred = []

            task_rewards = []

            task_n2 = []

            task_depth = []

            task_fidelity = []

            print(
                "\n"
                + "=" * 90
            )

            print(
                f"TT-RANK ABLATION | "
                f"seed={seed} | "
                f"rank={rank}"
            )

            print(
                "=" * 90
            )

            for task_id, (
                record,
                X_raw,
                y,
            ) in enumerate(
                tasks,
                start=1,
            ):

                data = prepare_task_table1(
                    X_raw,
                    y,
                    rank,
                    split_cache[
                        record
                    ],
                )

                candidate, context = search_task(
                    policy,

                    data[
                        "X_train"
                    ],
                    data[
                        "y_train"
                    ],

                    data[
                        "X_val"
                    ],
                    data[
                        "y_val"
                    ],

                    ewc_state=ewc_state,

                    reference_policy=(
                        reference_policy
                    ),

                    use_ewc=True,

                    use_kl=True,
                )

                result = final_refit_and_predict(
                    candidate,

                    data[
                        "X_train"
                    ],
                    data[
                        "y_train"
                    ],

                    data[
                        "X_val"
                    ],
                    data[
                        "y_val"
                    ],

                    data[
                        "X_test"
                    ],
                    data[
                        "y_test"
                    ],
                )

                pooled_true.extend(
                    result[
                        "y_true"
                    ].tolist()
                )

                pooled_pred.extend(
                    result[
                        "y_pred"
                    ].tolist()
                )

                task_rewards.append(
                    result[
                        "reward"
                    ]
                )

                task_n2.append(
                    result[
                        "n2"
                    ]
                )

                task_depth.append(
                    result[
                        "depth"
                    ]
                )

                task_fidelity.append(
                    data[
                        "fidelity"
                    ]
                )

                task_metrics = (
                    metrics_from_predictions(
                        result[
                            "y_true"
                        ],
                        result[
                            "y_pred"
                        ],
                    )
                )

                print(
                    f"Task {task_id:02d} "
                    f"(record {record}) | "
                    f"rank={rank} | "
                    f"Acc={task_metrics['acc']:.4f} | "
                    f"BAcc={task_metrics['bAcc']:.4f} | "
                    f"F1={task_metrics['F1']:.4f} | "
                    f"Fid={data['fidelity']:.4f} | "
                    f"#2Q={result['n2']:2d}"
                )

                task_rows.append(
                    {
                        "experiment": (
                            "rank_task"
                        ),

                        "seed": seed,

                        "method": (
                            "CL-QAS"
                        ),

                        "rank": rank,

                        "task": task_id,

                        "record": record,

                        "acc": task_metrics[
                            "acc"
                        ],

                        "bAcc": task_metrics[
                            "bAcc"
                        ],

                        "F1": task_metrics[
                            "F1"
                        ],

                        "reward": result[
                            "reward"
                        ],

                        "n2": result[
                            "n2"
                        ],

                        "depth": result[
                            "depth"
                        ],

                        "patterns": result[
                            "patterns"
                        ],

                        "fidelity": data[
                            "fidelity"
                        ],
                    }
                )

                # Full CL-QAS for every rank.
                ewc_state = estimate_fisher(
                    policy,
                    context,
                )

                reference_policy = (
                    copy.deepcopy(
                        policy
                    )
                    .eval()
                )

                for parameter in (
                    reference_policy
                    .parameters()
                ):

                    parameter.requires_grad_(
                        False
                    )

            pooled_true = np.asarray(
                pooled_true,
                dtype=np.int64,
            )

            pooled_pred = np.asarray(
                pooled_pred,
                dtype=np.int64,
            )

            metrics = metrics_from_predictions(
                pooled_true,
                pooled_pred,
            )

            row = {
                "experiment": (
                    "rank"
                ),

                "seed": seed,

                "method": (
                    "CL-QAS"
                ),

                "rank": rank,

                "acc": metrics[
                    "acc"
                ],

                "bAcc": metrics[
                    "bAcc"
                ],

                "F1": metrics[
                    "F1"
                ],

                "reward": float(
                    np.mean(
                        task_rewards
                    )
                ),

                "n2": float(
                    np.mean(
                        task_n2
                    )
                ),

                "depth": float(
                    np.mean(
                        task_depth
                    )
                ),

                "fidelity": float(
                    np.mean(
                        task_fidelity
                    )
                ),

                "n_test": int(
                    len(
                        pooled_true
                    )
                ),
            }

            seed_rows.append(
                row
            )

            print(
                "-" * 90
            )

            print(
                f"POOLED ECG | "
                f"rank={rank} | "
                f"seed={seed} | "
                f"Acc={metrics['acc']:.4f} | "
                f"BAcc={metrics['bAcc']:.4f} | "
                f"F1={metrics['F1']:.4f} | "
                f"Fid={np.mean(task_fidelity):.4f} | "
                f"#2Q={np.mean(task_n2):.2f}"
            )

    return (
        seed_rows,
        task_rows,
    )


# ============================================================
# Summary tables
# ============================================================

def summarize_component_ablation(
    df,
):

    component_df = df[
        df[
            "experiment"
        ] == "component"
    ].copy()

    summary = (
        component_df
        .groupby(
            "method"
        )[
            [
                "acc",
                "bAcc",
                "F1",
                "reward",
                "n2",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "COMPONENT ABLATION SUMMARY "
        "(POOLED ECG RESULTS; "
        "MEAN +/- STD ACROSS SEEDS)"
    )

    print(
        "=" * 110
    )

    print(
        summary.to_string()
    )

    return summary


def summarize_rank_ablation(
    df,
):

    rank_df = df[
        df[
            "experiment"
        ] == "rank"
    ].copy()

    summary = (
        rank_df
        .groupby(
            "rank"
        )[
            [
                "fidelity",
                "acc",
                "bAcc",
                "F1",
                "reward",
                "n2",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "TT-RANK ABLATION SUMMARY "
        "(POOLED ECG RESULTS; "
        "MEAN +/- STD ACROSS SEEDS)"
    )

    print(
        "=" * 110
    )

    print(
        summary.to_string()
    )

    return summary


# ============================================================
# Publication-friendly flattened summaries
# ============================================================

def flatten_summary_columns(
    summary_df,
):

    output = (
        summary_df
        .copy()
    )

    output.columns = [
        f"{metric}_{stat}"
        for (
            metric,
            stat,
        ) in output.columns
    ]

    return output.reset_index()


# ============================================================
# Save outputs
# ============================================================

def save_outputs(
    summary_rows,
    task_rows,
):

    summary_df = pd.DataFrame(
        summary_rows
    )

    task_df = pd.DataFrame(
        task_rows
    )

    component_summary = (
        summarize_component_ablation(
            summary_df
        )
    )

    rank_summary = (
        summarize_rank_ablation(
            summary_df
        )
    )

    flat_component = (
        flatten_summary_columns(
            component_summary
        )
    )

    flat_rank = (
        flatten_summary_columns(
            rank_summary
        )
    )

    # Seed-level manuscript results.
    summary_df.to_csv(
        "ecg_clqas_ablation_seed_results.csv",
        index=False,
    )

    # Detailed record/task diagnostics.
    task_df.to_csv(
        "ecg_clqas_ablation_task_diagnostics.csv",
        index=False,
    )

    # Multi-index summaries.
    component_summary.to_csv(
        "ecg_component_ablation_summary.csv"
    )

    rank_summary.to_csv(
        "ecg_tt_rank_ablation_summary.csv"
    )

    # Flattened versions, easier to use in LaTeX/Origin/Excel.
    flat_component.to_csv(
        "ecg_component_ablation_summary_flat.csv",
        index=False,
    )

    flat_rank.to_csv(
        "ecg_tt_rank_ablation_summary_flat.csv",
        index=False,
    )

    print(
        "\nSaved output files:"
    )

    print(
        "  ecg_clqas_ablation_seed_results.csv"
    )

    print(
        "  ecg_clqas_ablation_task_diagnostics.csv"
    )

    print(
        "  ecg_component_ablation_summary.csv"
    )

    print(
        "  ecg_tt_rank_ablation_summary.csv"
    )

    print(
        "  ecg_component_ablation_summary_flat.csv"
    )

    print(
        "  ecg_tt_rank_ablation_summary_flat.csv"
    )


# ============================================================
# Optional consistency check
# ============================================================

def consistency_check(
    summary_df,
):

    """
    Reports whether the nominal CL-QAS component result and the
    rank-3 CL-QAS result are statistically close.

    They use the same nominal algorithm and should therefore
    be reasonably consistent.
    """

    component = summary_df[
        (
            summary_df[
                "experiment"
            ] == "component"
        )
        &
        (
            summary_df[
                "method"
            ] == "CL-QAS"
        )
        &
        (
            summary_df[
                "rank"
            ] == DEFAULT_TT_RANK
        )
    ]

    rank3 = summary_df[
        (
            summary_df[
                "experiment"
            ] == "rank"
        )
        &
        (
            summary_df[
                "rank"
            ] == DEFAULT_TT_RANK
        )
    ]

    if (
        len(component) == 0
        or len(rank3) == 0
    ):
        return

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CONSISTENCY CHECK: "
        "COMPONENT CL-QAS vs TT-RANK r=3"
    )

    print(
        "=" * 90
    )

    for metric in (
        "acc",
        "bAcc",
        "F1",
        "n2",
        "fidelity",
    ):

        c_mean = component[
            metric
        ].mean()

        r_mean = rank3[
            metric
        ].mean()

        difference = (
            c_mean
            - r_mean
        )

        print(
            f"{metric:10s}: "
            f"component={c_mean:.6f} | "
            f"rank3={r_mean:.6f} | "
            f"diff={difference:+.6f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 90
    )

    print(
        "CL-QAS ECG COMPONENT AND TT-RANK ABLATIONS"
    )

    print(
        "=" * 90
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Nominal TT rank: "
        f"{DEFAULT_TT_RANK}"
    )

    print(
        f"TT rank study: "
        f"{TT_RANKS_TO_TEST}"
    )

    print(
        f"Search split: "
        f"{SEARCH_TRAIN_FRAC:.0%} train / "
        f"{SEARCH_VAL_FRAC:.0%} validation / "
        f"{1.0 - SEARCH_TRAIN_FRAC - SEARCH_VAL_FRAC:.0%} test"
    )

    print(
        "Final VQC refit: "
        "train + validation = 90%, "
        "test = 10%"
    )

    # --------------------------------------------------------
    # Load ECG tasks once
    # --------------------------------------------------------

    tasks = load_tasks()

    print(
        "\nSequential ECG tasks:"
    )

    print(
        [
            record
            for (
                record,
                _,
                _,
            ) in tasks
        ]
    )

    # --------------------------------------------------------
    # Component ablation
    # --------------------------------------------------------

    (
        component_seed_rows,
        component_task_rows,
    ) = run_component_ablation(
        tasks,
        rank=DEFAULT_TT_RANK,
    )

    # --------------------------------------------------------
    # TT-rank ablation
    # --------------------------------------------------------

    (
        rank_seed_rows,
        rank_task_rows,
    ) = run_rank_ablation(
        tasks
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    summary_rows = (
        component_seed_rows
        + rank_seed_rows
    )

    task_rows = (
        component_task_rows
        + rank_task_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    consistency_check(
        summary_df
    )

    # --------------------------------------------------------
    # Save and print results
    # --------------------------------------------------------

    save_outputs(
        summary_rows,
        task_rows,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
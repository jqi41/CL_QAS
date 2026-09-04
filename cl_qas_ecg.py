#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clqas_ecg_table1_aligned_with_table3.py

Main ECG experiment for Table 1, aligned exactly with the experimental
protocol used for the revised Table 3 ablation study.

Compared methods
----------------
1. Naive-VQC
2. QAS-No-CL
3. CL-QAS

Alignment with Table 3
----------------------
For a given random seed, all methods use exactly the same:

    - MIT-BIH records
    - ECG preprocessing
    - train / validation / test samples
    - train-only standardization
    - TT rank r = 3
    - TT tensorization (4, 16, 4)
    - amplitude encoding
    - VQC implementation
    - architecture search space
    - search budget
    - VQC optimization budget
    - hardware-aware reward
    - final 90/10 evaluation protocol
    - pooled ECG metric computation

The only methodological differences are:

    Naive-VQC:
        fixed depth-4 ring architecture

    QAS-No-CL:
        architecture search without EWC/KL

    CL-QAS:
        architecture search with EWC + KL policy preservation

Important statistical protocol
------------------------------
For each seed:

    1. Each MIT-BIH record is treated as one sequential task.
    2. Each task is split into:
           80% search-training
           10% architecture-validation
           10% untouched testing
    3. QAS uses only the 80/10 search split.
    4. After architecture selection, the selected architecture is
       reinitialized and trained from scratch on train + validation = 90%.
    5. Predictions from all eight held-out ECG record test sets are pooled.
    6. Accuracy, balanced accuracy, and F1 are computed once per seed.
    7. Table 1 reports mean +/- standard deviation across seeds.

This is the same evaluation logic used by the revised Table 3 code.

Dependencies
------------
pip install torch numpy pandas scikit-learn wfdb certifi
"""

# ============================================================
# Imports
# ============================================================

import os
import time
import copy
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


# ============================================================
# WFDB / SSL support
# ============================================================

try:
    import certifi

    os.environ.setdefault(
        "SSL_CERT_FILE",
        certifi.where(),
    )

    os.environ.setdefault(
        "REQUESTS_CA_BUNDLE",
        certifi.where(),
    )

except ImportError:
    pass


try:
    import wfdb

except ImportError as exc:

    raise ImportError(
        "Please install dependencies with:\n"
        "pip install wfdb certifi"
    ) from exc


# ============================================================
# 1. Configuration
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

REAL = torch.float32

COMPLEX = torch.complex64


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

MITDB_VERSION = "1.0.0"

MITDB_LOCAL_DIR = os.path.expanduser(
    "~/Documents/Projects/qnn/data/mitdb"
)

ECG_RECORDS = (
    105,
    106,
    109,
    114,
    116,
    119,
    200,
    201,
)

MAX_BEATS_PER_RECORD = 800

ECG_WINDOW_SEC = 0.6

INPUT_DIM = 256


# ------------------------------------------------------------
# AAMI binary classes
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Quantum representation
# ------------------------------------------------------------

NUM_QUBITS = 8

NUM_CLASSES = 2

TT_MODES = (
    4,
    16,
    4,
)

TT_RANK = 3


# ------------------------------------------------------------
# Architecture search
#
# SAME as Table 3
# ------------------------------------------------------------

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


SEARCH_STEPS = 10

CANDIDATES_PER_STEP = 3

SEARCH_INNER_EPOCHS = 8


# ------------------------------------------------------------
# Final refit
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
# Table-3-consistent data split
# ------------------------------------------------------------

SEARCH_TRAIN_FRAC = 0.80

SEARCH_VAL_FRAC = 0.10

# Remaining 10% = test

SPLIT_SEED_BASE = 5000


# ------------------------------------------------------------
# SAME seeds as Table 3
# ------------------------------------------------------------

SEEDS = (
    11,
    22,
    33,
)


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# 3. Robust MIT-BIH loading
# ============================================================

def read_mitdb_record(
    record,
    max_retries=5,
    retry_wait=3.0,
):

    record = str(
        record
    )

    # --------------------------------------------------------
    # Local database first
    # --------------------------------------------------------

    if os.path.isdir(
        MITDB_LOCAL_DIR
    ):

        record_base = os.path.join(
            MITDB_LOCAL_DIR,
            record,
        )

        required_files = (
            record_base + ".hea",
            record_base + ".dat",
            record_base + ".atr",
        )

        if all(
            os.path.exists(path)
            for path in required_files
        ):

            try:

                signal, fields = wfdb.rdsamp(
                    record_base
                )

                annotation = wfdb.rdann(
                    record_base,
                    "atr",
                )

                print(
                    f"[MIT-BIH] Record {record} "
                    f"loaded locally."
                )

                return (
                    signal,
                    fields,
                    annotation,
                )

            except Exception as exc:

                print(
                    f"[MIT-BIH] Local read failed "
                    f"for {record}: {exc}"
                )


    # --------------------------------------------------------
    # PhysioNet fallback
    # --------------------------------------------------------

    pn_dir = (
        f"mitdb/{MITDB_VERSION}"
    )

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            print(
                f"[MIT-BIH] Fetching {record} "
                f"from PhysioNet "
                f"({attempt}/{max_retries})"
            )

            signal, fields = wfdb.rdsamp(
                record,
                pn_dir=pn_dir,
            )

            annotation = wfdb.rdann(
                record,
                "atr",
                pn_dir=pn_dir,
            )

            return (
                signal,
                fields,
                annotation,
            )

        except Exception as exc:

            last_error = exc

            print(
                f"    failed: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < max_retries:

                time.sleep(
                    retry_wait
                    * attempt
                )

    raise RuntimeError(
        f"Unable to load MIT-BIH "
        f"record {record}.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# 4. ECG preprocessing
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

    names = [
        name.upper()
        for name in signal_names
    ]

    if "MLII" in names:

        return names.index(
            "MLII"
        )

    return 0


def fft_bandpass(
    signal: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 40.0,
) -> np.ndarray:

    n = len(
        signal
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / fs,
    )

    spectrum = np.fft.rfft(
        signal
    )

    mask = (
        (frequencies >= low)
        &
        (frequencies <= high)
    )

    spectrum *= mask

    output = np.fft.irfft(
        spectrum,
        n=n,
    )

    return output.astype(
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

    x = np.interp(
        target_grid,
        source_grid,
        x,
    ).astype(
        np.float32
    )

    # Per-beat normalization.
    mean = float(
        x.mean()
    )

    std = float(
        x.std()
        + 1e-6
    )

    x = (
        x - mean
    ) / std

    x = np.clip(
        x,
        -5.0,
        5.0,
    )

    return x.astype(
        np.float32
    )


def load_record(
    record,
    max_beats=MAX_BEATS_PER_RECORD,
    window_sec=ECG_WINDOW_SEC,
    min_per_class=10,
):

    (
        signal,
        fields,
        annotation,
    ) = read_mitdb_record(
        record
    )

    channel = choose_channel(
        fields["sig_name"]
    )

    fs = float(
        fields["fs"]
    )

    x = signal[
        :,
        channel
    ]

    x = fft_bandpass(
        x,
        fs,
    )

    half_window = int(
        window_sec
        * fs
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

        X.append(
            beat_vector_256(
                x[
                    start:end
                ]
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
            f"No usable beats "
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
# 5. EXACT SAME split protocol as Table 3
# ============================================================

def stratified_split_table3(
    y: torch.Tensor,
    train_frac: float = SEARCH_TRAIN_FRAC,
    val_frac: float = SEARCH_VAL_FRAC,
    seed: int = 1234,
):

    rng = np.random.RandomState(
        seed
    )

    labels = y.cpu().numpy()

    train_indices = []

    val_indices = []

    test_indices = []

    for class_id in (
        0,
        1,
    ):

        indices = np.where(
            labels == class_id
        )[0].copy()

        rng.shuffle(
            indices
        )

        n = len(
            indices
        )

        n_train = int(
            np.floor(
                train_frac
                * n
            )
        )

        n_val = int(
            np.floor(
                val_frac
                * n
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


def build_split_cache(
    tasks,
    seed,
):

    """
    One split per record per seed.

    Critically, Naive-VQC, QAS-No-CL and CL-QAS use the
    EXACT same samples.
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
        ) = stratified_split_table3(
            y,
            seed=split_seed,
        )

        cache[
            record
        ] = {
            "train_idx":
            train_idx,

            "val_idx":
            val_idx,

            "test_idx":
            test_idx,
        }

    return cache


# ============================================================
# 6. Train-only standardization
# ============================================================

@dataclass
class StandardizationState:

    mean: torch.Tensor

    std: torch.Tensor


def fit_standardizer(
    X_train,
):

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
    X,
    state,
):

    return (
        X
        - state.mean
    ) / state.std


# ============================================================
# 7. TT-SVD
# ============================================================

def normalize_state(
    x,
    eps=1e-8,
):

    return x / (
        torch.linalg.vector_norm(
            x
        )
        + eps
    )


def tt_svd_vector(
    x,
    modes=TT_MODES,
    max_rank=TT_RANK,
):

    if (
        int(
            np.prod(
                modes
            )
        )
        != x.numel()
    ):

        raise ValueError(
            "TT modes do not match "
            "vector dimension."
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
            rank_prev
            * n_k,
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
            :rank
        ]

        S = S[
            :rank
        ]

        Vh = Vh[
            :rank,
            :
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

    output = cores[
        0
    ]

    for core in cores[
        1:
    ]:

        output = torch.einsum(
            "...a,aib->...ib",
            output,
            core,
        )

    return (
        output
        .squeeze(0)
        .squeeze(-1)
        .reshape(-1)
    )


def tt_encode_dataset(
    X,
    rank=TT_RANK,
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
# 8. Task preparation
#
# EXACT SAME pipeline as Table 3
# ============================================================

def prepare_task(
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


    # --------------------------------------------------------
    # Training statistics only
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # TT representation after standardization
    # --------------------------------------------------------

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
        "X_train":
        X_train,

        "y_train":
        y_train,

        "X_val":
        X_val,

        "y_val":
        y_val,

        "X_test":
        X_test,

        "y_test":
        y_test,

        "fidelity":
        float(
            fidelity.mean()
            .item()
        ),
    }


# ============================================================
# 9. Quantum gates
# ============================================================

def rx(theta):

    c = torch.cos(
        theta / 2
    )

    s = torch.sin(
        theta / 2
    )

    return torch.stack([
        torch.stack([
            c,
            -1j * s,
        ]),
        torch.stack([
            -1j * s,
            c,
        ]),
    ]).to(
        COMPLEX
    )


def ry(theta):

    c = torch.cos(
        theta / 2
    )

    s = torch.sin(
        theta / 2
    )

    return torch.stack([
        torch.stack([
            c,
            -s,
        ]),
        torch.stack([
            s,
            c,
        ]),
    ]).to(
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

    return torch.stack([
        torch.stack([
            a,
            zero,
        ]),
        torch.stack([
            zero,
            b,
        ]),
    ]).to(
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
        +
        [
            i
            for i in range(
                1,
                NUM_QUBITS + 1,
            )
            if i != axis
        ]
        +
        [axis]
    )

    psi = (
        psi
        .permute(
            *permutation
        )
        .contiguous()
    )

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

    psi = (
        psi
        .permute(
            *inverse
        )
        .contiguous()
    )

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
        indices
        ^ target_mask,
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

        outputs.append(
            torch.sum(
                probabilities
                * sign.unsqueeze(0),
                dim=1,
            )
        )

    return torch.stack(
        outputs,
        dim=1,
    )


# ============================================================
# 10. Circuit architecture
# ============================================================

@dataclass
class Architecture:

    depth: int

    patterns: Tuple[
        str,
        ...
    ]


def edges_for_pattern(
    pattern,
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
    architecture,
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
        for pattern
        in architecture.patterns
    )

    return {
        "depth":
        architecture.depth,

        "n1":
        n1,

        "n2":
        n2,
    }


def naive_architecture():

    return Architecture(
        depth=4,
        patterns=(
            "ring",
            "ring",
            "ring",
            "ring",
        ),
    )


# ============================================================
# 11. VQC
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

        outputs = z_expectations(
            state
        )

        return outputs[
            :,
            :NUM_CLASSES,
        ]


# ============================================================
# 12. QAS policy
#
# IMPORTANT:
# no ring/depth initialization bias;
# identical to Table 3.
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


# ============================================================
# 13. Policy sampling
# ============================================================

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
# 14. EWC
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
        name:
        torch.zeros_like(
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
                    parameter.grad
                    .detach()
                    ** 2
                ) / num_samples

    means = {
        name:
        parameter
        .detach()
        .clone()

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

        penalty += torch.sum(
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

    return (
        0.5
        * penalty
    )


# ============================================================
# 15. KL regularization
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
# 16. VQC training
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

    # SAME as Table 3.
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
# 17. Predictions and metrics
# ============================================================

def predict_vqc(
    model,
    X,
    y,
):

    model.eval()

    truth = []

    predictions = []

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

            pred = (
                logits
                .argmax(
                    dim=1
                )
                .cpu()
            )

            predictions.extend(
                pred.tolist()
            )

            truth.extend(
                y[
                    start:
                    start + BATCH_SIZE
                ].tolist()
            )

    return (
        np.asarray(
            truth,
            dtype=np.int64,
        ),
        np.asarray(
            predictions,
            dtype=np.int64,
        ),
    )


def metrics_from_predictions(
    truth,
    predictions,
):

    return {
        "acc":
        accuracy_score(
            truth,
            predictions,
        ),

        "bAcc":
        balanced_accuracy_score(
            truth,
            predictions,
        ),

        "F1":
        f1_score(
            truth,
            predictions,
            zero_division=0,
        ),
    }


def evaluate(
    model,
    X,
    y,
):

    truth, predictions = predict_vqc(
        model,
        X,
        y,
    )

    return metrics_from_predictions(
        truth,
        predictions,
    )


# ============================================================
# 18. Reward
# ============================================================

def predictive_score(
    metrics,
):

    return (
        0.70
        * metrics[
            "bAcc"
        ]
        +
        0.30
        * metrics[
            "F1"
        ]
    )


def hardware_cost(
    architecture,
):

    statistics = architecture_stats(
        architecture
    )

    reference_n2 = (
        4
        * NUM_QUBITS
    )

    return (
        statistics[
            "n2"
        ]
        / reference_n2
    )


def architecture_reward(
    metrics,
    architecture,
):

    return (
        predictive_score(
            metrics
        )
        -
        LAMBDA_HW
        * hardware_cost(
            architecture
        )
    )


# ============================================================
# 19. Candidate evaluation
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

    reward = architecture_reward(
        metrics,
        architecture,
    )

    return CandidateResult(
        architecture=
        architecture,

        reward=
        float(
            reward
        ),

        val_metrics=
        metrics,

        state_dict=
        copy.deepcopy(
            model.state_dict()
        ),
    )


# ============================================================
# 20. Bi-loop QAS
#
# SAME as Table 3:
# - no ring anchor
# - no greedy architecture post-selection
# - no policy bias
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

    policy_optimizer = torch.optim.Adam(
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

    for _ in range(
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

                best_candidate = (
                    result
                )

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
            rewards
            - baseline,
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
            and
            ewc_state is not None
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
            and
            reference_policy is not None
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
            +
            MU_EWC
            * L_ewc
            +
            ETA_KL
            * L_kl
            -
            ENTROPY_COEF
            * entropy
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

        baseline = (
            0.9
            * baseline
            +
            0.1
            * mean_reward
        )

    return (
        best_candidate,
        context,
    )


# ============================================================
# 21. Final refit
#
# SAME as Table 3:
# architecture selected on 80/10;
# fresh VQC trained on combined 90%.
# ============================================================

def final_refit_and_predict(
    architecture,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):

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
        architecture,
        X_final,
        y_final,
        epochs=
        FINAL_REFIT_EPOCHS,
        initial_state=None,
    )

    truth, predictions = predict_vqc(
        model,
        X_test,
        y_test,
    )

    statistics = architecture_stats(
        architecture
    )

    return {
        "y_true":
        truth,

        "y_pred":
        predictions,

        "depth":
        statistics[
            "depth"
        ],

        "n1":
        statistics[
            "n1"
        ],

        "n2":
        statistics[
            "n2"
        ],

        "patterns":
        "|".join(
            architecture.patterns
        ),
    }


# ============================================================
# 22. One complete seed
# ============================================================

def run_seed(
    tasks,
    seed,
):

    # --------------------------------------------------------
    # SAME raw partitions for all three methods
    # --------------------------------------------------------

    split_cache = build_split_cache(
        tasks,
        seed,
    )


    # --------------------------------------------------------
    # Separate policies
    # --------------------------------------------------------

    set_seed(
        seed
    )

    policy_nocl = QASPolicy().to(
        DEVICE
    )


    set_seed(
        seed
    )

    policy_cl = QASPolicy().to(
        DEVICE
    )


    ewc_state = None

    reference_policy = None


    # --------------------------------------------------------
    # Pooled predictions
    # --------------------------------------------------------

    pooled = {

        "Naive-VQC": {
            "true": [],
            "pred": [],
            "n2": [],
            "depth": [],
            "fidelity": [],
            "reward": [],
        },

        "QAS-No-CL": {
            "true": [],
            "pred": [],
            "n2": [],
            "depth": [],
            "fidelity": [],
            "reward": [],
        },

        "CL-QAS": {
            "true": [],
            "pred": [],
            "n2": [],
            "depth": [],
            "fidelity": [],
            "reward": [],
        },
    }


    task_rows = []


    # ========================================================
    # Sequential tasks
    # ========================================================

    for task_id, (
        record,
        X_raw,
        y,
    ) in enumerate(
        tasks,
        start=1,
    ):

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"Seed {seed} | "
            f"Task {task_id} | "
            f"MIT-BIH record {record}"
        )

        print(
            "=" * 90
        )


        # ----------------------------------------------------
        # ONE prepared dataset shared by all methods
        # ----------------------------------------------------

        data = prepare_task(
            X_raw,
            y,
            rank=TT_RANK,
            split_info=
            split_cache[
                record
            ],
        )

        print(
            f"TT rank={TT_RANK} | "
            f"fidelity="
            f"{data['fidelity']:.6f}"
        )


        # ====================================================
        # A. Naive-VQC
        # ====================================================

        print(
            "\n[Naive-VQC]"
        )

        naive_arch = naive_architecture()

        naive_result = final_refit_and_predict(
            naive_arch,

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

        naive_metrics = metrics_from_predictions(
            naive_result[
                "y_true"
            ],
            naive_result[
                "y_pred"
            ],
        )

        pooled[
            "Naive-VQC"
        ][
            "true"
        ].extend(
            naive_result[
                "y_true"
            ].tolist()
        )

        pooled[
            "Naive-VQC"
        ][
            "pred"
        ].extend(
            naive_result[
                "y_pred"
            ].tolist()
        )

        pooled[
            "Naive-VQC"
        ][
            "n2"
        ].append(
            naive_result[
                "n2"
            ]
        )

        pooled[
            "Naive-VQC"
        ][
            "depth"
        ].append(
            naive_result[
                "depth"
            ]
        )

        pooled[
            "Naive-VQC"
        ][
            "fidelity"
        ].append(
            data[
                "fidelity"
            ]
        )

        task_rows.append({

            "seed":
            seed,

            "task":
            task_id,

            "record":
            record,

            "method":
            "Naive-VQC",

            **naive_metrics,

            "reward":
            np.nan,

            "depth":
            naive_result[
                "depth"
            ],

            "n1":
            naive_result[
                "n1"
            ],

            "n2":
            naive_result[
                "n2"
            ],

            "patterns":
            naive_result[
                "patterns"
            ],

            "tt_fidelity":
            data[
                "fidelity"
            ],
        })

        print(
            f"Test | "
            f"Acc={naive_metrics['acc']:.4f} | "
            f"BAcc={naive_metrics['bAcc']:.4f} | "
            f"F1={naive_metrics['F1']:.4f} | "
            f"#2Q={naive_result['n2']}"
        )


        # ====================================================
        # B. QAS-No-CL
        # ====================================================

        print(
            "\n[QAS-No-CL]"
        )

        qas_candidate, _ = search_task(

            policy_nocl,

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

            ewc_state=None,

            reference_policy=None,

            use_ewc=False,

            use_kl=False,
        )

        qas_result = final_refit_and_predict(

            qas_candidate.architecture,

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

        qas_metrics = metrics_from_predictions(
            qas_result[
                "y_true"
            ],
            qas_result[
                "y_pred"
            ],
        )

        pooled[
            "QAS-No-CL"
        ][
            "true"
        ].extend(
            qas_result[
                "y_true"
            ].tolist()
        )

        pooled[
            "QAS-No-CL"
        ][
            "pred"
        ].extend(
            qas_result[
                "y_pred"
            ].tolist()
        )

        pooled[
            "QAS-No-CL"
        ][
            "n2"
        ].append(
            qas_result[
                "n2"
            ]
        )

        pooled[
            "QAS-No-CL"
        ][
            "depth"
        ].append(
            qas_result[
                "depth"
            ]
        )

        pooled[
            "QAS-No-CL"
        ][
            "fidelity"
        ].append(
            data[
                "fidelity"
            ]
        )

        pooled[
            "QAS-No-CL"
        ][
            "reward"
        ].append(
            qas_candidate.reward
        )

        task_rows.append({

            "seed":
            seed,

            "task":
            task_id,

            "record":
            record,

            "method":
            "QAS-No-CL",

            **qas_metrics,

            "reward":
            qas_candidate.reward,

            "depth":
            qas_result[
                "depth"
            ],

            "n1":
            qas_result[
                "n1"
            ],

            "n2":
            qas_result[
                "n2"
            ],

            "patterns":
            qas_result[
                "patterns"
            ],

            "tt_fidelity":
            data[
                "fidelity"
            ],
        })

        print(
            f"Test | "
            f"Acc={qas_metrics['acc']:.4f} | "
            f"BAcc={qas_metrics['bAcc']:.4f} | "
            f"F1={qas_metrics['F1']:.4f} | "
            f"#2Q={qas_result['n2']} | "
            f"R={qas_candidate.reward:.4f}"
        )


        # ====================================================
        # C. CL-QAS
        # ====================================================

        print(
            "\n[CL-QAS]"
        )

        cl_candidate, context = search_task(

            policy_cl,

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

            ewc_state=
            ewc_state,

            reference_policy=
            reference_policy,

            use_ewc=True,

            use_kl=True,
        )

        cl_result = final_refit_and_predict(

            cl_candidate.architecture,

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

        cl_metrics = metrics_from_predictions(
            cl_result[
                "y_true"
            ],
            cl_result[
                "y_pred"
            ],
        )

        pooled[
            "CL-QAS"
        ][
            "true"
        ].extend(
            cl_result[
                "y_true"
            ].tolist()
        )

        pooled[
            "CL-QAS"
        ][
            "pred"
        ].extend(
            cl_result[
                "y_pred"
            ].tolist()
        )

        pooled[
            "CL-QAS"
        ][
            "n2"
        ].append(
            cl_result[
                "n2"
            ]
        )

        pooled[
            "CL-QAS"
        ][
            "depth"
        ].append(
            cl_result[
                "depth"
            ]
        )

        pooled[
            "CL-QAS"
        ][
            "fidelity"
        ].append(
            data[
                "fidelity"
            ]
        )

        pooled[
            "CL-QAS"
        ][
            "reward"
        ].append(
            cl_candidate.reward
        )

        task_rows.append({

            "seed":
            seed,

            "task":
            task_id,

            "record":
            record,

            "method":
            "CL-QAS",

            **cl_metrics,

            "reward":
            cl_candidate.reward,

            "depth":
            cl_result[
                "depth"
            ],

            "n1":
            cl_result[
                "n1"
            ],

            "n2":
            cl_result[
                "n2"
            ],

            "patterns":
            cl_result[
                "patterns"
            ],

            "tt_fidelity":
            data[
                "fidelity"
            ],
        })

        print(
            f"Test | "
            f"Acc={cl_metrics['acc']:.4f} | "
            f"BAcc={cl_metrics['bAcc']:.4f} | "
            f"F1={cl_metrics['F1']:.4f} | "
            f"#2Q={cl_result['n2']} | "
            f"R={cl_candidate.reward:.4f}"
        )


        # ====================================================
        # Update CL state after task
        # ====================================================

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


    # ========================================================
    # ONE pooled result per method per seed
    # ========================================================

    seed_rows = []

    print(
        "\n"
        + "-" * 90
    )

    print(
        f"POOLED RESULTS | seed={seed}"
    )

    print(
        "-" * 90
    )


    for method in (
        "Naive-VQC",
        "QAS-No-CL",
        "CL-QAS",
    ):

        truth = np.asarray(
            pooled[
                method
            ][
                "true"
            ],
            dtype=np.int64,
        )

        predictions = np.asarray(
            pooled[
                method
            ][
                "pred"
            ],
            dtype=np.int64,
        )

        metrics = metrics_from_predictions(
            truth,
            predictions,
        )

        mean_n2 = float(
            np.mean(
                pooled[
                    method
                ][
                    "n2"
                ]
            )
        )

        mean_depth = float(
            np.mean(
                pooled[
                    method
                ][
                    "depth"
                ]
            )
        )

        mean_fidelity = float(
            np.mean(
                pooled[
                    method
                ][
                    "fidelity"
                ]
            )
        )

        if method == "Naive-VQC":

            mean_reward = np.nan

        else:

            mean_reward = float(
                np.mean(
                    pooled[
                        method
                    ][
                        "reward"
                    ]
                )
            )

        seed_rows.append({

            "seed":
            seed,

            "method":
            method,

            "acc":
            metrics[
                "acc"
            ],

            "bAcc":
            metrics[
                "bAcc"
            ],

            "F1":
            metrics[
                "F1"
            ],

            "reward":
            mean_reward,

            "n2":
            mean_n2,

            "depth":
            mean_depth,

            "tt_fidelity":
            mean_fidelity,

            "n_test":
            int(
                len(
                    truth
                )
            ),
        })

        print(
            f"{method:12s} | "
            f"Acc={metrics['acc']:.4f} | "
            f"BAcc={metrics['bAcc']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"#2Q={mean_n2:.2f}"
        )

    return (
        seed_rows,
        task_rows,
    )


# ============================================================
# 23. Summary
# ============================================================

def summarize(
    seed_df,
):

    summary = (
        seed_df
        .groupby(
            "method"
        )[
            [
                "acc",
                "bAcc",
                "F1",
                "reward",
                "n2",
                "tt_fidelity",
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
        "TABLE 1: OVERALL ECG PERFORMANCE "
        "(POOLED TEST PREDICTIONS; "
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
# 24. Flatten summary
# ============================================================

def flatten_summary(
    summary,
):

    output = summary.copy()

    output.columns = [
        f"{metric}_{stat}"
        for (
            metric,
            stat,
        ) in output.columns
    ]

    return output.reset_index()


# ============================================================
# 25. Main
# ============================================================

def main():

    print(
        "=" * 90
    )

    print(
        "CL-QAS TABLE 1 ECG EXPERIMENT "
        "ALIGNED WITH TABLE 3"
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
        f"TT rank: {TT_RANK}"
    )

    print(
        f"Search budget: "
        f"{SEARCH_STEPS} steps x "
        f"{CANDIDATES_PER_STEP} candidates x "
        f"{SEARCH_INNER_EPOCHS} inner epochs"
    )

    print(
        f"Search split: "
        f"{SEARCH_TRAIN_FRAC:.0%} train / "
        f"{SEARCH_VAL_FRAC:.0%} validation / "
        f"{1.0 - SEARCH_TRAIN_FRAC - SEARCH_VAL_FRAC:.0%} test"
    )

    print(
        "Final evaluation: "
        "train + validation = 90%, "
        "test = 10%"
    )


    # --------------------------------------------------------
    # Load data once
    # --------------------------------------------------------

    tasks = load_tasks()

    print(
        "\nSequential tasks:"
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
    # Run seeds
    # --------------------------------------------------------

    all_seed_rows = []

    all_task_rows = []

    for seed in SEEDS:

        (
            seed_rows,
            task_rows,
        ) = run_seed(
            tasks,
            seed,
        )

        all_seed_rows.extend(
            seed_rows
        )

        all_task_rows.extend(
            task_rows
        )


    seed_df = pd.DataFrame(
        all_seed_rows
    )

    task_df = pd.DataFrame(
        all_task_rows
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarize(
        seed_df
    )

    flat_summary = flatten_summary(
        summary
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    seed_df.to_csv(
        "table1_ecg_seed_results.csv",
        index=False,
    )

    task_df.to_csv(
        "table1_ecg_task_diagnostics.csv",
        index=False,
    )

    summary.to_csv(
        "table1_ecg_summary.csv"
    )

    flat_summary.to_csv(
        "table1_ecg_summary_flat.csv",
        index=False,
    )


    print(
        "\nSaved:"
    )

    print(
        "  table1_ecg_seed_results.csv"
    )

    print(
        "  table1_ecg_task_diagnostics.csv"
    )

    print(
        "  table1_ecg_summary.csv"
    )

    print(
        "  table1_ecg_summary_flat.csv"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()

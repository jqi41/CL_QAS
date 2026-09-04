#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ecg_clqas_controlled_noise.py

Controlled-noise robustness study for CL-QAS on MIT-BIH ECG.

======================================================================
Purpose
======================================================================

This script implements the controlled-noise experiment for Section 6.4.

Compared methods
----------------
1. Naive-VQC
2. QAS-No-CL
3. CL-QAS

Controlled noise scenarios
--------------------------
1. Ideal
2. Single-qubit depolarizing:
       p_1q = 0.001
3. Two-qubit depolarizing:
       p_2q = 0.01
4. Dephasing:
       p_z = 0.001
5. Readout error:
       p_ro = 0.10
6. Combined gate noise:
       p_1q = 0.001
       p_2q = 0.01
       p_z  = 0.001
7. Combined noise:
       p_1q = 0.001
       p_2q = 0.01
       p_z  = 0.001
       p_ro = 0.10

======================================================================
Experimental interpretation
======================================================================

The experiment evaluates POST-TRAINING robustness.

Architecture search:
    ideal

VQC optimization:
    ideal

Noise:
    applied only during final test-time evaluation

Therefore, the experiment asks:

    Given models and architectures learned under the same ideal
    optimization protocol, how sensitive are Naive-VQC,
    QAS-No-CL, and CL-QAS to controlled quantum noise?

This should NOT be described as noise-aware training or
noise-aware architecture search.

======================================================================
Noise implementation
======================================================================

The simulator is a pure-state statevector simulator.

Gate noise is therefore implemented using stochastic
Monte-Carlo Pauli trajectories.

Single-qubit depolarizing:
    E(rho)
      = (1-p) rho
        + p/3 (X rho X + Y rho Y + Z rho Z)

Two-qubit depolarizing:
    E(rho)
      = (1-p) rho
        + p/15 sum_{P != II} P rho P

Dephasing:
    E(rho)
      = (1-p) rho
        + p Z rho Z

Readout:
    symmetric bit-flip error with finite-shot sampling.

======================================================================
Main ECG protocol
======================================================================

Records:
    105, 106, 109, 114, 116, 119, 200, 201

Classes:
    AAMI N vs V

Input:
    256-dimensional ECG beat

Representation:
    TT-SVD
    modes = (4,16,4)
    rank = 3

Per sequential task:
    80% search-training
    10% architecture validation
    10% test

After architecture selection:
    final VQC reinitialized
    train + validation = 90%
    test = 10%

Reported statistics:
    pooled predictions across ECG tasks within each seed
    mean +/- std across seeds

Dependencies
------------
pip install torch numpy pandas scikit-learn wfdb certifi
"""

# =====================================================================
# Imports
# =====================================================================

import os
import time
import copy
import random

from contextlib import contextmanager
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


# =====================================================================
# SSL / WFDB
# =====================================================================

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
        "Please install dependencies:\n"
        "pip install wfdb certifi"
    ) from exc


# =====================================================================
# 1. Global configuration
# =====================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

REAL = torch.float32
COMPLEX = torch.complex64


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# AAMI binary labels
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Quantum representation
# ---------------------------------------------------------------------

NUM_QUBITS = 8

NUM_CLASSES = 2

TT_MODES = (
    4,
    16,
    4,
)

TT_RANK = 3


# ---------------------------------------------------------------------
# Architecture search
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Final VQC refit
# ---------------------------------------------------------------------

FINAL_REFIT_EPOCHS = 40

BATCH_SIZE = 64


# ---------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------

VQC_LR = 3e-3

POLICY_LR = 1e-3

WEIGHT_DECAY = 1e-5


# ---------------------------------------------------------------------
# Continual-learning regularization
# ---------------------------------------------------------------------

MU_EWC = 0.5

ETA_KL = 0.01

ENTROPY_COEF = 0.002


# ---------------------------------------------------------------------
# Hardware-aware search reward
# ---------------------------------------------------------------------

LAMBDA_HW = 0.01


# ---------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------

SEARCH_TRAIN_FRAC = 0.80

SEARCH_VAL_FRAC = 0.10

# remaining 10% = test

SPLIT_SEED_BASE = 5000


# ---------------------------------------------------------------------
# Random seeds
# ---------------------------------------------------------------------

SEEDS = (
    11,
    22,
    33,
)


# =====================================================================
# 2. Controlled-noise configuration
# =====================================================================

# Single-qubit depolarizing probability
P1_DEPOLARIZING = 0.001

# Two-qubit depolarizing probability
P2_DEPOLARIZING = 0.01

# Phase-flip / dephasing probability
P_DEPHASING = 0.001

# Symmetric readout bit-flip probability
P_READOUT = 0.10


# Number of stochastic gate-noise trajectories
#
# 16 is a reasonable initial value.
# Increase to 32 or 64 for final high-precision results.
NOISE_TRAJECTORIES = 16


# Number of measurement shots used when readout error is present
READOUT_SHOTS = 1024


@dataclass(frozen=True)
class NoiseConfig:

    name: str

    p1_depolarizing: float = 0.0

    p2_depolarizing: float = 0.0

    p_dephasing: float = 0.0

    p_readout: float = 0.0


NOISE_SCENARIOS = (

    # ---------------------------------------------------------
    # Ideal
    # ---------------------------------------------------------

    NoiseConfig(
        name="Ideal",
    ),

    # ---------------------------------------------------------
    # Individual controlled-noise sources
    # ---------------------------------------------------------

    NoiseConfig(
        name="1Q-depolarizing",
        p1_depolarizing=
        P1_DEPOLARIZING,
    ),

    NoiseConfig(
        name="2Q-depolarizing",
        p2_depolarizing=
        P2_DEPOLARIZING,
    ),

    NoiseConfig(
        name="Dephasing",
        p_dephasing=
        P_DEPHASING,
    ),

    NoiseConfig(
        name="Readout",
        p_readout=
        P_READOUT,
    ),

    # ---------------------------------------------------------
    # Combined gate noise
    # ---------------------------------------------------------

    NoiseConfig(
        name="Combined-gate",
        p1_depolarizing=
        P1_DEPOLARIZING,
        p2_depolarizing=
        P2_DEPOLARIZING,
        p_dephasing=
        P_DEPHASING,
    ),

    # ---------------------------------------------------------
    # Combined gate + readout noise
    # ---------------------------------------------------------

    NoiseConfig(
        name="Combined",
        p1_depolarizing=
        P1_DEPOLARIZING,
        p2_depolarizing=
        P2_DEPOLARIZING,
        p_dephasing=
        P_DEPHASING,
        p_readout=
        P_READOUT,
    ),
)


# =====================================================================
# 3. Reproducibility
# =====================================================================

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


# =====================================================================
# 4. Preserve random state during noisy evaluation
# =====================================================================

@contextmanager
def preserve_rng_state():
    """
    Preserve Python / NumPy / PyTorch RNG states.

    This is important because Monte-Carlo noisy evaluation consumes
    random numbers. Without restoring the RNG state, simply inserting
    the controlled-noise experiment would alter later QAS trajectories
    and therefore change the ideal architectures.
    """

    python_state = random.getstate()

    numpy_state = np.random.get_state()

    torch_state = (
        torch.random.get_rng_state()
    )

    if torch.cuda.is_available():

        cuda_states = (
            torch.cuda.get_rng_state_all()
        )

    else:

        cuda_states = None

    try:

        yield

    finally:

        random.setstate(
            python_state
        )

        np.random.set_state(
            numpy_state
        )

        torch.random.set_rng_state(
            torch_state
        )

        if cuda_states is not None:

            torch.cuda.set_rng_state_all(
                cuda_states
            )


# =====================================================================
# 5. MIT-BIH loading
# =====================================================================

def read_mitdb_record(
    record,
    max_retries=5,
    retry_wait=3.0,
):

    record = str(
        record
    )

    # -----------------------------------------------------------------
    # Local database first
    # -----------------------------------------------------------------

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


    # -----------------------------------------------------------------
    # PhysioNet fallback
    # -----------------------------------------------------------------

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
                f"{type(exc).__name__}: "
                f"{exc}"
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


# =====================================================================
# 6. ECG preprocessing
# =====================================================================

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
        fields[
            "sig_name"
        ]
    )

    fs = float(
        fields[
            "fs"
        ]
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
            or
            end >= len(x)
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
            and
            len(X) >= max_beats
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
        or
        counts[1] < min_per_class
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


# =====================================================================
# 7. Shared stratified split
# =====================================================================

def stratified_split(
    y: torch.Tensor,
    train_frac: float =
    SEARCH_TRAIN_FRAC,
    val_frac: float =
    SEARCH_VAL_FRAC,
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
                n
                - n_train
                - 1,
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
            +
            100
            * seed
            +
            task_id
        )

        (
            train_idx,
            val_idx,
            test_idx,
        ) = stratified_split(
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


# =====================================================================
# 8. Train-only feature standardization
# =====================================================================

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

    std = (
        X_train.std(
            dim=0
        )
        + 1e-6
    )

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


# =====================================================================
# 9. TT-SVD representation
# =====================================================================

def normalize_state(
    x,
    eps=1e-8,
):

    return (
        x
        /
        (
            torch.linalg.vector_norm(
                x
            )
            + eps
        )
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
            "the input dimension."
        )

    tensor = x.reshape(
        *modes
    )

    cores = []

    rank_previous = 1

    remainder = tensor

    for k in range(
        len(modes) - 1
    ):

        n_k = modes[
            k
        ]

        matrix = remainder.reshape(
            rank_previous
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
            rank_previous,
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

        rank_previous = rank

        remainder = remainder.reshape(
            rank_previous,
            *modes[
                k + 1:
            ],
        )

    cores.append(
        remainder.reshape(
            rank_previous,
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


# =====================================================================
# 10. Task preparation
# =====================================================================

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


    # -----------------------------------------------------------------
    # Train-only standardization
    # -----------------------------------------------------------------

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


    # -----------------------------------------------------------------
    # Rank-r TT representation
    # -----------------------------------------------------------------

    X_train, F_train = (
        tt_encode_dataset(
            X_train_std,
            rank,
        )
    )

    X_val, F_val = (
        tt_encode_dataset(
            X_val_std,
            rank,
        )
    )

    X_test, F_test = (
        tt_encode_dataset(
            X_test_std,
            rank,
        )
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


# =====================================================================
# 11. Ideal quantum gates
# =====================================================================

def rx(
    theta,
):

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


def ry(
    theta,
):

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


def rz(
    theta,
):

    a = torch.exp(
        -0.5j
        * theta
    )

    b = torch.exp(
        0.5j
        * theta
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

    batch_size = (
        state.shape[
            0
        ]
    )

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
            for i
            in range(
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

    original_shape = (
        psi.shape
    )

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
        2
        ** NUM_QUBITS
    )

    indices = torch.arange(
        dimension,
        device=state.device,
    )

    control_bit = (
        indices
        >>
        (
            NUM_QUBITS
            - 1
            - control
        )
    ) & 1

    target_mask = (
        1
        <<
        (
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
            >>
            (
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


# =====================================================================
# 12. Pauli gates for stochastic noise trajectories
# =====================================================================

def pauli_x(
    device,
):

    return torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=COMPLEX,
        device=device,
    )


def pauli_y(
    device,
):

    return torch.tensor(
        [
            [0.0, -1.0j],
            [1.0j, 0.0],
        ],
        dtype=COMPLEX,
        device=device,
    )


def pauli_z(
    device,
):

    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=COMPLEX,
        device=device,
    )


def get_pauli(
    name,
    device,
):

    if name == "I":

        return None

    if name == "X":

        return pauli_x(
            device
        )

    if name == "Y":

        return pauli_y(
            device
        )

    if name == "Z":

        return pauli_z(
            device
        )

    raise ValueError(
        f"Unknown Pauli: {name}"
    )


def apply_pauli_subset(
    state,
    mask,
    wire,
    gate,
):
    """
    Apply the selected Pauli gate only to the statevectors
    corresponding to mask=True.
    """

    if not torch.any(
        mask
    ):

        return state

    output = state.clone()

    selected = output[
        mask
    ]

    selected = apply_1q(
        selected,
        gate,
        wire,
    )

    output[
        mask
    ] = selected

    return output


# =====================================================================
# 13. Single-qubit depolarizing noise
# =====================================================================

def apply_single_qubit_depolarizing(
    state,
    wire,
    probability,
):

    if probability <= 0.0:

        return state

    batch_size = (
        state.shape[
            0
        ]
    )

    event = torch.rand(
        batch_size,
        device=state.device,
    )

    noisy_mask = (
        event
        <
        probability
    )

    if not torch.any(
        noisy_mask
    ):

        return state

    choice = torch.randint(
        low=0,
        high=3,
        size=(
            batch_size,
        ),
        device=state.device,
    )

    state = apply_pauli_subset(
        state,
        noisy_mask
        &
        (
            choice == 0
        ),
        wire,
        pauli_x(
            state.device
        ),
    )

    state = apply_pauli_subset(
        state,
        noisy_mask
        &
        (
            choice == 1
        ),
        wire,
        pauli_y(
            state.device
        ),
    )

    state = apply_pauli_subset(
        state,
        noisy_mask
        &
        (
            choice == 2
        ),
        wire,
        pauli_z(
            state.device
        ),
    )

    return state


# =====================================================================
# 14. Dephasing noise
# =====================================================================

def apply_dephasing(
    state,
    wire,
    probability,
):

    if probability <= 0.0:

        return state

    batch_size = (
        state.shape[
            0
        ]
    )

    mask = (
        torch.rand(
            batch_size,
            device=state.device,
        )
        <
        probability
    )

    if not torch.any(
        mask
    ):

        return state

    return apply_pauli_subset(
        state,
        mask,
        wire,
        pauli_z(
            state.device
        ),
    )


# =====================================================================
# 15. Two-qubit depolarizing noise
# =====================================================================

TWO_QUBIT_PAULI_PAIRS = (

    ("I", "X"),
    ("I", "Y"),
    ("I", "Z"),

    ("X", "I"),
    ("X", "X"),
    ("X", "Y"),
    ("X", "Z"),

    ("Y", "I"),
    ("Y", "X"),
    ("Y", "Y"),
    ("Y", "Z"),

    ("Z", "I"),
    ("Z", "X"),
    ("Z", "Y"),
    ("Z", "Z"),
)


def apply_two_qubit_depolarizing(
    state,
    wire_a,
    wire_b,
    probability,
):

    if probability <= 0.0:

        return state

    batch_size = (
        state.shape[
            0
        ]
    )

    event = torch.rand(
        batch_size,
        device=state.device,
    )

    noisy_mask = (
        event
        <
        probability
    )

    if not torch.any(
        noisy_mask
    ):

        return state

    choices = torch.randint(
        low=0,
        high=len(
            TWO_QUBIT_PAULI_PAIRS
        ),
        size=(
            batch_size,
        ),
        device=state.device,
    )

    for index, (
        first,
        second,
    ) in enumerate(
        TWO_QUBIT_PAULI_PAIRS
    ):

        mask = (
            noisy_mask
            &
            (
                choices
                == index
            )
        )

        if not torch.any(
            mask
        ):

            continue

        first_gate = get_pauli(
            first,
            state.device,
        )

        second_gate = get_pauli(
            second,
            state.device,
        )

        if first_gate is not None:

            state = apply_pauli_subset(
                state,
                mask,
                wire_a,
                first_gate,
            )

        if second_gate is not None:

            state = apply_pauli_subset(
                state,
                mask,
                wire_b,
                second_gate,
            )

    return state


# =====================================================================
# 16. Circuit architecture
# =====================================================================

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
            for q
            in range(
                NUM_QUBITS
            )
        ]

    if pattern == "linear":

        return [
            (
                q,
                q + 1,
            )
            for q
            in range(
                NUM_QUBITS - 1
            )
        ]

    if pattern == "brick_even":

        return [
            (
                q,
                q + 1,
            )
            for q
            in range(
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
            for q
            in range(
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
        *
        NUM_QUBITS
        *
        3
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


# =====================================================================
# 17. VQC model: ideal + controlled-noise forward
# =====================================================================

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
            *
            torch.randn(
                architecture.depth,
                NUM_QUBITS,
                3,
            )
        )


    # -----------------------------------------------------------------
    # Ideal forward
    # -----------------------------------------------------------------

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
            :NUM_CLASSES
        ]


    # -----------------------------------------------------------------
    # Controlled-noise forward
    # -----------------------------------------------------------------

    def forward_noisy(
        self,
        amplitudes,
        noise_config,
        trajectories=
        NOISE_TRAJECTORIES,
        readout_shots=
        READOUT_SHOTS,
    ):
        """
        Controlled-noise evaluation using stochastic trajectories.

        No gradients are needed in this function because it is used
        only after training.
        """

        has_gate_noise = (

            noise_config
            .p1_depolarizing
            > 0.0

            or

            noise_config
            .p2_depolarizing
            > 0.0

            or

            noise_config
            .p_dephasing
            > 0.0
        )


        # -------------------------------------------------------------
        # Fully ideal evaluation
        # -------------------------------------------------------------

        if (
            not has_gate_noise
            and
            noise_config
            .p_readout
            <= 0.0
        ):

            return self.forward(
                amplitudes
            )


        original_batch = (
            amplitudes.shape[
                0
            ]
        )


        # -------------------------------------------------------------
        # Only gate noise requires trajectory replication.
        # -------------------------------------------------------------

        if has_gate_noise:

            num_trajectories = (
                trajectories
            )

        else:

            num_trajectories = 1


        state = amplitudes.repeat(
            num_trajectories,
            1,
        ).to(
            DEVICE,
            dtype=COMPLEX,
        )


        # =============================================================
        # Noisy gate evolution
        # =============================================================

        for layer in range(
            self.architecture.depth
        ):

            for qubit in range(
                NUM_QUBITS
            ):

                # -----------------------------------------------------
                # RX
                # -----------------------------------------------------

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

                state = (
                    apply_single_qubit_depolarizing(
                        state,
                        qubit,
                        noise_config
                        .p1_depolarizing,
                    )
                )

                state = apply_dephasing(
                    state,
                    qubit,
                    noise_config
                    .p_dephasing,
                )


                # -----------------------------------------------------
                # RY
                # -----------------------------------------------------

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

                state = (
                    apply_single_qubit_depolarizing(
                        state,
                        qubit,
                        noise_config
                        .p1_depolarizing,
                    )
                )

                state = apply_dephasing(
                    state,
                    qubit,
                    noise_config
                    .p_dephasing,
                )


                # -----------------------------------------------------
                # RZ
                # -----------------------------------------------------

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

                state = (
                    apply_single_qubit_depolarizing(
                        state,
                        qubit,
                        noise_config
                        .p1_depolarizing,
                    )
                )

                state = apply_dephasing(
                    state,
                    qubit,
                    noise_config
                    .p_dephasing,
                )


            # ---------------------------------------------------------
            # Entangling layer
            # ---------------------------------------------------------

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

                state = (
                    apply_two_qubit_depolarizing(
                        state,
                        control,
                        target,
                        noise_config
                        .p2_depolarizing,
                    )
                )


        # =============================================================
        # Measurement expectation values
        # =============================================================

        z = z_expectations(
            state
        )[
            :,
            :NUM_CLASSES
        ]


        # -------------------------------------------------------------
        # Trajectory average
        # -------------------------------------------------------------

        z = z.reshape(
            num_trajectories,
            original_batch,
            NUM_CLASSES,
        )

        z = z.mean(
            dim=0
        )


        # =============================================================
        # Readout error
        # =============================================================

        if (
            noise_config
            .p_readout
            > 0.0
        ):

            # p(1) = (1 - <Z>)/2
            probability_one = (
                1.0
                - z
            ) / 2.0


            error = (
                noise_config
                .p_readout
            )


            # symmetric measurement-error model
            #
            # p_obs(1)
            # =
            # e + (1-2e) p_true(1)

            observed_probability_one = (

                error

                +

                (
                    1.0
                    - 2.0
                    * error
                )
                * probability_one
            )


            observed_probability_one = (
                observed_probability_one
                .clamp(
                    0.0,
                    1.0,
                )
            )


            # ---------------------------------------------------------
            # Finite-shot measurement
            # ---------------------------------------------------------

            total_counts = (
                torch.full_like(
                    observed_probability_one,
                    float(
                        readout_shots
                    ),
                )
            )


            counts_one = torch.binomial(
                total_counts,
                observed_probability_one,
            )


            z = (
                1.0
                -
                2.0
                * counts_one
                /
                float(
                    readout_shots
                )
            )


        return z


# =====================================================================
# 18. Architecture policy
# =====================================================================

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
            *
            torch.randn(
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

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
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

        task_feature = (
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
            +
            task_feature.unsqueeze(1)
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


# =====================================================================
# 19. Policy sampling
# =====================================================================

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
                logits=
                pattern_logits[
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
                    pattern_index
                    .item()
                )
            ]
        )

        log_prob = (
            log_prob
            +
            pattern_dist.log_prob(
                pattern_index
            )
        )

        entropy = (
            entropy
            +
            pattern_dist.entropy()
        )

    entropy = (
        entropy
        /
        (
            depth + 1
        )
    )

    return PolicySample(

        architecture=
        Architecture(
            depth=depth,
            patterns=tuple(
                patterns
            ),
        ),

        log_prob=
        log_prob,

        entropy=
        entropy,
    )


# =====================================================================
# 20. EWC
# =====================================================================

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

            *

            (
                parameter
                -
                state.means[
                    name
                ]
            ) ** 2
        )

    return (
        0.5
        *
        penalty
    )


# =====================================================================
# 21. KL policy regularization
# =====================================================================

def categorical_kl(
    current_logits,
    reference_logits,
):

    probability = torch.softmax(
        current_logits,
        dim=-1,
    )

    log_probability = (
        torch.log_softmax(
            current_logits,
            dim=-1,
        )
    )

    log_reference = (
        torch.log_softmax(
            reference_logits,
            dim=-1,
        )
    )

    return torch.sum(

        probability

        *

        (
            log_probability
            -
            log_reference
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

    pattern_term = (
        categorical_kl(
            pattern_logits,
            reference_pattern,
        )
        .mean()
    )

    return (
        depth_term
        +
        pattern_term
    )


# =====================================================================
# 22. VQC training
# =====================================================================

def class_weighted_loss(
    y_train,
):

    counts = torch.bincount(
        y_train,
        minlength=2,
    ).float()

    weights = (
        counts.sum()
        /
        (
            2.0
            *
            counts.clamp_min(
                1
            )
        )
    )

    weights = torch.clamp(
        weights,
        max=4.0,
    )

    return nn.CrossEntropyLoss(
        weight=
        weights.to(
            DEVICE
        )
    )


def train_vqc(
    architecture,
    X_train,
    y_train,
    epochs,
):

    model = VQC(
        architecture
    ).to(
        DEVICE
    )

    criterion = (
        class_weighted_loss(
            y_train
        )
    )

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=
        VQC_LR,

        weight_decay=
        WEIGHT_DECAY,
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


# =====================================================================
# 23. Ideal prediction / metrics
# =====================================================================

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
            len(
                X
            ),
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

    truth, predictions = (
        predict_vqc(
            model,
            X,
            y,
        )
    )

    return metrics_from_predictions(
        truth,
        predictions,
    )


# =====================================================================
# 24. Noisy prediction
# =====================================================================

def predict_vqc_noisy(
    model,
    X,
    y,
    noise_config,
):

    model.eval()

    truth = []

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(
                X
            ),
            BATCH_SIZE,
        ):

            xb = X[
                start:
                start + BATCH_SIZE
            ].to(
                DEVICE
            )

            logits = model.forward_noisy(
                xb,
                noise_config,
                trajectories=
                NOISE_TRAJECTORIES,
                readout_shots=
                READOUT_SHOTS,
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


# =====================================================================
# 25. Architecture-search reward
# =====================================================================

def predictive_score(
    metrics,
):

    return (
        0.70
        *
        metrics[
            "bAcc"
        ]

        +

        0.30
        *
        metrics[
            "F1"
        ]
    )


def hardware_cost(
    architecture,
):

    statistics = (
        architecture_stats(
            architecture
        )
    )

    reference_n2 = (
        4
        *
        NUM_QUBITS
    )

    return (
        statistics[
            "n2"
        ]
        /
        reference_n2
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
        *
        hardware_cost(
            architecture
        )
    )


# =====================================================================
# 26. Candidate evaluation
# =====================================================================

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

    reward = (
        architecture_reward(
            metrics,
            architecture,
        )
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


# =====================================================================
# 27. Bi-loop architecture search
# =====================================================================

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

            policy_sample = (
                sample_architecture(
                    policy,
                    context,
                )
            )

            result = evaluate_candidate(

                policy_sample
                .architecture,

                X_train,
                y_train,

                X_val,
                y_val,
            )

            result.log_prob = (
                policy_sample
                .log_prob
            )

            result.entropy = (
                policy_sample
                .entropy
            )

            candidates.append(
                result
            )

            if (
                best_candidate
                is None
                or
                result.reward
                >
                best_candidate.reward
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
            -
            baseline,
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

        reinforce_loss = (
            -torch.mean(
                advantages.detach()
                *
                log_probs
            )
        )


        # -------------------------------------------------------------
        # EWC
        # -------------------------------------------------------------

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


        # -------------------------------------------------------------
        # KL
        # -------------------------------------------------------------

        if (
            use_kl
            and
            reference_policy
            is not None
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
            *
            L_ewc

            +

            ETA_KL
            *
            L_kl

            -

            ENTROPY_COEF
            *
            entropy
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
            *
            baseline

            +

            0.1
            *
            mean_reward
        )

    return (
        best_candidate,
        context,
    )


# =====================================================================
# 28. Final ideal refit
# =====================================================================

def final_refit_model(
    architecture,
    X_train,
    y_train,
    X_val,
    y_val,
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
    )

    return model


# =====================================================================
# 29. Evaluate one trained VQC under all noise scenarios
# =====================================================================

def evaluate_all_noise_scenarios(
    model,
    X_test,
    y_test,
    *,
    seed,
    task_id,
    method_index,
):
    """
    Each trained model is evaluated under the same predefined noise
    conditions.

    Noise RNG is deterministic and isolated from the training/search RNG.
    """

    outputs = {}

    for scenario_index, (
        noise_config
    ) in enumerate(
        NOISE_SCENARIOS
    ):

        noise_seed = (

            1_000_000

            +

            10_000
            * seed

            +

            100
            * task_id

            +

            10
            * method_index

            +

            scenario_index
        )

        with preserve_rng_state():

            set_seed(
                noise_seed
            )

            truth, predictions = (
                predict_vqc_noisy(
                    model,
                    X_test,
                    y_test,
                    noise_config,
                )
            )

        outputs[
            noise_config.name
        ] = {

            "true":
            truth,

            "pred":
            predictions,
        }

    return outputs


# =====================================================================
# 30. One complete seed
# =====================================================================

def run_seed(
    tasks,
    seed,
):

    split_cache = build_split_cache(
        tasks,
        seed,
    )


    # -----------------------------------------------------------------
    # Independent architecture policies
    # -----------------------------------------------------------------

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


    methods = (
        "Naive-VQC",
        "QAS-No-CL",
        "CL-QAS",
    )


    # -----------------------------------------------------------------
    # Separate prediction pools for each noise scenario
    # -----------------------------------------------------------------

    pooled = {}

    for method in methods:

        pooled[
            method
        ] = {

            noise.name: {

                "true": [],

                "pred": [],

            }

            for noise
            in NOISE_SCENARIOS
        }


    task_rows = []


    # =================================================================
    # Sequential ECG tasks
    # =================================================================

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
            +
            "=" * 100
        )

        print(
            f"Seed {seed} | "
            f"Task {task_id} | "
            f"MIT-BIH record {record}"
        )

        print(
            "=" * 100
        )


        # -------------------------------------------------------------
        # Identical TT representation for all methods
        # -------------------------------------------------------------

        data = prepare_task(

            X_raw,

            y,

            rank=
            TT_RANK,

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


        # =============================================================
        # A. Naive-VQC
        # =============================================================

        print(
            "\n[Naive-VQC]"
        )

        naive_arch = (
            naive_architecture()
        )

        naive_model = final_refit_model(

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
        )

        naive_outputs = (
            evaluate_all_noise_scenarios(

                naive_model,

                data[
                    "X_test"
                ],

                data[
                    "y_test"
                ],

                seed=
                seed,

                task_id=
                task_id,

                method_index=
                0,
            )
        )

        naive_stats = (
            architecture_stats(
                naive_arch
            )
        )


        # =============================================================
        # B. QAS-No-CL
        # =============================================================

        print(
            "[QAS-No-CL]"
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

        qas_model = final_refit_model(

            qas_candidate
            .architecture,

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
        )

        qas_outputs = (
            evaluate_all_noise_scenarios(

                qas_model,

                data[
                    "X_test"
                ],

                data[
                    "y_test"
                ],

                seed=
                seed,

                task_id=
                task_id,

                method_index=
                1,
            )
        )

        qas_stats = (
            architecture_stats(
                qas_candidate
                .architecture
            )
        )


        # =============================================================
        # C. CL-QAS
        # =============================================================

        print(
            "[CL-QAS]"
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

        cl_model = final_refit_model(

            cl_candidate
            .architecture,

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
        )

        cl_outputs = (
            evaluate_all_noise_scenarios(

                cl_model,

                data[
                    "X_test"
                ],

                data[
                    "y_test"
                ],

                seed=
                seed,

                task_id=
                task_id,

                method_index=
                2,
            )
        )

        cl_stats = (
            architecture_stats(
                cl_candidate
                .architecture
            )
        )


        # =============================================================
        # Collect per-task noisy results
        # =============================================================

        method_outputs = {

            "Naive-VQC": (
                naive_outputs,
                naive_stats,
                np.nan,
            ),

            "QAS-No-CL": (
                qas_outputs,
                qas_stats,
                qas_candidate.reward,
            ),

            "CL-QAS": (
                cl_outputs,
                cl_stats,
                cl_candidate.reward,
            ),
        }


        for method, (
            noise_results,
            statistics,
            reward,
        ) in method_outputs.items():

            print(
                f"\n  {method}"
            )

            for noise_name, values in (
                noise_results.items()
            ):

                pooled[
                    method
                ][
                    noise_name
                ][
                    "true"
                ].extend(
                    values[
                        "true"
                    ].tolist()
                )

                pooled[
                    method
                ][
                    noise_name
                ][
                    "pred"
                ].extend(
                    values[
                        "pred"
                    ].tolist()
                )


                task_metrics = (
                    metrics_from_predictions(

                        values[
                            "true"
                        ],

                        values[
                            "pred"
                        ],
                    )
                )


                task_rows.append({

                    "seed":
                    seed,

                    "task":
                    task_id,

                    "record":
                    record,

                    "method":
                    method,

                    "noise":
                    noise_name,

                    "acc":
                    task_metrics[
                        "acc"
                    ],

                    "bAcc":
                    task_metrics[
                        "bAcc"
                    ],

                    "F1":
                    task_metrics[
                        "F1"
                    ],

                    "n2":
                    statistics[
                        "n2"
                    ],

                    "depth":
                    statistics[
                        "depth"
                    ],

                    "reward":
                    reward,

                    "tt_fidelity":
                    data[
                        "fidelity"
                    ],
                })


                print(
                    f"    {noise_name:16s} | "
                    f"Acc={task_metrics['acc']:.4f} | "
                    f"BAcc={task_metrics['bAcc']:.4f} | "
                    f"F1={task_metrics['F1']:.4f}"
                )


        # =============================================================
        # Update continual policy state AFTER task
        # =============================================================

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
            reference_policy
            .parameters()
        ):

            parameter.requires_grad_(
                False
            )


    # =================================================================
    # Pooled results for this seed
    # =================================================================

    seed_rows = []


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        f"POOLED CONTROLLED-NOISE RESULTS | "
        f"seed={seed}"
    )

    print(
        "=" * 100
    )


    for method in methods:

        # -------------------------------------------------------------
        # Ideal reference for degradation
        # -------------------------------------------------------------

        ideal_truth = np.asarray(
            pooled[
                method
            ][
                "Ideal"
            ][
                "true"
            ],
            dtype=np.int64,
        )

        ideal_predictions = np.asarray(
            pooled[
                method
            ][
                "Ideal"
            ][
                "pred"
            ],
            dtype=np.int64,
        )

        ideal_metrics = (
            metrics_from_predictions(
                ideal_truth,
                ideal_predictions,
            )
        )


        for noise_config in (
            NOISE_SCENARIOS
        ):

            noise_name = (
                noise_config.name
            )

            truth = np.asarray(
                pooled[
                    method
                ][
                    noise_name
                ][
                    "true"
                ],
                dtype=np.int64,
            )

            predictions = np.asarray(
                pooled[
                    method
                ][
                    noise_name
                ][
                    "pred"
                ],
                dtype=np.int64,
            )

            current_metrics = (
                metrics_from_predictions(
                    truth,
                    predictions,
                )
            )


            # ---------------------------------------------------------
            # Positive value means degradation from ideal.
            # ---------------------------------------------------------

            acc_drop = (
                ideal_metrics[
                    "acc"
                ]
                -
                current_metrics[
                    "acc"
                ]
            )

            bacc_drop = (
                ideal_metrics[
                    "bAcc"
                ]
                -
                current_metrics[
                    "bAcc"
                ]
            )

            f1_drop = (
                ideal_metrics[
                    "F1"
                ]
                -
                current_metrics[
                    "F1"
                ]
            )


            seed_rows.append({

                "seed":
                seed,

                "method":
                method,

                "noise":
                noise_name,

                "p1_depolarizing":
                noise_config
                .p1_depolarizing,

                "p2_depolarizing":
                noise_config
                .p2_depolarizing,

                "p_dephasing":
                noise_config
                .p_dephasing,

                "p_readout":
                noise_config
                .p_readout,

                "acc":
                current_metrics[
                    "acc"
                ],

                "bAcc":
                current_metrics[
                    "bAcc"
                ],

                "F1":
                current_metrics[
                    "F1"
                ],

                "acc_drop":
                acc_drop,

                "bAcc_drop":
                bacc_drop,

                "F1_drop":
                f1_drop,

                "n_test":
                int(
                    len(
                        truth
                    )
                ),
            })


            print(
                f"{method:12s} | "
                f"{noise_name:16s} | "
                f"Acc={current_metrics['acc']:.4f} | "
                f"BAcc={current_metrics['bAcc']:.4f} | "
                f"F1={current_metrics['F1']:.4f} | "
                f"dF1={f1_drop:+.4f}"
            )


    return (
        seed_rows,
        task_rows,
    )


# =====================================================================
# 31. Summary
# =====================================================================

def summarize_noise(
    seed_df,
):

    summary = (

        seed_df

        .groupby(
            [
                "noise",
                "method",
            ],
            sort=False,
        )[
            [
                "acc",
                "bAcc",
                "F1",
                "acc_drop",
                "bAcc_drop",
                "F1_drop",
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
        +
        "=" * 150
    )

    print(
        "CONTROLLED-NOISE ROBUSTNESS "
        "(POOLED ECG RESULTS; "
        "MEAN +/- STD ACROSS SEEDS)"
    )

    print(
        "=" * 150
    )

    print(
        summary.to_string()
    )

    return summary


# =====================================================================
# 32. Compact publication table
# =====================================================================

def build_compact_summary(
    seed_df,
):
    """
    Produces one compact row for every noise/method combination.

    Useful for manuscript-table construction.
    """

    rows = []

    noise_order = [
        config.name
        for config in NOISE_SCENARIOS
    ]

    method_order = [
        "Naive-VQC",
        "QAS-No-CL",
        "CL-QAS",
    ]

    for noise_name in noise_order:
        for method in method_order:
            subset = seed_df[
                (
                    seed_df[
                        "noise"
                    ]
                    == noise_name
                )
                &
                (
                    seed_df[
                        "method"
                    ]
                    == method
                )
            ]

            if len(
                subset
            ) == 0:

                continue

            rows.append({

                "noise":
                noise_name,

                "method":
                method,

                "acc_mean":
                subset[
                    "acc"
                ].mean(),

                "acc_std":
                subset[
                    "acc"
                ].std(
                    ddof=1
                ),

                "bAcc_mean":
                subset[
                    "bAcc"
                ].mean(),

                "bAcc_std":
                subset[
                    "bAcc"
                ].std(
                    ddof=1
                ),

                "F1_mean":
                subset[
                    "F1"
                ].mean(),

                "F1_std":
                subset[
                    "F1"
                ].std(
                    ddof=1
                ),

                "acc_drop_mean":
                subset[
                    "acc_drop"
                ].mean(),

                "bAcc_drop_mean":
                subset[
                    "bAcc_drop"
                ].mean(),

                "F1_drop_mean":
                subset[
                    "F1_drop"
                ].mean(),
            })

    return pd.DataFrame(
        rows
    )


# =====================================================================
# 33. Noise degradation summary
# =====================================================================
def build_degradation_summary(
    seed_df,
):

    noisy_df = seed_df[
        seed_df[
            "noise"
        ]
        != "Ideal"
    ].copy()

    degradation = (

        noisy_df

        .groupby(
            [
                "noise",
                "method",
            ],
            sort=False,
        )[
            [
                "acc_drop",
                "bAcc_drop",
                "F1_drop",
            ]
        ]

        .agg(
            [
                "mean",
                "std",
            ]
        )
    )
    return degradation


# =====================================================================
# 34. Main
# =====================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "CL-QAS ECG CONTROLLED-NOISE ROBUSTNESS"
    )

    print(
        "=" * 100
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
        "\nNoise configuration:"
    )

    print(
        f"  1Q depolarizing: "
        f"{P1_DEPOLARIZING}"
    )

    print(
        f"  2Q depolarizing: "
        f"{P2_DEPOLARIZING}"
    )

    print(
        f"  dephasing: "
        f"{P_DEPHASING}"
    )

    print(
        f"  readout error: "
        f"{P_READOUT}"
    )

    print(
        f"  stochastic trajectories: "
        f"{NOISE_TRAJECTORIES}"
    )

    print(
        f"  measurement shots: "
        f"{READOUT_SHOTS}"
    )


    print(
        "\nNoise scenarios:"
    )

    for config in (
        NOISE_SCENARIOS
    ):

        print(
            f"  {config.name:16s} | "
            f"p1={config.p1_depolarizing:.4f} | "
            f"p2={config.p2_depolarizing:.4f} | "
            f"pz={config.p_dephasing:.4f} | "
            f"pro={config.p_readout:.4f}"
        )


    # -----------------------------------------------------------------
    # Load ECG data
    # -----------------------------------------------------------------

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


    # -----------------------------------------------------------------
    # Experiments
    # -----------------------------------------------------------------

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


    # -----------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------

    summary = summarize_noise(
        seed_df
    )

    compact_summary = (
        build_compact_summary(
            seed_df
        )
    )

    degradation = (
        build_degradation_summary(
            seed_df
        )
    )

    print(
        "\n"
        +
        "=" * 130
    )

    print(
        "COMPACT CONTROLLED-NOISE TABLE"
    )

    print(
        "=" * 130
    )

    print(
        compact_summary.to_string(
            index=False
        )
    )

    print(
        "\n"
        +
        "=" * 130
    )

    print(
        "DEGRADATION RELATIVE TO IDEAL"
    )

    print(
        "=" * 130
    )

    print(
        degradation.to_string()
    )

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------

    seed_df.to_csv(
        "ecg_controlled_noise_seed_results.csv",
        index=False,
    )

    task_df.to_csv(
        "ecg_controlled_noise_task_results.csv",
        index=False,
    )

    summary.to_csv(
        "ecg_controlled_noise_summary.csv"
    )

    compact_summary.to_csv(
        "ecg_controlled_noise_compact_summary.csv",
        index=False,
    )

    degradation.to_csv(
        "ecg_controlled_noise_degradation.csv"
    )


    print(
        "\nSaved:"
    )

    print(
        "  ecg_controlled_noise_seed_results.csv"
    )

    print(
        "  ecg_controlled_noise_task_results.csv"
    )

    print(
        "  ecg_controlled_noise_summary.csv"
    )

    print(
        "  ecg_controlled_noise_compact_summary.csv"
    )

    print(
        "  ecg_controlled_noise_degradation.csv"
    )

# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":

    main()

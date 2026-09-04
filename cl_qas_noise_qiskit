#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clqas_ecg_backend_informed_noise_mps.py

Backend-informed realistic-noise evaluation for CL-QAS on MIT-BIH ECG.\n\nRevision v3:\n- uses an explicit Qiskit CouplingMap;\n- keeps all noisy simulation at 8 qubits;\n- compiles the architecture-dependent VQC once per trained model;\n- prepends ideal/common amplitude initialization only after compilation;\n- prints a per-edge 2Q-error audit before experiments.

Protocol
--------
1. Perform architecture search and VQC training under ideal PyTorch statevector simulation.
2. Use 80% search-train / 10% architecture-validation / 10% untouched test.
3. Refit a fresh VQC on train+validation (90%) after architecture selection.
4. Evaluate the frozen final model on the untouched test data:
   (a) ideal PyTorch simulation;
   (b) Qiskit Aer backend-informed noisy simulation.
5. Pool predictions across sequential ECG tasks within each seed.
6. Compute one Accuracy/BAcc/F1 per seed, then mean ± std across seeds.

Important
---------
This is backend-informed simulation, not physical-device execution.

Key stability updates
---------------------
- FakeSherbrooke is used only as a calibration/noise source.
- Simulation remains exactly 8 qubits wide.
- A connected 8-qubit physical region supplies gate/readout error rates.
- State preparation is ideal/common and is not decomposed into a noisy gate network.
- Backend-informed noise acts on the architecture-dependent VQC and readout.
- transpile optimization_level=1 and a stratified debug subset are used.
"""

import os
import gc
import time
import copy
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
try:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
except Exception:
    try:
        from qiskit.providers.fake_provider import FakeSherbrooke
    except Exception as exc:
        raise ImportError(
            "Could not import FakeSherbrooke. Install compatible Qiskit packages."
        ) from exc

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

try:
    import wfdb
except ImportError as exc:
    raise ImportError("Install with: pip install wfdb certifi") from exc


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL = torch.float32
COMPLEX = torch.complex64

MITDB_VERSION = "1.0.0"
MITDB_LOCAL_DIR = os.path.expanduser("~/Documents/Projects/qnn/data/mitdb")

ECG_RECORDS = (105, 106, 109, 114, 116, 119, 200, 201)
MAX_BEATS_PER_RECORD = 800
ECG_WINDOW_SEC = 0.6
INPUT_DIM = 256

AAMI_N = {"N", "L", "R", "e", "j"}
AAMI_V = {"V", "E"}

NUM_QUBITS = 8
NUM_CLASSES = 2
TT_MODES = (4, 16, 4)
TT_RANK = 3

DEPTH_CHOICES = (2, 3, 4)
ENT_PATTERNS = ("ring", "linear", "brick_even", "brick_odd")

SEARCH_STEPS = 10
CANDIDATES_PER_STEP = 3
SEARCH_INNER_EPOCHS = 8

FINAL_REFIT_EPOCHS = 40
BATCH_SIZE = 64

VQC_LR = 3e-3
POLICY_LR = 1e-3
WEIGHT_DECAY = 1e-5

MU_EWC = 0.5
ETA_KL = 0.01
ENTROPY_COEF = 0.002

LAMBDA_HW = 0.01

SEARCH_TRAIN_FRAC = 0.80
SEARCH_VAL_FRAC = 0.10
SPLIT_SEED_BASE = 5000

SEEDS = (11, 22, 33)

# Backend-informed settings
QISKIT_SHOTS = 1024
TRANSPILE_OPT_LEVEL = 1
TRANSPILE_SEED_BASE = 9000
SIMULATOR_SEED_BASE = 19000

# First run: True. Final manuscript: False and optionally QISKIT_SHOTS=2048.
DEBUG_BACKEND = True
DEBUG_NUM_TEST_SAMPLES = 8

# Backend calibration sanity controls.
# Values close to 1.0 are treated as unusable/disabled calibration edges.
INVALID_2Q_ERROR_THRESHOLD = 0.50
MAX_REGION_2Q_ERROR = 0.05

# FakeSherbrooke is used ONLY as a calibration/noise source.
BACKEND = FakeSherbrooke()

COMMON_PHYSICAL_LAYOUT: Optional[List[int]] = None
LOCAL_COUPLING = None
LOCAL_COUPLING_MAP = None
LOCAL_NOISE_MODEL = None
LOCAL_SIMULATOR = None
LOCAL_NOISE_METADATA = None


# =====================================================================
# 2. REPRODUCIBILITY
# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# 3. BACKEND-INFORMED LOCAL 8-QUBIT NOISE MODEL
# =====================================================================

def backend_name():
    value = getattr(BACKEND, "name", "fake_sherbrooke")
    return value() if callable(value) else str(value)


def backend_num_qubits():
    value = getattr(BACKEND, "num_qubits", None)
    if value is not None:
        return int(value)
    try:
        return int(BACKEND.configuration().num_qubits)
    except Exception:
        return 127


def get_backend_edges():
    """Return physical coupling edges without using the backend for simulation."""
    try:
        cm = BACKEND.target.build_coupling_map()
        return [(int(a), int(b)) for a, b in cm.get_edges()]
    except Exception:
        pass
    try:
        cm = BACKEND.coupling_map
        edges = cm.get_edges() if hasattr(cm, "get_edges") else cm
        return [(int(a), int(b)) for a, b in edges]
    except Exception:
        pass
    try:
        return [(int(a), int(b)) for a, b in BACKEND.configuration().coupling_map]
    except Exception as exc:
        raise RuntimeError("Unable to obtain FakeSherbrooke coupling map.") from exc


def _native_2q_error_for_edge(a, b):
    native_gate, native_qargs = supported_two_qubit_gate(a, b)
    if native_gate is None:
        return None, None, None

    p2 = extract_gate_error(
        native_gate,
        native_qargs,
        np.nan,
    )

    if not np.isfinite(p2):
        return native_gate, native_qargs, None

    return native_gate, native_qargs, float(p2)


def connected_region_from_backend(num_qubits=NUM_QUBITS):
    """
    Select one fixed connected physical region using calibration only.

    Edges with missing/pathological calibration are excluded. Candidate
    8-qubit regions are formed from edges with p2 <= MAX_REGION_2Q_ERROR.
    Among valid candidates, choose the one with the lowest induced mean p2.
    The selected region is reused for all methods, tasks, and random seeds.
    """
    undirected_edges = sorted(
        {tuple(sorted((int(a), int(b)))) for a, b in get_backend_edges()}
    )

    valid_edge_info = {}
    adjacency = {q: set() for q in range(backend_num_qubits())}
    pathological = []

    for a, b in undirected_edges:
        native_gate, native_qargs, p2 = _native_2q_error_for_edge(a, b)

        if p2 is None:
            continue

        if p2 >= INVALID_2Q_ERROR_THRESHOLD:
            pathological.append((a, b, native_gate, p2))
            continue

        if p2 > MAX_REGION_2Q_ERROR:
            continue

        valid_edge_info[(a, b)] = {
            "native_gate": native_gate,
            "native_qargs": native_qargs,
            "p2": p2,
        }

        adjacency[a].add(b)
        adjacency[b].add(a)

    starts = sorted(
        adjacency.keys(),
        key=lambda q: (-len(adjacency[q]), q),
    )

    candidates = []

    for start_q in starts:
        if not adjacency[start_q]:
            continue

        queue = [start_q]
        seen = {start_q}
        order = []

        while queue and len(order) < num_qubits:
            q = queue.pop(0)
            order.append(q)

            for r in sorted(
                adjacency[q],
                key=lambda x: (-len(adjacency[x]), x),
            ):
                if r not in seen:
                    seen.add(r)
                    queue.append(r)

        if len(order) != num_qubits:
            continue

        selected = set(order)
        induced_errors = []
        induced_edges = []

        for edge, info in valid_edge_info.items():
            a, b = edge
            if a in selected and b in selected:
                induced_edges.append(edge)
                induced_errors.append(info["p2"])

        # Check induced connectivity.
        local_adj = {q: set() for q in order}
        for a, b in induced_edges:
            local_adj[a].add(b)
            local_adj[b].add(a)

        reached = {order[0]}
        stack = [order[0]]

        while stack:
            q = stack.pop()
            for r in local_adj[q]:
                if r not in reached:
                    reached.add(r)
                    stack.append(r)

        if len(reached) != num_qubits or not induced_errors:
            continue

        score = (
            float(np.mean(induced_errors)),
            float(np.max(induced_errors)),
            -len(induced_edges),
            tuple(order),
        )

        candidates.append(
            (score, order, induced_edges)
        )

    if not candidates:
        raise RuntimeError(
            "No connected calibration-valid 8-qubit region was found. "
            f"Current MAX_REGION_2Q_ERROR={MAX_REGION_2Q_ERROR:.3f}."
        )

    candidates.sort(key=lambda item: item[0])
    best_score, best_order, best_edges = candidates[0]

    print(
        "Calibration-aware region selection: "
        f"mean p2={best_score[0]:.6f}, "
        f"max p2={best_score[1]:.6f}, "
        f"induced valid edges={len(best_edges)}"
    )

    if pathological:
        print(
            f"Excluded {len(pathological)} pathological 2Q calibration "
            f"edge(s) with p2 >= {INVALID_2Q_ERROR_THRESHOLD:.2f}."
        )
        for a, b, gate_name, p2 in pathological[:10]:
            print(
                f"  excluded physical=({a}, {b}) | "
                f"native={gate_name} | p2={p2:.6f}"
            )

    return list(best_order)


def target_instruction_properties(name, qargs):
    try:
        return BACKEND.target[name][tuple(qargs)]
    except Exception:
        return None


def extract_gate_error(name, qargs, fallback):
    """
    Read the raw backend gate-error calibration without clipping p~=1 to 0.999.
    Pathological values are filtered explicitly during region/noise construction.
    """
    props = target_instruction_properties(name, qargs)
    try:
        err = float(props.error)
        if np.isfinite(err) and err >= 0:
            return err
    except Exception:
        pass

    try:
        properties = BACKEND.properties()
        err = float(properties.gate_error(name, list(qargs)))
        if np.isfinite(err) and err >= 0:
            return err
    except Exception:
        pass

    return float(fallback)


def supported_two_qubit_gate(a, b):
    for name in ("ecr", "cx", "cz"):
        try:
            mapping = BACKEND.target[name]
            if (a, b) in mapping:
                return name, (a, b)
            if (b, a) in mapping:
                return name, (b, a)
        except Exception:
            continue
    return None, None


def extract_readout_probabilities(physical_qubit, fallback=0.02):
    """Return p01=P(1|0) and p10=P(0|1)."""
    p01 = p10 = None
    try:
        properties = BACKEND.properties()
        p01 = properties.qubit_property(physical_qubit, "prob_meas1_prep0")
        p10 = properties.qubit_property(physical_qubit, "prob_meas0_prep1")
        if isinstance(p01, tuple):
            p01 = p01[0]
        if isinstance(p10, tuple):
            p10 = p10[0]
    except Exception:
        pass

    if p01 is None or p10 is None:
        try:
            r = float(BACKEND.properties().readout_error(physical_qubit))
        except Exception:
            r = float(fallback)
        p01 = p10 = r

    p01 = min(max(float(p01), 0.0), 0.499)
    p10 = min(max(float(p10), 0.0), 0.499)
    return p01, p10


def build_local_backend_noise_model(physical_layout):
    """
    Build the local 8-qubit backend-informed NoiseModel.

    Only calibration-valid physical 2Q edges are retained in the local
    coupling map. An edge with p2 >= INVALID_2Q_ERROR_THRESHOLD is treated
    as unusable instead of simulating an almost-random 2Q gate.
    """
    model = NoiseModel()

    metadata = {
        "physical_layout": list(physical_layout),
        "one_qubit": {},
        "two_qubit": {},
        "rejected_two_qubit": {},
        "readout": {},
    }

    # 1Q errors.
    for local_q, physical_q in enumerate(physical_layout):
        p1 = extract_gate_error(
            "sx",
            (physical_q,),
            0.001,
        )

        if not np.isfinite(p1) or p1 < 0 or p1 >= 1:
            p1 = 0.001

        err1 = depolarizing_error(float(p1), 1)

        for opname in ("rx", "ry", "sx", "x"):
            try:
                model.add_quantum_error(
                    err1,
                    opname,
                    [local_q],
                )
            except Exception:
                pass

        metadata["one_qubit"][str(local_q)] = {
            "physical_qubit": int(physical_q),
            "p1": float(p1),
        }

    physical_edges = {
        tuple(sorted((a, b)))
        for a, b in get_backend_edges()
    }

    local_coupling = []

    # 2Q errors.
    for i in range(NUM_QUBITS):
        for j in range(i + 1, NUM_QUBITS):
            pa = int(physical_layout[i])
            pb = int(physical_layout[j])

            if tuple(sorted((pa, pb))) not in physical_edges:
                continue

            native_gate, native_qargs = supported_two_qubit_gate(pa, pb)

            if native_gate is None:
                metadata["rejected_two_qubit"][f"{i}-{j}"] = {
                    "physical_edge": (pa, pb),
                    "reason": "no supported native 2Q gate",
                }
                continue

            p2 = extract_gate_error(
                native_gate,
                native_qargs,
                np.nan,
            )

            if (
                not np.isfinite(p2)
                or p2 < 0
                or p2 >= INVALID_2Q_ERROR_THRESHOLD
            ):
                metadata["rejected_two_qubit"][f"{i}-{j}"] = {
                    "physical_edge": (pa, pb),
                    "native_gate": native_gate,
                    "p2": float(p2) if np.isfinite(p2) else np.nan,
                    "reason": "invalid/pathological calibration edge",
                }
                continue

            err2 = depolarizing_error(float(p2), 2)

            model.add_quantum_error(
                err2,
                "cx",
                [i, j],
            )
            model.add_quantum_error(
                err2,
                "cx",
                [j, i],
            )

            local_coupling.extend(
                [(i, j), (j, i)]
            )

            metadata["two_qubit"][f"{i}-{j}"] = {
                "physical_edge": (pa, pb),
                "native_gate": native_gate,
                "native_qargs": tuple(int(q) for q in native_qargs),
                "p2": float(p2),
            }

    # Verify local connectivity after filtering.
    local_adj = {q: set() for q in range(NUM_QUBITS)}
    for a, b in local_coupling:
        local_adj[a].add(b)

    reached = {0}
    stack = [0]

    while stack:
        q = stack.pop()
        for r in local_adj[q]:
            if r not in reached:
                reached.add(r)
                stack.append(r)

    if len(reached) != NUM_QUBITS:
        raise RuntimeError(
            "Selected physical region is disconnected after filtering "
            "pathological 2Q calibration edges."
        )

    # Readout errors.
    for local_q, physical_q in enumerate(physical_layout):
        p01, p10 = extract_readout_probabilities(physical_q)

        ro = ReadoutError(
            [
                [1.0 - p01, p01],
                [p10, 1.0 - p10],
            ]
        )

        model.add_readout_error(
            ro,
            [local_q],
        )

        metadata["readout"][str(local_q)] = {
            "physical_qubit": int(physical_q),
            "p_meas1_prep0": float(p01),
            "p_meas0_prep1": float(p10),
        }

    return model, local_coupling, metadata


def initialize_backend_informed_simulator():
    global COMMON_PHYSICAL_LAYOUT, LOCAL_COUPLING, LOCAL_COUPLING_MAP
    global LOCAL_NOISE_MODEL, LOCAL_SIMULATOR, LOCAL_NOISE_METADATA

    COMMON_PHYSICAL_LAYOUT = connected_region_from_backend(NUM_QUBITS)

    LOCAL_NOISE_MODEL, LOCAL_COUPLING, LOCAL_NOISE_METADATA = (
        build_local_backend_noise_model(COMMON_PHYSICAL_LAYOUT)
    )

    # Newer Qiskit versions require one explicit CouplingMap object here.
    # Passing a raw Python list can be interpreted as multiple coupling maps.
    LOCAL_COUPLING_MAP = CouplingMap(
        couplinglist=LOCAL_COUPLING
    )

    # Exactly eight simulated qubits: dense statevector is safe and simple.
    LOCAL_SIMULATOR = AerSimulator(
        method="statevector",
        noise_model=LOCAL_NOISE_MODEL,
    )

    print("Common physical layout:", COMMON_PHYSICAL_LAYOUT)
    print("Local simulated width:", NUM_QUBITS)
    print("Local directed coupling:", LOCAL_COUPLING)

    print("\nRetained backend-informed two-qubit errors:")
    for local_edge, info in LOCAL_NOISE_METADATA["two_qubit"].items():
        print(
            f"  local {local_edge:>3s} | "
            f"physical={info['physical_edge']} | "
            f"native={info['native_gate']} | "
            f"p2={info['p2']:.6f}"
        )

    if LOCAL_NOISE_METADATA["rejected_two_qubit"]:
        print("\nRejected induced two-qubit edges:")
        for local_edge, info in LOCAL_NOISE_METADATA[
            "rejected_two_qubit"
        ].items():
            print(
                f"  local {local_edge:>3s} | "
                f"physical={info.get('physical_edge')} | "
                f"p2={info.get('p2', np.nan)} | "
                f"reason={info.get('reason')}"
            )

    p2_values = [
        info["p2"]
        for info in LOCAL_NOISE_METADATA["two_qubit"].values()
    ]

    if p2_values:
        mean_p2 = float(np.mean(p2_values))
        max_p2 = float(np.max(p2_values))

        print(
            f"2Q-error audit: mean={mean_p2:.6f}, "
            f"max={max_p2:.6f}"
        )

        if mean_p2 > 0.10 or max_p2 > 0.20:
            print(
                "WARNING: unusually large extracted 2Q error detected. "
                "Inspect the per-edge values above before using these "
                "backend-informed results in the manuscript."
            )


# =====================================================================
# 4. MIT-BIH LOADING
# =====================================================================

def read_mitdb_record(record, max_retries=5, retry_wait=3.0):
    record = str(record)

    if os.path.isdir(MITDB_LOCAL_DIR):
        record_base = os.path.join(MITDB_LOCAL_DIR, record)
        required = (
            record_base + ".hea",
            record_base + ".dat",
            record_base + ".atr",
        )

        if all(os.path.exists(path) for path in required):
            try:
                signal, fields = wfdb.rdsamp(record_base)
                annotation = wfdb.rdann(record_base, "atr")
                print(f"[MIT-BIH] {record} loaded locally.")
                return signal, fields, annotation
            except Exception as exc:
                print(f"Local read failed for {record}: {exc}")

    pn_dir = f"mitdb/{MITDB_VERSION}"
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[MIT-BIH] Fetching {record}: {attempt}/{max_retries}")
            signal, fields = wfdb.rdsamp(record, pn_dir=pn_dir)
            annotation = wfdb.rdann(record, "atr", pn_dir=pn_dir)
            return signal, fields, annotation
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_wait * attempt)

    raise RuntimeError(
        f"Unable to load record {record}. Last error: {last_error}"
    )


# =====================================================================
# 5. ECG PREPROCESSING
# =====================================================================

def map_symbol_to_binary(symbol):
    if symbol in AAMI_N:
        return 0
    if symbol in AAMI_V:
        return 1
    return None


def choose_channel(signal_names):
    names = [name.upper() for name in signal_names]
    return names.index("MLII") if "MLII" in names else 0


def fft_bandpass(signal, fs, low=0.5, high=40.0):
    n = len(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.fft.rfft(signal)
    mask = (frequencies >= low) & (frequencies <= high)
    spectrum *= mask
    return np.fft.irfft(spectrum, n=n).astype(np.float32)


def beat_vector_256(segment, target_len=INPUT_DIM):
    x = np.asarray(segment, dtype=np.float32)

    source_grid = np.linspace(
        0.0, 1.0, len(x), endpoint=False, dtype=np.float32
    )
    target_grid = np.linspace(
        0.0, 1.0, target_len, endpoint=False, dtype=np.float32
    )

    x = np.interp(target_grid, source_grid, x).astype(np.float32)

    mean = float(x.mean())
    std = float(x.std() + 1e-6)

    x = (x - mean) / std
    x = np.clip(x, -5.0, 5.0)

    return x.astype(np.float32)


def load_record(
    record,
    max_beats=MAX_BEATS_PER_RECORD,
    window_sec=ECG_WINDOW_SEC,
    min_per_class=10,
):
    signal, fields, annotation = read_mitdb_record(record)

    channel = choose_channel(fields["sig_name"])
    fs = float(fields["fs"])

    x = fft_bandpass(signal[:, channel], fs)
    half_window = int(window_sec * fs)

    X, y = [], []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        label = map_symbol_to_binary(symbol)
        if label is None:
            continue

        start = sample - half_window
        end = sample + half_window

        if start < 0 or end >= len(x):
            continue

        X.append(beat_vector_256(x[start:end]))
        y.append(label)

        if max_beats is not None and len(X) >= max_beats:
            break

    if not X:
        raise RuntimeError(f"No usable beats for record {record}.")

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)

    counts = np.bincount(y, minlength=2)

    if counts[0] < min_per_class or counts[1] < min_per_class:
        raise RuntimeError(
            f"Record {record}: insufficient classes {counts.tolist()}."
        )

    return (
        torch.tensor(X, dtype=REAL),
        torch.tensor(y, dtype=torch.long),
    )


def load_tasks():
    tasks = []

    for record in ECG_RECORDS:
        X, y = load_record(record)
        counts = torch.bincount(y, minlength=2)

        print(
            f"Record {record}: "
            f"N={counts[0].item()}, V={counts[1].item()}"
        )

        tasks.append((record, X, y))

    return tasks


# =====================================================================
# 6. SPLITS
# =====================================================================

def stratified_split(
    y,
    train_frac=SEARCH_TRAIN_FRAC,
    val_frac=SEARCH_VAL_FRAC,
    seed=1234,
):
    rng = np.random.RandomState(seed)
    labels = y.cpu().numpy()

    train_indices, val_indices, test_indices = [], [], []

    for class_id in (0, 1):
        indices = np.where(labels == class_id)[0].copy()
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(np.floor(train_frac * n))
        n_val = int(np.floor(val_frac * n))

        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = min(n_val, n - n_train - 1)

        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        torch.tensor(train_indices, dtype=torch.long),
        torch.tensor(val_indices, dtype=torch.long),
        torch.tensor(test_indices, dtype=torch.long),
    )


def build_split_cache(tasks, seed):
    cache = {}

    for task_id, (record, _, y) in enumerate(tasks, start=1):
        split_seed = SPLIT_SEED_BASE + 100 * seed + task_id
        train_idx, val_idx, test_idx = stratified_split(
            y,
            seed=split_seed,
        )

        cache[record] = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }

    return cache


# =====================================================================
# 7. STANDARDIZATION
# =====================================================================

@dataclass
class StandardizationState:
    mean: torch.Tensor
    std: torch.Tensor


def fit_standardizer(X_train):
    return StandardizationState(
        mean=X_train.mean(dim=0),
        std=X_train.std(dim=0) + 1e-6,
    )


def standardize(X, state):
    return (X - state.mean) / state.std


# =====================================================================
# 8. TT REPRESENTATION
# =====================================================================

def normalize_state(x, eps=1e-8):
    return x / (torch.linalg.vector_norm(x) + eps)


def tt_svd_vector(x, modes=TT_MODES, max_rank=TT_RANK):
    tensor = x.reshape(*modes)

    cores = []
    rank_prev = 1
    remainder = tensor

    for k in range(len(modes) - 1):
        n_k = modes[k]

        matrix = remainder.reshape(rank_prev * n_k, -1)

        U, S, Vh = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )

        rank = min(max_rank, U.shape[1])

        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        cores.append(U.reshape(rank_prev, n_k, rank))

        remainder = S.unsqueeze(1) * Vh
        rank_prev = rank

        remainder = remainder.reshape(
            rank_prev,
            *modes[k + 1:],
        )

    cores.append(
        remainder.reshape(
            rank_prev,
            modes[-1],
            1,
        )
    )

    output = cores[0]

    for core in cores[1:]:
        output = torch.einsum(
            "...a,aib->...ib",
            output,
            core,
        )

    return (
        output.squeeze(0)
        .squeeze(-1)
        .reshape(-1)
    )


def tt_encode_dataset(X, rank=TT_RANK):
    encoded, fidelities = [], []

    with torch.no_grad():
        for x in X:
            exact = normalize_state(x)

            approximation = tt_svd_vector(
                x,
                max_rank=rank,
            )

            approximation = normalize_state(approximation)

            overlap = torch.dot(exact, approximation)
            fidelity = torch.abs(overlap) ** 2

            encoded.append(approximation)
            fidelities.append(fidelity)

    return torch.stack(encoded), torch.stack(fidelities)


# =====================================================================
# 9. PREPARE TASK
# =====================================================================

def prepare_task(X_raw, y, split_info):
    train_idx = split_info["train_idx"]
    val_idx = split_info["val_idx"]
    test_idx = split_info["test_idx"]

    X_train_raw, y_train = X_raw[train_idx], y[train_idx]
    X_val_raw, y_val = X_raw[val_idx], y[val_idx]
    X_test_raw, y_test = X_raw[test_idx], y[test_idx]

    standardizer = fit_standardizer(X_train_raw)

    X_train_std = standardize(X_train_raw, standardizer)
    X_val_std = standardize(X_val_raw, standardizer)
    X_test_std = standardize(X_test_raw, standardizer)

    X_train, F_train = tt_encode_dataset(X_train_std)
    X_val, F_val = tt_encode_dataset(X_val_std)
    X_test, F_test = tt_encode_dataset(X_test_std)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "fidelity": float(
            torch.cat([F_train, F_val, F_test]).mean()
        ),
    }


# =====================================================================
# 10. IDEAL PYTORCH GATES
# =====================================================================

def rx(theta):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)

    return torch.stack(
        [
            torch.stack([c, -1j * s]),
            torch.stack([-1j * s, c]),
        ]
    ).to(COMPLEX)


def ry(theta):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)

    return torch.stack(
        [
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ]
    ).to(COMPLEX)


def rz(theta):
    a = torch.exp(-0.5j * theta)
    b = torch.exp(0.5j * theta)
    zero = torch.zeros_like(a)

    return torch.stack(
        [
            torch.stack([a, zero]),
            torch.stack([zero, b]),
        ]
    ).to(COMPLEX)


def apply_1q(state, gate, wire):
    batch = state.shape[0]

    psi = state.reshape(batch, *([2] * NUM_QUBITS))

    axis = wire + 1

    permutation = (
        [0]
        + [
            i
            for i in range(1, NUM_QUBITS + 1)
            if i != axis
        ]
        + [axis]
    )

    psi = psi.permute(*permutation).contiguous()

    old_shape = psi.shape

    psi = psi.reshape(-1, 2)

    psi = torch.einsum(
        "bi,ji->bj",
        psi,
        gate,
    )

    psi = psi.reshape(old_shape)

    inverse = np.argsort(permutation)

    psi = psi.permute(*inverse).contiguous()

    return psi.reshape(batch, -1)


def apply_cnot(state, control, target):
    dimension = 2 ** NUM_QUBITS

    indices = torch.arange(
        dimension,
        device=state.device,
    )

    control_bit = (
        indices
        >> (NUM_QUBITS - 1 - control)
    ) & 1

    target_mask = (
        1
        << (NUM_QUBITS - 1 - target)
    )

    mapped = torch.where(
        control_bit.bool(),
        indices ^ target_mask,
        indices,
    )

    return state[:, mapped]


def z_expectations(state):
    probabilities = state.abs() ** 2

    indices = torch.arange(
        2 ** NUM_QUBITS,
        device=state.device,
    )

    outputs = []

    for qubit in range(NUM_QUBITS):
        bit = (
            indices
            >> (NUM_QUBITS - 1 - qubit)
        ) & 1

        sign = 1.0 - 2.0 * bit.float()

        outputs.append(
            torch.sum(
                probabilities * sign.unsqueeze(0),
                dim=1,
            )
        )

    return torch.stack(outputs, dim=1)


# =====================================================================
# 11. ARCHITECTURE
# =====================================================================

@dataclass
class Architecture:
    depth: int
    patterns: Tuple[str, ...]


def edges_for_pattern(pattern):
    if pattern == "ring":
        return [
            (q, (q + 1) % NUM_QUBITS)
            for q in range(NUM_QUBITS)
        ]

    if pattern == "linear":
        return [
            (q, q + 1)
            for q in range(NUM_QUBITS - 1)
        ]

    if pattern == "brick_even":
        return [
            (q, q + 1)
            for q in range(0, NUM_QUBITS - 1, 2)
        ]

    if pattern == "brick_odd":
        edges = [
            (q, q + 1)
            for q in range(1, NUM_QUBITS - 1, 2)
        ]
        edges.append((NUM_QUBITS - 1, 0))
        return edges

    raise ValueError(f"Unknown entanglement pattern: {pattern}")


def architecture_stats(architecture):
    n1 = architecture.depth * NUM_QUBITS * 3

    n2 = sum(
        len(edges_for_pattern(pattern))
        for pattern in architecture.patterns
    )

    return {
        "depth": architecture.depth,
        "n1": n1,
        "n2": n2,
    }


def naive_architecture():
    return Architecture(
        depth=4,
        patterns=("ring", "ring", "ring", "ring"),
    )


# =====================================================================
# 12. VQC
# =====================================================================

class VQC(nn.Module):
    def __init__(self, architecture):
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

    def forward(self, amplitudes):
        state = amplitudes.to(
            DEVICE,
            dtype=COMPLEX,
        )

        for layer in range(self.architecture.depth):
            for qubit in range(NUM_QUBITS):
                state = apply_1q(
                    state,
                    rx(self.theta[layer, qubit, 0]),
                    qubit,
                )
                state = apply_1q(
                    state,
                    ry(self.theta[layer, qubit, 1]),
                    qubit,
                )
                state = apply_1q(
                    state,
                    rz(self.theta[layer, qubit, 2]),
                    qubit,
                )

            pattern = self.architecture.patterns[layer]

            for control, target in edges_for_pattern(pattern):
                state = apply_cnot(
                    state,
                    control,
                    target,
                )

        return z_expectations(state)[:, :NUM_CLASSES]


# =====================================================================
# 13. QAS POLICY
# =====================================================================

class QASPolicy(nn.Module):
    def __init__(
        self,
        feature_dim=INPUT_DIM,
        d_model=64,
        nhead=8,
        num_layers=2,
    ):
        super().__init__()

        max_depth = max(DEPTH_CHOICES)

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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
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

    def forward(self, context):
        task_embedding = (
            self.feature_proj(context)
            .mean(dim=0, keepdim=True)
        )

        hidden = (
            self.layer_tokens
            + task_embedding.unsqueeze(1)
        )

        hidden = self.encoder(hidden)

        pooled = hidden.mean(dim=1)

        return (
            self.depth_head(pooled).squeeze(0),
            self.pattern_head(hidden).squeeze(0),
        )


# =====================================================================
# 14. ARCHITECTURE SAMPLING
# =====================================================================

@dataclass
class PolicySample:
    architecture: Architecture
    log_prob: torch.Tensor
    entropy: torch.Tensor


def sample_architecture(policy, context):
    depth_logits, pattern_logits = policy(context)

    depth_dist = torch.distributions.Categorical(
        logits=depth_logits
    )

    depth_index = depth_dist.sample()

    depth = DEPTH_CHOICES[int(depth_index.item())]

    log_prob = depth_dist.log_prob(depth_index)
    entropy = depth_dist.entropy()

    patterns = []

    for layer in range(depth):
        dist = torch.distributions.Categorical(
            logits=pattern_logits[layer]
        )

        index = dist.sample()

        patterns.append(
            ENT_PATTERNS[int(index.item())]
        )

        log_prob = log_prob + dist.log_prob(index)
        entropy = entropy + dist.entropy()

    entropy = entropy / (depth + 1)

    return PolicySample(
        architecture=Architecture(
            depth=depth,
            patterns=tuple(patterns),
        ),
        log_prob=log_prob,
        entropy=entropy,
    )


# =====================================================================
# 15. EWC
# =====================================================================

@dataclass
class EWCState:
    means: Dict[str, torch.Tensor]
    fisher: Dict[str, torch.Tensor]


def estimate_fisher(policy, context, num_samples=24):
    fisher = {
        name: torch.zeros_like(parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }

    for _ in range(num_samples):
        policy.zero_grad(set_to_none=True)

        sample = sample_architecture(
            policy,
            context,
        )

        (-sample.log_prob).backward()

        for name, parameter in policy.named_parameters():
            if parameter.grad is not None:
                fisher[name] += (
                    parameter.grad.detach() ** 2
                ) / num_samples

    means = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }

    return EWCState(
        means=means,
        fisher=fisher,
    )


def ewc_penalty(policy, state):
    if state is None:
        return torch.tensor(
            0.0,
            device=DEVICE,
        )

    penalty = torch.tensor(
        0.0,
        device=DEVICE,
    )

    for name, parameter in policy.named_parameters():
        if name not in state.fisher:
            continue

        penalty = penalty + torch.sum(
            state.fisher[name]
            * (parameter - state.means[name]) ** 2
        )

    return 0.5 * penalty


# =====================================================================
# 16. KL
# =====================================================================

def categorical_kl(current_logits, reference_logits):
    p = torch.softmax(current_logits, dim=-1)
    log_p = torch.log_softmax(current_logits, dim=-1)
    log_q = torch.log_softmax(reference_logits, dim=-1)

    return torch.sum(
        p * (log_p - log_q),
        dim=-1,
    )


def policy_kl(policy, reference_policy, context):
    if reference_policy is None:
        return torch.tensor(
            0.0,
            device=DEVICE,
        )

    depth_logits, pattern_logits = policy(context)

    with torch.no_grad():
        reference_depth, reference_pattern = (
            reference_policy(context)
        )

    return (
        categorical_kl(
            depth_logits,
            reference_depth,
        )
        +
        categorical_kl(
            pattern_logits,
            reference_pattern,
        ).mean()
    )


# =====================================================================
# 17. TRAINING
# =====================================================================

def class_weighted_loss(y_train):
    counts = torch.bincount(
        y_train,
        minlength=2,
    ).float()

    weights = (
        counts.sum()
        / (2.0 * counts.clamp_min(1))
    )

    weights = torch.clamp(weights, max=4.0)

    return nn.CrossEntropyLoss(
        weight=weights.to(DEVICE)
    )


def train_vqc(
    architecture,
    X_train,
    y_train,
    epochs,
):
    model = VQC(architecture).to(DEVICE)

    criterion = class_weighted_loss(y_train)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=VQC_LR,
        weight_decay=WEIGHT_DECAY,
    )

    for _ in range(epochs):
        permutation = torch.randperm(len(X_train))

        for start in range(
            0,
            len(X_train),
            BATCH_SIZE,
        ):
            index = permutation[
                start:start + BATCH_SIZE
            ]

            xb = X_train[index].to(DEVICE)
            yb = y_train[index].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

    return model


# =====================================================================
# 18. IDEAL PREDICTION / METRICS
# =====================================================================

def predict_ideal(model, X, y):
    model.eval()

    truth, predictions = [], []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            xb = X[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            logits = model(xb)

            pred = (
                logits.argmax(dim=1)
                .cpu()
            )

            predictions.extend(pred.tolist())

            truth.extend(
                y[
                    start:start + BATCH_SIZE
                ].tolist()
            )

    return (
        np.asarray(truth, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
    )


def metrics_from_predictions(truth, predictions):
    return {
        "acc": accuracy_score(truth, predictions),
        "bAcc": balanced_accuracy_score(truth, predictions),
        "F1": f1_score(
            truth,
            predictions,
            zero_division=0,
        ),
    }


def evaluate_ideal(model, X, y):
    true, pred = predict_ideal(model, X, y)
    return metrics_from_predictions(true, pred)


# =====================================================================
# 19. REWARD
# =====================================================================

def predictive_score(metrics):
    return (
        0.70 * metrics["bAcc"]
        + 0.30 * metrics["F1"]
    )


def hardware_cost(architecture):
    stats = architecture_stats(architecture)
    reference_n2 = 4 * NUM_QUBITS
    return stats["n2"] / reference_n2


def architecture_reward(metrics, architecture):
    return (
        predictive_score(metrics)
        - LAMBDA_HW * hardware_cost(architecture)
    )


# =====================================================================
# 20. CANDIDATE EVALUATION
# =====================================================================

@dataclass
class CandidateResult:
    architecture: Architecture
    reward: float
    val_metrics: Dict[str, float]
    log_prob: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None


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

    metrics = evaluate_ideal(
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
        reward=float(reward),
        val_metrics=metrics,
    )


# =====================================================================
# 21. SEARCH
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
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=POLICY_LR,
    )

    context = X_train[
        :min(192, len(X_train))
    ].to(DEVICE)

    baseline = None
    best = None

    for _ in range(SEARCH_STEPS):
        candidates = []

        for _ in range(CANDIDATES_PER_STEP):
            sample = sample_architecture(
                policy,
                context,
            )

            result = evaluate_candidate(
                sample.architecture,
                X_train,
                y_train,
                X_val,
                y_val,
            )

            result.log_prob = sample.log_prob
            result.entropy = sample.entropy

            candidates.append(result)

            if best is None or result.reward > best.reward:
                best = result

        rewards = np.asarray(
            [candidate.reward for candidate in candidates],
            dtype=np.float32,
        )

        mean_reward = float(rewards.mean())

        if baseline is None:
            baseline = mean_reward

        advantages = torch.tensor(
            rewards - baseline,
            dtype=REAL,
            device=DEVICE,
        )

        log_probs = torch.stack(
            [candidate.log_prob for candidate in candidates]
        )

        entropy = torch.stack(
            [candidate.entropy for candidate in candidates]
        ).mean()

        reinforce_loss = -torch.mean(
            advantages.detach() * log_probs
        )

        if use_ewc and ewc_state is not None:
            L_ewc = ewc_penalty(
                policy,
                ewc_state,
            )
        else:
            L_ewc = torch.tensor(
                0.0,
                device=DEVICE,
            )

        if use_kl and reference_policy is not None:
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

        loss = (
            reinforce_loss
            + MU_EWC * L_ewc
            + ETA_KL * L_kl
            - ENTROPY_COEF * entropy
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            1.0,
        )

        optimizer.step()

        baseline = 0.9 * baseline + 0.1 * mean_reward

    return best, context


# =====================================================================
# 22. FINAL REFIT
# =====================================================================

def final_refit_model(
    architecture,
    X_train,
    y_train,
    X_val,
    y_val,
):
    X_final = torch.cat(
        [X_train, X_val],
        dim=0,
    )

    y_final = torch.cat(
        [y_train, y_val],
        dim=0,
    )

    return train_vqc(
        architecture,
        X_final,
        y_final,
        FINAL_REFIT_EPOCHS,
    )


# =====================================================================
# 23. PYTORCH -> QISKIT AMPLITUDE ORDER
# =====================================================================

def pytorch_to_qiskit_amplitudes(amplitudes):
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    amplitudes /= np.linalg.norm(amplitudes) + 1e-15
    tensor = amplitudes.reshape(*([2] * NUM_QUBITS))
    tensor = np.transpose(tensor, axes=list(reversed(range(NUM_QUBITS))))
    return tensor.reshape(-1)


# =====================================================================
# 24. QISKIT VQC TEMPLATE
# =====================================================================

def append_vqc_body(qc, model):
    theta = model.theta.detach().cpu().numpy()
    architecture = model.architecture
    for layer in range(architecture.depth):
        for qubit in range(NUM_QUBITS):
            qc.rx(float(theta[layer, qubit, 0]), qubit)
            qc.ry(float(theta[layer, qubit, 1]), qubit)
            qc.rz(float(theta[layer, qubit, 2]), qubit)
        for control, target in edges_for_pattern(architecture.patterns[layer]):
            qc.cx(control, target)


def build_qiskit_vqc_measurement_template(model):
    """Architecture-dependent VQC plus measurement; no state preparation."""
    qc = QuantumCircuit(NUM_QUBITS, NUM_CLASSES)
    append_vqc_body(qc, model)
    for qubit in range(NUM_CLASSES):
        qc.measure(qubit, qubit)
    return qc


def build_qiskit_vqc_only(model):
    qc = QuantumCircuit(NUM_QUBITS)
    append_vqc_body(qc, model)
    return qc


def transpile_local(qc, seed):
    """
    Transpile only to the selected local 8-qubit connectivity.

    FakeSherbrooke is deliberately NOT passed as ``backend``. This prevents
    127-wire expansion and avoids the Runtime backend-transpiler plugin path.

    LOCAL_COUPLING_MAP is an explicit qiskit.transpiler.CouplingMap, which
    fixes the raw-list coupling_map API error seen in newer Qiskit releases.
    """
    if LOCAL_COUPLING_MAP is None:
        raise RuntimeError(
            "LOCAL_COUPLING_MAP is not initialized. "
            "Call initialize_backend_informed_simulator() first."
        )

    tqc = transpile(
        qc,
        basis_gates=[
            "rx",
            "ry",
            "rz",
            "cx",
            "reset",
            "measure",
        ],
        coupling_map=LOCAL_COUPLING_MAP,
        initial_layout=list(range(NUM_QUBITS)),
        optimization_level=TRANSPILE_OPT_LEVEL,
        seed_transpiler=seed,
    )

    if tqc.num_qubits != NUM_QUBITS:
        raise RuntimeError(
            f"Expected local width {NUM_QUBITS}, "
            f"got {tqc.num_qubits}."
        )

    return tqc


def make_sample_circuit(amplitudes, transpiled_template):
    """
    Prepend an ideal Aer initialize instruction AFTER transpiling the VQC template.

    This is the key change that prevents arbitrary amplitude state preparation from
    being decomposed into hundreds of noisy two-qubit gates.  The initialize
    instruction has no error entry in LOCAL_NOISE_MODEL, whereas subsequent VQC
    operations and measurements do.
    """
    qc = QuantumCircuit(NUM_QUBITS, NUM_CLASSES)
    qc.initialize(
        pytorch_to_qiskit_amplitudes(amplitudes),
        list(range(NUM_QUBITS)),
    )
    qc.compose(transpiled_template, inplace=True)
    return qc


# =====================================================================
# 25. RESOURCE COUNTING
# =====================================================================

def transpiled_two_qubit_count(circuit):
    return sum(
        1 for instruction in circuit.data
        if instruction.operation.num_qubits == 2
    )


def transpiled_one_qubit_count(circuit):
    return sum(
        1 for instruction in circuit.data
        if instruction.operation.num_qubits == 1
        and instruction.operation.name not in ("measure", "reset")
    )


def transpiled_resource_stats(circuit):
    return {
        "depth": int(circuit.depth()),
        "n1": int(transpiled_one_qubit_count(circuit)),
        "n2": int(transpiled_two_qubit_count(circuit)),
    }


def get_backend_vqc_resource_stats(model, *, seed, task_id, method_index):
    qc = build_qiskit_vqc_only(model)
    transpiler_seed = (
        TRANSPILE_SEED_BASE + 1000 * seed + 100 * task_id + 10 * method_index
    )
    tqc = transpile_local(qc, transpiler_seed)
    stats = transpiled_resource_stats(tqc)
    del qc, tqc
    gc.collect()
    return stats


# =====================================================================
# 26. COUNTS -> LOGITS
# =====================================================================

def counts_to_logits(counts):
    total = float(sum(counts.values()))
    if total <= 0:
        raise RuntimeError("Qiskit returned zero counts.")
    z_values = np.zeros(NUM_CLASSES, dtype=np.float64)
    for bitstring, count in counts.items():
        bits = list(reversed(bitstring.replace(" ", "")))
        if len(bits) < NUM_CLASSES:
            raise RuntimeError(f"Unexpected count key: {bitstring}")
        for qubit in range(NUM_CLASSES):
            bit = int(bits[qubit])
            z_values[qubit] += (1.0 if bit == 0 else -1.0) * count
    return z_values / total


# =====================================================================
# 27. BACKEND-INFORMED PREDICTION
# =====================================================================

def predict_backend_informed(
    model,
    X_test,
    y_test,
    *,
    seed,
    task_id,
    method_index,
):
    if LOCAL_SIMULATOR is None:
        raise RuntimeError("Backend-informed simulator has not been initialized.")

    predictions, truth = [], []
    depths, n1_values, n2_values = [], [], []

    transpiler_seed = (
        TRANSPILE_SEED_BASE + 1000 * seed + 100 * task_id + 10 * method_index
    )

    # Compile the VQC+measurement template exactly once for this trained model/task.
    # No per-sample transpilation is performed.
    logical_template = build_qiskit_vqc_measurement_template(model)
    transpiled_template = transpile_local(logical_template, transpiler_seed)
    template_stats = transpiled_resource_stats(transpiled_template)

    print(f"    Backend-informed evaluation: {len(X_test)} samples")
    print(f"      transpiled width: {transpiled_template.num_qubits}")
    print(f"      VQC+measurement depth: {template_stats['depth']}")
    print(f"      VQC transpiled #2Q: {template_stats['n2']}")

    for sample_index in range(len(X_test)):
        if sample_index % 10 == 0:
            print(f"      sample {sample_index}/{len(X_test)}")

        amplitude = X_test[sample_index].detach().cpu().numpy()
        qc = make_sample_circuit(amplitude, transpiled_template)

        if qc.num_qubits != NUM_QUBITS:
            raise RuntimeError(
                f"Executable noisy circuit unexpectedly has "
                f"{qc.num_qubits} qubits; expected {NUM_QUBITS}."
            )

        simulator_seed = (
            SIMULATOR_SEED_BASE
            + 100000 * seed
            + 1000 * task_id
            + 100 * method_index
            + sample_index
        )

        result = LOCAL_SIMULATOR.run(
            qc,
            shots=QISKIT_SHOTS,
            seed_simulator=simulator_seed,
        ).result()
        counts = result.get_counts(0)
        logits = counts_to_logits(counts)

        predictions.append(int(np.argmax(logits)))
        truth.append(int(y_test[sample_index].item()))
        depths.append(template_stats["depth"])
        n1_values.append(template_stats["n1"])
        n2_values.append(template_stats["n2"])

        del qc, result, counts
        if (sample_index + 1) % 20 == 0:
            gc.collect()

    del logical_template, transpiled_template
    gc.collect()

    return {
        "y_true": np.asarray(truth, dtype=np.int64),
        "y_pred": np.asarray(predictions, dtype=np.int64),
        "total_transpiled_depth": float(np.mean(depths)),
        "total_transpiled_n1": float(np.mean(n1_values)),
        "total_transpiled_n2": float(np.mean(n2_values)),
    }


# =====================================================================
# 28. FINAL MODEL EVALUATION
# =====================================================================

def evaluate_final_model(
    model,
    X_test,
    y_test,
    *,
    seed,
    task_id,
    method_index,
):
    ideal_true, ideal_pred = predict_ideal(model, X_test, y_test)
    ideal_metrics = metrics_from_predictions(ideal_true, ideal_pred)

    vqc_backend_stats = get_backend_vqc_resource_stats(
        model, seed=seed, task_id=task_id, method_index=method_index
    )

    qiskit_result = predict_backend_informed(
        model,
        X_test,
        y_test,
        seed=seed,
        task_id=task_id,
        method_index=method_index,
    )
    noisy_metrics = metrics_from_predictions(
        qiskit_result["y_true"], qiskit_result["y_pred"]
    )

    return {
        "ideal_true": ideal_true,
        "ideal_pred": ideal_pred,
        "noisy_true": qiskit_result["y_true"],
        "noisy_pred": qiskit_result["y_pred"],
        "ideal_acc": ideal_metrics["acc"],
        "ideal_bAcc": ideal_metrics["bAcc"],
        "ideal_F1": ideal_metrics["F1"],
        "noisy_acc": noisy_metrics["acc"],
        "noisy_bAcc": noisy_metrics["bAcc"],
        "noisy_F1": noisy_metrics["F1"],
        "vqc_transpiled_depth": vqc_backend_stats["depth"],
        "vqc_transpiled_n1": vqc_backend_stats["n1"],
        "vqc_transpiled_n2": vqc_backend_stats["n2"],
        "total_transpiled_depth": qiskit_result["total_transpiled_depth"],
        "total_transpiled_n1": qiskit_result["total_transpiled_n1"],
        "total_transpiled_n2": qiskit_result["total_transpiled_n2"],
    }


# =====================================================================
# 29. STRATIFIED DEBUG SUBSET
# =====================================================================

def backend_subset(data, *, seed, task_id):
    X_test, y_test = data["X_test"], data["y_test"]
    if not DEBUG_BACKEND:
        return X_test, y_test

    n = min(DEBUG_NUM_TEST_SAMPLES, len(y_test))
    if n >= len(y_test):
        return X_test, y_test

    rng = np.random.RandomState(70000 + 1000 * seed + task_id)
    labels = y_test.cpu().numpy()
    selected = []

    for cls in np.unique(labels):
        ids = np.where(labels == cls)[0]
        if len(ids) > 0:
            selected.append(int(rng.choice(ids)))

    remaining = [i for i in range(len(y_test)) if i not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, n - len(selected))])
    selected = selected[:n]

    idx = torch.tensor(selected, dtype=torch.long)
    return X_test[idx], y_test[idx]


# =====================================================================
# 30. RUN ONE SEED
# =====================================================================

def run_seed(tasks, seed):
    split_cache = build_split_cache(
        tasks,
        seed,
    )

    set_seed(seed)
    policy_nocl = QASPolicy().to(DEVICE)

    set_seed(seed)
    policy_cl = QASPolicy().to(DEVICE)

    ewc_state = None
    reference_policy = None

    methods = (
        "Naive-VQC",
        "QAS-No-CL",
        "CL-QAS",
    )

    pooled = {
        method: {
            "ideal_true": [],
            "ideal_pred": [],
            "noisy_true": [],
            "noisy_pred": [],

            "logical_depth": [],
            "logical_n2": [],

            "vqc_transpiled_depth": [],
            "vqc_transpiled_n1": [],
            "vqc_transpiled_n2": [],

            "total_transpiled_depth": [],
            "total_transpiled_n1": [],
            "total_transpiled_n2": [],
        }
        for method in methods
    }

    task_rows = []

    for task_id, (record, X_raw, y) in enumerate(
        tasks,
        start=1,
    ):
        print("\n" + "=" * 110)
        print(
            f"Seed={seed} | "
            f"Task={task_id} | "
            f"Record={record}"
        )
        print("=" * 110)

        data = prepare_task(
            X_raw,
            y,
            split_cache[record],
        )

        X_backend_test, y_backend_test = (
            backend_subset(data, seed=seed, task_id=task_id)
        )

        if DEBUG_BACKEND:
            print(
                f"[DEBUG] Backend evaluation uses "
                f"{len(X_backend_test)} test samples."
            )

        # -------------------------------------------------------------
        # A. Naive-VQC
        # -------------------------------------------------------------

        naive_arch = naive_architecture()

        naive_model = final_refit_model(
            naive_arch,
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
        )

        naive_eval = evaluate_final_model(
            naive_model,
            X_backend_test,
            y_backend_test,
            seed=seed,
            task_id=task_id,
            method_index=0,
        )

        # -------------------------------------------------------------
        # B. QAS-No-CL
        # -------------------------------------------------------------

        qas_candidate, _ = search_task(
            policy_nocl,
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            ewc_state=None,
            reference_policy=None,
            use_ewc=False,
            use_kl=False,
        )

        qas_model = final_refit_model(
            qas_candidate.architecture,
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
        )

        qas_eval = evaluate_final_model(
            qas_model,
            X_backend_test,
            y_backend_test,
            seed=seed,
            task_id=task_id,
            method_index=1,
        )

        # -------------------------------------------------------------
        # C. CL-QAS
        # -------------------------------------------------------------

        cl_candidate, context = search_task(
            policy_cl,
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            ewc_state=ewc_state,
            reference_policy=reference_policy,
            use_ewc=True,
            use_kl=True,
        )

        cl_model = final_refit_model(
            cl_candidate.architecture,
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
        )

        cl_eval = evaluate_final_model(
            cl_model,
            X_backend_test,
            y_backend_test,
            seed=seed,
            task_id=task_id,
            method_index=2,
        )

        outputs = {
            "Naive-VQC": (
                naive_arch,
                naive_eval,
            ),
            "QAS-No-CL": (
                qas_candidate.architecture,
                qas_eval,
            ),
            "CL-QAS": (
                cl_candidate.architecture,
                cl_eval,
            ),
        }

        for method, (
            architecture,
            evaluation,
        ) in outputs.items():
            logical_stats = architecture_stats(
                architecture
            )

            pooled[method]["ideal_true"].extend(
                evaluation["ideal_true"].tolist()
            )
            pooled[method]["ideal_pred"].extend(
                evaluation["ideal_pred"].tolist()
            )
            pooled[method]["noisy_true"].extend(
                evaluation["noisy_true"].tolist()
            )
            pooled[method]["noisy_pred"].extend(
                evaluation["noisy_pred"].tolist()
            )

            pooled[method]["logical_depth"].append(
                logical_stats["depth"]
            )
            pooled[method]["logical_n2"].append(
                logical_stats["n2"]
            )

            pooled[method][
                "vqc_transpiled_depth"
            ].append(
                evaluation[
                    "vqc_transpiled_depth"
                ]
            )
            pooled[method][
                "vqc_transpiled_n1"
            ].append(
                evaluation[
                    "vqc_transpiled_n1"
                ]
            )
            pooled[method][
                "vqc_transpiled_n2"
            ].append(
                evaluation[
                    "vqc_transpiled_n2"
                ]
            )

            pooled[method][
                "total_transpiled_depth"
            ].append(
                evaluation[
                    "total_transpiled_depth"
                ]
            )
            pooled[method][
                "total_transpiled_n1"
            ].append(
                evaluation[
                    "total_transpiled_n1"
                ]
            )
            pooled[method][
                "total_transpiled_n2"
            ].append(
                evaluation[
                    "total_transpiled_n2"
                ]
            )

            task_rows.append(
                {
                    "seed": seed,
                    "task": task_id,
                    "record": record,
                    "method": method,
                    "debug_backend": DEBUG_BACKEND,
                    "backend_test_n": len(X_backend_test),

                    "logical_depth":
                    logical_stats["depth"],

                    "logical_n2":
                    logical_stats["n2"],

                    "vqc_transpiled_depth":
                    evaluation[
                        "vqc_transpiled_depth"
                    ],

                    "vqc_transpiled_n1":
                    evaluation[
                        "vqc_transpiled_n1"
                    ],

                    "vqc_transpiled_n2":
                    evaluation[
                        "vqc_transpiled_n2"
                    ],

                    "total_transpiled_depth":
                    evaluation[
                        "total_transpiled_depth"
                    ],

                    "total_transpiled_n1":
                    evaluation[
                        "total_transpiled_n1"
                    ],

                    "total_transpiled_n2":
                    evaluation[
                        "total_transpiled_n2"
                    ],

                    "ideal_acc":
                    evaluation["ideal_acc"],

                    "ideal_bAcc":
                    evaluation["ideal_bAcc"],

                    "ideal_F1":
                    evaluation["ideal_F1"],

                    "noisy_acc":
                    evaluation["noisy_acc"],

                    "noisy_bAcc":
                    evaluation["noisy_bAcc"],

                    "noisy_F1":
                    evaluation["noisy_F1"],
                }
            )

            print(
                f"{method:12s} | "
                f"logical #2Q={logical_stats['n2']:2d} | "
                f"VQC transpiled #2Q="
                f"{evaluation['vqc_transpiled_n2']:.1f} | "
                f"total #2Q="
                f"{evaluation['total_transpiled_n2']:.1f} | "
                f"noisy BAcc="
                f"{evaluation['noisy_bAcc']:.4f} | "
                f"noisy F1="
                f"{evaluation['noisy_F1']:.4f}"
            )

        # Continual policy state.
        ewc_state = estimate_fisher(
            policy_cl,
            context,
        )

        reference_policy = (
            copy.deepcopy(policy_cl)
            .eval()
        )

        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)

        del naive_model, qas_model, cl_model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Pooled per-seed results
    # -------------------------------------------------------------

    seed_rows = []

    for method in methods:
        ideal_true = np.asarray(
            pooled[method]["ideal_true"],
            dtype=np.int64,
        )
        ideal_pred = np.asarray(
            pooled[method]["ideal_pred"],
            dtype=np.int64,
        )

        noisy_true = np.asarray(
            pooled[method]["noisy_true"],
            dtype=np.int64,
        )
        noisy_pred = np.asarray(
            pooled[method]["noisy_pred"],
            dtype=np.int64,
        )

        ideal_metrics = metrics_from_predictions(
            ideal_true,
            ideal_pred,
        )

        noisy_metrics = metrics_from_predictions(
            noisy_true,
            noisy_pred,
        )

        seed_rows.append(
            {
                "seed": seed,
                "method": method,
                "debug_backend": DEBUG_BACKEND,
                "pooled_test_n": len(noisy_true),

                "ideal_acc":
                ideal_metrics["acc"],

                "ideal_bAcc":
                ideal_metrics["bAcc"],

                "ideal_F1":
                ideal_metrics["F1"],

                "noisy_acc":
                noisy_metrics["acc"],

                "noisy_bAcc":
                noisy_metrics["bAcc"],

                "noisy_F1":
                noisy_metrics["F1"],

                "acc_drop":
                ideal_metrics["acc"]
                - noisy_metrics["acc"],

                "bAcc_drop":
                ideal_metrics["bAcc"]
                - noisy_metrics["bAcc"],

                "F1_drop":
                ideal_metrics["F1"]
                - noisy_metrics["F1"],

                "logical_depth":
                float(
                    np.mean(
                        pooled[method]["logical_depth"]
                    )
                ),

                "logical_n2":
                float(
                    np.mean(
                        pooled[method]["logical_n2"]
                    )
                ),

                "vqc_transpiled_depth":
                float(
                    np.mean(
                        pooled[method][
                            "vqc_transpiled_depth"
                        ]
                    )
                ),

                "vqc_transpiled_n1":
                float(
                    np.mean(
                        pooled[method][
                            "vqc_transpiled_n1"
                        ]
                    )
                ),

                "vqc_transpiled_n2":
                float(
                    np.mean(
                        pooled[method][
                            "vqc_transpiled_n2"
                        ]
                    )
                ),

                "total_transpiled_depth":
                float(
                    np.mean(
                        pooled[method][
                            "total_transpiled_depth"
                        ]
                    )
                ),

                "total_transpiled_n1":
                float(
                    np.mean(
                        pooled[method][
                            "total_transpiled_n1"
                        ]
                    )
                ),

                "total_transpiled_n2":
                float(
                    np.mean(
                        pooled[method][
                            "total_transpiled_n2"
                        ]
                    )
                ),
            }
        )

    return seed_rows, task_rows


# =====================================================================
# 31. SUMMARY TABLES
# =====================================================================

def build_summary(seed_df):
    columns = [
        "ideal_acc",
        "ideal_bAcc",
        "ideal_F1",

        "noisy_acc",
        "noisy_bAcc",
        "noisy_F1",

        "acc_drop",
        "bAcc_drop",
        "F1_drop",

        "logical_depth",
        "logical_n2",

        "vqc_transpiled_depth",
        "vqc_transpiled_n1",
        "vqc_transpiled_n2",

        "total_transpiled_depth",
        "total_transpiled_n1",
        "total_transpiled_n2",
    ]

    return (
        seed_df
        .groupby("method")[columns]
        .agg(["mean", "std"])
    )


def build_compact_table(seed_df):
    rows = []

    for method in (
        "Naive-VQC",
        "QAS-No-CL",
        "CL-QAS",
    ):
        df = seed_df[
            seed_df["method"] == method
        ]

        rows.append(
            {
                "method": method,

                "Accuracy_mean":
                df["noisy_acc"].mean(),

                "Accuracy_std":
                df["noisy_acc"].std(ddof=1),

                "BAcc_mean":
                df["noisy_bAcc"].mean(),

                "BAcc_std":
                df["noisy_bAcc"].std(ddof=1),

                "F1_mean":
                df["noisy_F1"].mean(),

                "F1_std":
                df["noisy_F1"].std(ddof=1),

                "F1_drop_mean":
                df["F1_drop"].mean(),

                "F1_drop_std":
                df["F1_drop"].std(ddof=1),

                "logical_2Q_mean":
                df["logical_n2"].mean(),

                "logical_2Q_std":
                df["logical_n2"].std(ddof=1),

                "VQC_transpiled_2Q_mean":
                df["vqc_transpiled_n2"].mean(),

                "VQC_transpiled_2Q_std":
                df["vqc_transpiled_n2"].std(ddof=1),

                "VQC_transpiled_depth_mean":
                df["vqc_transpiled_depth"].mean(),

                "VQC_transpiled_depth_std":
                df["vqc_transpiled_depth"].std(ddof=1),

                "total_transpiled_2Q_mean":
                df["total_transpiled_n2"].mean(),

                "total_transpiled_2Q_std":
                df["total_transpiled_n2"].std(ddof=1),
            }
        )

    return pd.DataFrame(rows)


def build_backend_metadata():
    p1_values = [v["p1"] for v in LOCAL_NOISE_METADATA["one_qubit"].values()]
    p2_values = [v["p2"] for v in LOCAL_NOISE_METADATA["two_qubit"].values()]
    readout_values = []
    for v in LOCAL_NOISE_METADATA["readout"].values():
        readout_values.extend([v["p_meas1_prep0"], v["p_meas0_prep1"]])

    return pd.DataFrame([{
        "backend": backend_name(),
        "backend_num_qubits": backend_num_qubits(),
        "physical_layout": str(COMMON_PHYSICAL_LAYOUT),
        "simulated_qubits": NUM_QUBITS,
        "shots": QISKIT_SHOTS,
        "transpile_optimization_level": TRANSPILE_OPT_LEVEL,
        "aer_method": "statevector",
        "state_preparation": "ideal/common Aer initialize",
        "noise_scope": "VQC gates + readout",
                "region_selection": "fixed calibration-aware connected 8-qubit region",
                "invalid_2Q_threshold": INVALID_2Q_ERROR_THRESHOLD,
                "max_region_2Q_error": MAX_REGION_2Q_ERROR,
        "mean_local_1Q_error": float(np.mean(p1_values)) if p1_values else np.nan,
        "mean_local_2Q_error": float(np.mean(p2_values)) if p2_values else np.nan,
        "mean_local_readout_error": (
            float(np.mean(readout_values)) if readout_values else np.nan
        ),
        "debug_backend": DEBUG_BACKEND,
        "debug_test_samples_per_task": (
            DEBUG_NUM_TEST_SAMPLES if DEBUG_BACKEND else "ALL"
        ),
        "TT_rank": TT_RANK,
        "seeds": str(SEEDS),
    }])


# =====================================================================
# 32. MAIN
# =====================================================================

def main():
    global COMMON_PHYSICAL_LAYOUT

    print("=" * 100)
    print(
        "CL-QAS BACKEND-INFORMED "
        "REALISTIC-NOISE EVALUATION"
    )
    print("=" * 100)

    print(f"Device: {DEVICE}")
    print(f"Noise source: {backend_name()}")
    print(f"Physical backend qubits: {backend_num_qubits()}")
    print(f"Simulated qubits: {NUM_QUBITS}")
    print(f"Shots: {QISKIT_SHOTS}")
    print(
        f"Transpiler optimization: "
        f"{TRANSPILE_OPT_LEVEL}"
    )
    print(f"TT rank: {TT_RANK}")
    print(f"Seeds: {SEEDS}")
    print(f"DEBUG_BACKEND: {DEBUG_BACKEND}")

    if DEBUG_BACKEND:
        print(
            f"DEBUG_NUM_TEST_SAMPLES: "
            f"{DEBUG_NUM_TEST_SAMPLES}"
        )

    print("\nInitializing local 8-qubit backend-informed noise model...")
    initialize_backend_informed_simulator()
    print(build_backend_metadata().to_string(index=False))

    tasks = load_tasks()

    all_seed_rows = []
    all_task_rows = []

    for seed in SEEDS:
        seed_rows, task_rows = run_seed(
            tasks,
            seed,
        )

        all_seed_rows.extend(seed_rows)
        all_task_rows.extend(task_rows)

    seed_df = pd.DataFrame(
        all_seed_rows
    )

    task_df = pd.DataFrame(
        all_task_rows
    )

    summary = build_summary(
        seed_df
    )

    compact = build_compact_table(
        seed_df
    )

    metadata = build_backend_metadata()

    print("\n" + "=" * 160)
    print(
        "BACKEND-INFORMED "
        "REALISTIC-NOISE SUMMARY"
    )
    print("=" * 160)

    print(summary.to_string())

    print("\n" + "=" * 160)
    print("COMPACT MANUSCRIPT TABLE")
    print("=" * 160)

    print(
        compact.to_string(index=False)
    )

    if DEBUG_BACKEND:
        print(
            "\nWARNING: DEBUG_BACKEND=True. "
            "These noisy results use a small test subset and "
            "must not be reported as final manuscript results."
        )

    seed_df.to_csv(
        "backend_noise_seed_results.csv",
        index=False,
    )

    task_df.to_csv(
        "backend_noise_task_results.csv",
        index=False,
    )

    summary.to_csv(
        "backend_noise_summary.csv"
    )

    compact.to_csv(
        "backend_noise_compact_table.csv",
        index=False,
    )

    metadata.to_csv(
        "backend_noise_metadata.csv",
        index=False,
    )

    print("\nSaved:")
    print("  backend_noise_seed_results.csv")
    print("  backend_noise_task_results.csv")
    print("  backend_noise_summary.csv")
    print("  backend_noise_compact_table.csv")
    print("  backend_noise_metadata.csv")

    print("\nFor final manuscript results:")
    print("  DEBUG_BACKEND = False")
    print(
        "After confirming stability, you may also use "
        "QISKIT_SHOTS = 2048."
    )


if __name__ == "__main__":
    main()

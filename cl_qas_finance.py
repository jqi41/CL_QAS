#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finance_cl_qas_tt256_noise.py

Financial time-series CL-QAS with 256-D inputs + TT-encoding (4,16,4)->(3,2,2):
- tt_in_modes=(4,16,4), tt_out_modes=(3,2,2), tt_ranks=(1,2,3,1)  => 256 -> 12 angles
- QASPolicy is feature-adaptive: takes 256-D, internally produces per-qubit tokens (Q=12)
- Noise model: 1q depolarizing, 2q Pauli after CZ, symmetric readout error
- Methods: naive-vqc (fixed arch), qas-no-cl, cl-qas (with EWC + replay)
- Reporting: pretty console tables, CSV per-task + summary (means/stds), rewards included

Dependencies:
  torch, numpy, pandas, scikit-learn, torchquantum
"""

import os, math, random, argparse
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import torchquantum as tq
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd

# ========================= Repro =========================
def set_seed(seed=1234):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ========================= Finance data & 256-D features =========================
def _ema(x, span):
    alpha = 2.0/(span+1.0)
    y = np.zeros_like(x, dtype=np.float64)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha*x[t] + (1-alpha)*y[t-1]
    return y

def _rsi(close, period=14):
    r = np.diff(close, prepend=close[0])
    up = np.maximum(r, 0.0)
    dn = np.maximum(-r, 0.0)
    ema_up = _ema(up, period)
    ema_dn = _ema(dn, period)
    rs = np.divide(ema_up, np.maximum(1e-12, ema_dn))
    return 100.0 - (100.0/(1.0+rs))

def _macd(close, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def _rolling_mean(x, w):
    c = np.cumsum(np.insert(x,0,0.0))
    res = (c[w:] - c[:-w]) / float(w)
    res = np.concatenate([np.full(w-1, res[0]), res])
    return res

def _rolling_std(x, w):
    m = _rolling_mean(x, w)
    s2 = _rolling_mean((x-m)**2, w)
    return np.sqrt(np.maximum(0.0, s2))

def _base8_features(close, lookback=32):
    """8 indicators per timestep (N x 8)."""
    close = close.astype(np.float64)
    ret = np.diff(close, prepend=close[0]) / np.maximum(1e-8, close)
    r_mean = _rolling_mean(ret, lookback)
    r_std  = _rolling_std(ret, lookback)
    rsi = _rsi(close, period=14)/100.0
    macd, sig, hist = _macd(close)
    mom = close / np.maximum(1e-8, _ema(close, lookback)) - 1.0
    bb_z = (close - _rolling_mean(close, lookback)) / np.maximum(1e-8, _rolling_std(close, lookback))
    vol = _rolling_std(ret, 5)
    feats = np.stack([ret, r_mean, r_std, rsi, hist, mom, bb_z, vol], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32)  # (N, 8)

def _labels_from_returns(close, horizon=1):
    r = np.diff(close, n=horizon, prepend=[close[0]]*horizon) / np.maximum(1e-8, close)
    y = (r > 0.0).astype(np.int64)
    return y

def _stack_last_L(feats, L=32):
    """Create 256-D by stacking last L=32 timesteps of 8-D -> 256-D."""
    N, D = feats.shape
    assert D == 8
    if N < L:
        raise ValueError("Not enough timesteps to build 256-D windows.")
    X = []
    for t in range(L-1, N):
        window = feats[t-L+1:t+1].reshape(-1)  # (L*8) == 256
        X.append(window)
    return np.stack(X, axis=0)  # (N-L+1, 256)

def load_close_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    for c in ['Close','close','Adj Close','adj_close']:
        if c in df.columns:
            arr = df[c].values.astype(np.float64)
            if np.all(np.isfinite(arr)) and arr.size > 300:
                return arr
    raise RuntimeError(f"Could not find a close-like column in {csv_path}")

def make_synthetic_close(n=6000, regimes=6, seed=777):
    rs = np.random.RandomState(seed)
    x = np.zeros(n, dtype=np.float64)
    price = 100.0; seg = n // regimes; idx = 0
    for k in range(regimes):
        mu = rs.uniform(-0.0002, 0.0005)
        sigma = rs.uniform(0.004, 0.02)
        phi = rs.uniform(0.1, 0.8)
        T = seg if k < regimes-1 else n-idx
        eps = rs.normal(0, sigma, size=T)
        r = np.zeros(T)
        for t in range(T):
            r[t] = mu + phi*(r[t-1] if t>0 else 0.0) + eps[t]
        for t in range(T):
            price *= (1.0 + r[t])
            x[idx] = price
            idx += 1
    return x

def build_finance_tasks(close, n_tasks=8, lookback=32, horizon=1, min_per_task=500):
    """
    Build tasks with 256-D inputs:
      - Build base 8-D features per timestep
      - Stack last L=32 steps -> 256-D vectors
      - Align labels to the 256-D samples (drop first L-1)
    """
    feats8 = _base8_features(close, lookback=lookback)     # (N, 8)
    X256 = _stack_last_L(feats8, L=lookback)               # (N-L+1, 256)
    y_all = _labels_from_returns(close, horizon=horizon)   # (N,)
    y_all = np.roll(y_all, -1); y_all[-1] = y_all[-2]
    y = y_all[lookback-1:]                                 # align to X256 length

    # Robust normalization (per task later as well)
    Ltot = len(X256)
    cuts = np.linspace(0, Ltot, n_tasks+1, dtype=int)
    tasks = []
    for i in range(n_tasks):
        s, e = cuts[i], cuts[i+1]
        Xi, yi = X256[s:e], y[s:e]
        if len(Xi) < min_per_task:
            continue
        med = np.median(Xi, axis=0, keepdims=True)
        mad = np.median(np.abs(Xi - med), axis=0, keepdims=True) + 1e-6
        Zi = np.tanh((Xi - med) / (1.4826 * mad))
        tasks.append((torch.tensor(Zi, dtype=torch.float32),
                      torch.tensor(yi, dtype=torch.long)))
    return tasks

# ========================= Replay Buffer (CL) =========================
class ReplayBuffer:
    def __init__(self, per_class_cap=400):
        self.data = {0: [], 1: []}
        self.cap = per_class_cap
    def add(self, X, y, take_per_class=120):
        for c in [0, 1]:
            idx = (y == c).nonzero(as_tuple=True)[0]
            if len(idx) == 0: continue
            sel = idx[torch.randperm(len(idx))[:min(len(idx), take_per_class)]]
            feats = X[sel].cpu(); labs = y[sel].cpu()
            self.data[c].extend(list(zip(feats, labs)))
            if len(self.data[c]) > self.cap:
                self.data[c] = self.data[c][-self.cap:]
    def sample_missing_class(self, missing_class, n=120):
        pool = self.data.get(missing_class, [])
        if not pool: return None, None
        take = min(n, len(pool))
        sel = random.sample(pool, take)
        feats = torch.stack([t[0] for t in sel])
        labs = torch.stack([t[1] for t in sel])
        return feats, labs

# ========================= Policy (feature-adaptive, 256->Q tokens) =========================
class QASPolicy(nn.Module):
    """
    Feature-adaptive: consumes 256-D inputs and emits per-qubit action logits (B, Q, 4),
    where Q is the number of qubits. A linear 'adapt' maps 256 -> Q tokens.
    """
    def __init__(self, num_qubits=12, fea_dim=256, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_qubits = num_qubits
        self.fea_dim = fea_dim
        self.adapt = nn.Linear(fea_dim, num_qubits)
        self.token_embed = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout, dim_feedforward=4*d_model
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.randn(1, num_qubits, d_model) * 0.02)
        self.head = nn.Linear(d_model, 4)  # RX, RY, RZ, CZ
    def forward(self, x):
        tok = self.adapt(x).unsqueeze(-1)     # (B, Q, 1)
        h = self.token_embed(tok)             # (B, Q, d_model)
        h = self.encoder(h + self.pos)
        return self.head(h)                   # (B, Q, 4)

# ========================= TT-encoding (256 -> 12) =========================
class TTMatrix(nn.Module):
    """
    TT-matrix mapping R^{prod(in_modes)} -> R^{prod(out_modes)} with cores G_k in R^{r_{k-1} x n_k x m_k x r_k}
    """
    def __init__(self, in_modes, out_modes, tt_ranks):
        super().__init__()
        assert len(in_modes) == len(out_modes)
        self.in_modes = tuple(in_modes)
        self.out_modes = tuple(out_modes)
        self.d = len(in_modes)
        self.tt_ranks = tuple(tt_ranks)
        assert self.tt_ranks[0] == 1 and self.tt_ranks[-1] == 1 and len(self.tt_ranks) == self.d + 1
        cores = []
        for k in range(self.d):
            r0, r1 = self.tt_ranks[k], self.tt_ranks[k+1]
            nk, mk = self.in_modes[k], self.out_modes[k]
            G = nn.Parameter(0.02 * torch.randn(r0, nk, mk, r1))
            cores.append(G)
        self.cores = nn.ParameterList(cores)
    @property
    def in_features(self):  return int(np.prod(self.in_modes))
    @property
    def out_features(self): return int(np.prod(self.out_modes))
    def forward(self, x):
        B = x.shape[0]
        y = x.view(B, *self.in_modes)  # (B, n1,...,nd)
        left = y.unsqueeze(1)          # (B, 1, n1,...,nd)
        r_prev = 1
        for k in range(self.d):
            G = self.cores[k]  # (r_{k-1}, n_k, m_k, r_k)
            nk, mk = self.in_modes[k], self.out_modes[k]
            left = left.contiguous().view(B, r_prev, nk, -1)  # (B, r_prev, nk, R)
            R = left.shape[-1]
            left_flat = left.view(B, r_prev*nk, R)                    # (B, r_prev*nk, R)
            core_flat = G.view(r_prev*nk, mk*self.tt_ranks[k+1])      # (r_prev*nk, mk*r_k)
            out = torch.matmul(left_flat.transpose(1,2), core_flat)   # (B, R, mk*r_k)
            out = out.view(B, R, mk, self.tt_ranks[k+1]).transpose(1,3)  # (B, r_k, mk, R)
            left = out
            r_prev = self.tt_ranks[k+1]
        left = left.squeeze(1)          # (B, mk, R_final)
        left = left.contiguous().view(B, -1)
        return left.view(B, self.out_features)

class TTEncoder(nn.Module):
    """
    Wraps TTMatrix to produce per-qubit angles:
      256 -> 12 via TT, then LayerNorm + tanh scaling -> [-pi, pi]
    """
    def __init__(self,
                 in_modes=(4,16,4),
                 out_modes=(3,2,2),
                 tt_ranks=(1,2,3,1),
                 angle_scale=np.pi,
                 post_norm=True):
        super().__init__()
        self.tt = TTMatrix(in_modes, out_modes, tt_ranks)
        assert self.tt.in_features == 256, "tt_in_modes must multiply to 256"
        assert self.tt.out_features == 12, "tt_out_modes must multiply to 12"
        self.post = nn.LayerNorm(self.tt.out_features) if post_norm else nn.Identity()
        self.angle_scale = float(angle_scale)
    def forward(self, x):
        y = self.tt(x)               # (B, 12)
        y = self.post(y)
        return torch.tanh(y) * self.angle_scale

# ========================= Noise helpers (torchquantum compatible) =========================
class NoiseConfig:
    def __init__(self, p_depol_1q=0.0, p_error_2q=0.0, p_readout=0.0):
        self.p1 = float(p_depol_1q)   # single-qubit depolarizing
        self.p2 = float(p_error_2q)   # two-qubit Pauli error after CZ
        self.pro = float(p_readout)   # readout bit-flip prob

def _apply_random_pauli_1q(qdev, wire: int, which: int):
    device = getattr(qdev, "device", None)
    if device is None:
        try:
            device = qdev.states.device
        except Exception:
            device = "cpu"
    angle = torch.full((qdev.bsz,), math.pi, device=device)
    if which == 0:   tq.RX(has_params=False)(qdev, wires=wire, params=angle)
    elif which == 1: tq.RY(has_params=False)(qdev, wires=wire, params=angle)
    else:            tq.RZ(has_params=False)(qdev, wires=wire, params=angle)

def _maybe_depolarize_layer(qdev, num_qubits: int, p1: float):
    if p1 <= 0.0: return
    import random as pyr
    for i in range(num_qubits):
        if pyr.random() < p1:
            _apply_random_pauli_1q(qdev, i, pyr.randint(0, 2))

def _maybe_2q_error(qdev, i: int, j: int, p2: float):
    if p2 <= 0.0: return
    import random as pyr
    if pyr.random() < p2:
        a = pyr.randint(0, 3)  # 0=I,1=X,2=Y,3=Z
        b = pyr.randint(0, 3)
        if a != 0: _apply_random_pauli_1q(qdev, i, a-1)
        if b != 0: _apply_random_pauli_1q(qdev, j, b-1)

# ========================= Hybrid adapter + QNN with noise =========================
class FeatureAdapter(nn.Module):
    """
    256 -> (TT) -> 12 angles; optional small MLP to refine angles (here: identity after TT+LN+tanh*pi).
    """
    def __init__(self, tt_in_modes=(4,16,4), tt_out_modes=(3,2,2), tt_ranks=(1,2,3,1), angle_scale=np.pi):
        super().__init__()
        self.tt_enc = TTEncoder(in_modes=tt_in_modes, out_modes=tt_out_modes, tt_ranks=tt_ranks,
                                angle_scale=angle_scale, post_norm=True)
    def forward(self, x):
        return self.tt_enc(x)  # (B, 12) in [-pi, pi]

def build_qnn_from_actions(actions, num_qubits=12, depth=4, reupload=True, extra_cz=True,
                           noise: Optional[NoiseConfig] = None):
    if noise is None: noise = NoiseConfig()
    class GeneratedQNN(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.num_qubits = num_qubits
            self.depth = depth
            self.actions = actions.detach().cpu().tolist()
            self.noise = noise
        def _encode(self, qdev, angles):
            # angles already in radians [-pi, pi]
            for i in range(self.num_qubits):
                tq.RY(has_params=False)(qdev, wires=i, params=angles[:, i])
        def _single_qubit_layer(self, qdev):
            for i, a in enumerate(self.actions):
                if a == 0:   tq.RX(has_params=True, trainable=True, init_params=0.08)(qdev, wires=i)
                elif a == 1: tq.RY(has_params=True, trainable=True, init_params=0.08)(qdev, wires=i)
                elif a == 2: tq.RZ(has_params=True, trainable=True, init_params=0.08)(qdev, wires=i)
                else:        tq.RY(has_params=True, trainable=True, init_params=0.03)(qdev, wires=i)
            _maybe_depolarize_layer(qdev, self.num_qubits, self.noise.p1)
        def _entangle_ring(self, qdev):
            for i in range(self.num_qubits):
                j = (i + 1) % self.num_qubits
                tq.CZ()(qdev, wires=[i, j])
                _maybe_2q_error(qdev, i, j, self.noise.p2)
            if extra_cz:
                for i, a in enumerate(self.actions):
                    if a == 3:
                        j = (i + 2) % self.num_qubits
                        tq.CZ()(qdev, wires=[i, j])
                        _maybe_2q_error(qdev, i, j, self.noise.p2)
        def forward(self, qdev, angles):
            if reupload:
                for _ in range(self.depth):
                    self._encode(qdev, angles)
                    self._single_qubit_layer(qdev)
                    self._entangle_ring(qdev)
            else:
                self._encode(qdev, angles)
                for _ in range(self.depth):
                    self._single_qubit_layer(qdev)
                    self._entangle_ring(qdev)
    return GeneratedQNN()

class QNNClassifier(nn.Module):
    def __init__(self, qlayer: tq.QuantumModule, num_qubits=12, num_classes=2,
                 readout_prob: float = 0.01,
                 tt_in_modes=(4,16,4), tt_out_modes=(3,2,2), tt_ranks=(1,2,3,1)):
        super().__init__()
        self.adapter = FeatureAdapter(tt_in_modes, tt_out_modes, tt_ranks, angle_scale=np.pi)
        self.q_layer = qlayer
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.fc = nn.Linear(num_qubits, num_classes)
        self.readout_prob = float(readout_prob)
    def forward(self, x256):
        angles = self.adapter(x256)  # (B, 12) in radians
        bsz = angles.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.fc.in_features, bsz=bsz, device=angles.device)
        self.q_layer(qdev, angles)
        z = self.measure(qdev)  # (B, 12)
        if self.readout_prob > 0.0:
            z = (1.0 - 2.0 * self.readout_prob) * z
        return self.fc(z)

# ========================= EWC for policy =========================
class EWCLoss:
    def __init__(self, model: nn.Module, dataloader, device='cpu', lambda_ewc=50.0):
        self.model = model; self.device = device; self.lambda_ewc = lambda_ewc
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = self._compute_fisher(dataloader)
    def _compute_fisher(self, dataloader):
        fim = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters()}
        self.model.eval()
        for xb, _ in dataloader:
            xb = xb.to(self.device)
            self.model.zero_grad()
            logits = self.model(xb)  # (B,Q,4)
            probs = torch.softmax(logits, dim=-1)
            pm = probs.mean(dim=0)
            dists = [torch.distributions.Categorical(pm[q]) for q in range(pm.size(0))]
            actions = torch.stack([d.sample() for d in dists])
            logp = sum(d.log_prob(a) for d, a in zip(dists, actions))
            (-logp).backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fim[n] += (p.grad.detach() ** 2) / len(dataloader)
        return fim
    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            loss = loss + (self.fisher[n] * (p - self.params[n])**2).sum()
        return self.lambda_ewc * loss

# ========================= Metrics, loaders, training =========================
def make_class_weighted_ce(y, device, n_classes=2, label_smoothing=0.05):
    counts = torch.bincount(y, minlength=n_classes).float()
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=label_smoothing), weights

def maybe_make_sampler(y, n_classes=2):
    counts = torch.bincount(y, minlength=n_classes).float()
    if (counts > 0).sum() < 2: return None
    class_weights = 1.0 / (counts + 1e-6)
    sample_weights = class_weights[y]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(y), replacement=True)

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    total = cm.sum()
    acc = (np.trace(cm) / total) if total > 0 else 0.0
    tpr0 = cm[0,0] / max(1, cm[0].sum()); tpr1 = cm[1,1] / max(1, cm[1].sum())
    bacc = 0.5*(tpr0 + tpr1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    return cm, acc, bacc, f1

def time_split(X, y, train_frac=0.8, val_frac=0.1):
    n = len(X); n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_train = max(32, n_train); n_val = max(16, n_val)
    n_train = min(n-32, n_train); n_val = min(n - n_train - 16, n_val)
    i1 = n_train; i2 = n_train + n_val
    Xtr, ytr = X[:i1], y[:i1]
    Xva, yva = X[i1:i2], y[i1:i2]
    Xte, yte = X[i2:], y[i2:]
    return Xtr, ytr, Xva, yva, Xte, yte

def train_qnn_fixed_arch(qnn, train_loader, val_loader, ce_loss, device='cpu',
                         epochs=40, lr=2e-3, wd=1e-4, patience=6, clip=1.0):
    qnn.train()
    opt = optim.Adam(qnn.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.1)
    best_val = float('inf'); best_state = None; bad = 0
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = qnn(xb)
            loss = ce_loss(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qnn.parameters(), clip)
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
        ce = loss_sum / max(1,total)
        acc = correct / max(1,total)
        qnn.eval()
        with torch.no_grad():
            vloss, vcount = 0.0, 0
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                vloss += ce_loss(qnn(xb), yb).item() * xb.size(0)
                vcount += xb.size(0)
            vce = vloss / max(1, vcount)
        qnn.train(); sched.step()
        print(f"    [QNN] Epoch {ep+1:02d} | tr_loss={ce:.4f} | tr_acc={acc:.4f} | val_loss={vce:.4f}")
        if vce < best_val - 1e-4:
            best_val = vce; best_state = {k: v.detach().cpu().clone() for k, v in qnn.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"    Early stop at epoch {ep+1} (best val_loss={best_val:.4f})")
                break
    if best_state is not None:
        qnn.load_state_dict(best_state)

# ========================= Policy helpers & reward =========================
def apply_cz_prior(pm, max_expected_cz=0.30, scale=0.5, eps=1e-8):
    expected_cz = pm[:, 3].mean()
    if expected_cz.detach().item() > max_expected_cz:
        pm = pm.clone()
        pm[:, 3] = pm[:, 3] * scale
        pm = pm / (pm.sum(dim=-1, keepdim=True) + eps)
    return pm

def greedy_actions_from_policy(policy, x_batch):
    with torch.no_grad():
        logits = policy(x_batch)                 # (B,Q,4)
        pm = torch.softmax(logits, dim=-1).mean(dim=0)
        pm = apply_cz_prior(pm)
        actions = torch.argmax(pm, dim=-1)
    return actions

def reward_from_metrics(bacc, f1, actions, num_qubits):
    cz_frac = float((actions == 3).sum().item()) / num_qubits
    # lightweight structural penalty
    reward = 0.6 * bacc + 0.4 * f1 - 0.05 * cz_frac
    return reward, cz_frac

class RewardBaseline:
    def __init__(self, beta=0.9): self.beta = beta; self.val = None
    def update(self, r):
        self.val = r if self.val is None else (self.beta*self.val + (1-self.beta)*r)
        return self.val

def reinforce_update(policy, x_batch, reward_scalar, baseline_val=None,
                     ewc_prev=None, device='cpu', lr=2e-3, beta_kl=0.01, clip=1.0):
    policy.train()
    opt = optim.Adam(policy.parameters(), lr=lr)
    logits = policy(x_batch)                         # (B,Q,4)
    pm = torch.softmax(logits, dim=-1).mean(dim=0)  # (Q,4)
    pm = apply_cz_prior(pm)
    dists = [torch.distributions.Categorical(pm[q]) for q in range(pm.size(0))]
    actions = torch.stack([d.sample() for d in dists])  # (Q,)
    logp = sum(d.log_prob(a) for d, a in zip(dists, actions))
    uni = torch.full_like(pm, 1.0/pm.size(-1))
    kl = torch.sum(pm * (pm.clamp_min(1e-8).log() - uni.log()))
    ewc_pen = ewc_prev.penalty(policy) if ewc_prev is not None else 0.0
    adv = torch.tensor(float(reward_scalar if baseline_val is None else reward_scalar - baseline_val), device=device)
    L = -(adv * logp) + beta_kl * kl + ewc_pen
    opt.zero_grad(); L.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
    opt.step()
    return actions.detach()

# ========================= Data loaders (time-aware) =========================
def _make_loaders_time_aware(X, y, device, batch_size=128):
    Xtr, ytr, Xva, yva, Xte, yte = time_split(X, y, train_frac=0.8, val_frac=0.1)
    sampler = maybe_make_sampler(ytr)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                              sampler=sampler if sampler is not None else None,
                              shuffle=(sampler is None))
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)
    return (Xtr, ytr, Xva, yva, Xte, yte), train_loader, val_loader, test_loader

# ========================= Method runners =========================
def run_naive_vqc_on_task(X, y, device, epochs=40, lr=2e-3, batch_size=128,
                          noise: Optional[NoiseConfig] = None, num_qubits=12):
    (Xtr,ytr,Xva,yva,Xte,yte), tr_loader, va_loader, te_loader = _make_loaders_time_aware(X, y, device, batch_size)
    ce_loss, _ = make_class_weighted_ce(ytr, device=device, label_smoothing=0.05)
    # Hand-crafted actions: all RY + ring CZ at the end qubit
    actions = torch.tensor([1]*(num_qubits-1) + [3], dtype=torch.long, device=device)
    adapter = None  # Naive-VQC uses TT inside classifier; adapter is in QNNClassifier already
    qlayer = build_qnn_from_actions(actions, num_qubits=num_qubits, depth=4,
                                    reupload=True, extra_cz=False, noise=noise).to(device)
    qnn = QNNClassifier(qlayer, num_qubits=num_qubits, num_classes=2,
                        readout_prob=(0.0 if noise is None else noise.pro),
                        tt_in_modes=(4,16,4), tt_out_modes=(3,2,2), tt_ranks=(1,2,3,1)).to(device)
    print("  Training QNN (naive-vqc, TT 256->12)...")
    train_qnn_fixed_arch(qnn, tr_loader, va_loader, ce_loss, device=device, epochs=epochs, lr=lr)
    # Test
    qnn.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(device)
            logits = qnn(xb)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(yb.cpu().tolist())
    cm, acc, bacc, f1 = compute_metrics(y_true, y_pred)
    print(f"  Test | acc={acc:.4f} | bAcc={bacc:.4f} | F1={f1:.4f}\n  Confusion:\n{cm}")
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    return {"acc":acc, "bacc":bacc, "f1":f1, "cm":cm, "cz_frac":0.0, "reward":np.nan,
            "tn":tn, "fp":fp, "fn":fn, "tp":tp}

def run_qas_on_task(policy, X, y, device, prev_ewc=None, use_cl=False,
                    replay=None, reward_baseline=None, epochs=40, lr=2e-3, batch_size=128,
                    noise: Optional[NoiseConfig] = None, num_qubits=12):
    (Xtr,ytr,Xva,yva,Xte,yte), tr_loader, va_loader, te_loader = _make_loaders_time_aware(X, y, device, batch_size)
    if use_cl:
        counts_tr = torch.bincount(ytr, minlength=2)
        missing = (counts_tr == 0).nonzero(as_tuple=True)[0].tolist()
        for m in missing:
            Xbuf, ybuf = (None, None) if replay is None else replay.sample_missing_class(m, n=120)
            if Xbuf is not None:
                Xtr = torch.cat([Xtr, Xbuf], dim=0)
                ytr = torch.cat([ytr, ybuf], dim=0)
        tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    ce_loss, _ = make_class_weighted_ce(ytr, device=device, label_smoothing=0.05)

    with torch.no_grad():
        xb_small = Xtr[:min(512, len(Xtr))].to(device)
        actions = greedy_actions_from_policy(policy, xb_small)  # length Q=num_qubits

    qlayer = build_qnn_from_actions(actions, num_qubits=num_qubits, depth=4,
                                    reupload=True, extra_cz=True, noise=noise).to(device)
    qnn = QNNClassifier(qlayer, num_qubits=num_qubits, num_classes=2,
                        readout_prob=(0.0 if noise is None else noise.pro),
                        tt_in_modes=(4,16,4), tt_out_modes=(3,2,2), tt_ranks=(1,2,3,1)).to(device)

    print("  Training QNN (policy-selected arch, TT 256->12)...")
    train_qnn_fixed_arch(qnn, tr_loader, va_loader, ce_loss, device=device, epochs=epochs, lr=lr)

    # Test metrics
    qnn.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(device)
            logits = qnn(xb)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(yb.cpu().tolist())
    cm, acc, bacc, f1 = compute_metrics(y_true, y_pred)

    # Validation reward for policy update
    y_true_v, y_pred_v = [], []
    qnn.eval()
    with torch.no_grad():
        for xb, yb in va_loader:
            xb = xb.to(device)
            logits = qnn(xb)
            y_pred_v.extend(logits.argmax(dim=1).cpu().tolist())
            y_true_v.extend(yb.cpu().tolist())
    cm_v, _, bacc_v, f1_v = compute_metrics(y_true_v, y_pred_v)
    reward, cz_frac = reward_from_metrics(bacc_v, f1_v, actions, num_qubits)

    print(f"  Test | acc={acc:.4f} | bAcc={bacc:.4f} | F1={f1:.4f}\n  Confusion:\n{cm}")
    print(f"  Policy update... reward={reward:.4f} (cz_frac={cz_frac:.2f})")

    xb_pol = Xtr[:min(512, len(Xtr))].to(device)
    baseline_val = reward_baseline.update(reward) if reward_baseline is not None else None
    actions_after = reinforce_update(
        policy, xb_pol, reward, baseline_val=baseline_val,
        ewc_prev=prev_ewc if use_cl else None, device=device, lr=1.5e-3, beta_kl=0.01
    )

    if use_cl and replay is not None:
        replay.add(Xtr, ytr, take_per_class=140)

    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    return {"acc":acc, "bacc":bacc, "f1":f1, "cm":cm, "reward":float(reward),
            "actions":actions.tolist(), "actions_after":actions_after.tolist(),
            "cz_frac":float(cz_frac), "tn":tn, "fp":fp, "fn":fn, "tp":tp}

# ========================= Reporting =========================
def accumulate_results(rows: List[Dict], method: str, task_idx: int, res: Dict):
    row = {
        "method": method,
        "task": task_idx,
        "acc": float(res["acc"]),
        "bacc": float(res["bacc"]),
        "f1": float(res["f1"]),
        "reward": (np.nan if np.isnan(res.get("reward", np.nan)) else float(res.get("reward", np.nan))),
        "cz_frac": float(res.get("cz_frac", 0.0)),
        "tn": int(res.get("tn", 0)),
        "fp": int(res.get("fp", 0)),
        "fn": int(res.get("fn", 0)),
        "tp": int(res.get("tp", 0)),
    }
    rows.append(row)

def print_method_table(method: str, rows_df: pd.DataFrame):
    dfm = rows_df[rows_df["method"] == method].sort_values("task")
    if dfm.empty: return
    mean_std = dfm[["acc","bacc","f1","reward"]].agg(["mean","std"])
    print("\n" + "="*70)
    print(f"{method.upper():^70}")
    print("="*70)
    print(dfm[["task","acc","bacc","f1","reward","cz_frac","tn","fp","fn","tp"]].to_string(
        index=False,
        formatters={
            "acc":"{:.3f}".format,"bacc":"{:.3f}".format,"f1":"{:.3f}".format,
            "reward":(lambda x: "" if np.isnan(x) else f"{x:.4f}"),
            "cz_frac":"{:.2f}".format
        })
    )
    print("-"*70)
    rmean = mean_std.loc['mean','reward'] if 'reward' in mean_std.columns else np.nan
    rstd  = mean_std.loc['std','reward']  if 'reward' in mean_std.columns else np.nan
    if method == "naive-vqc":
        print(f" mean  | acc={mean_std.loc['mean','acc']:.3f}±{mean_std.loc['std','acc']:.3f} "
              f"| bAcc={mean_std.loc['mean','bacc']:.3f}±{mean_std.loc['std','bacc']:.3f} "
              f"| F1={mean_std.loc['mean','f1']:.3f}±{mean_std.loc['std','f1']:.3f}")
    else:
        print(f" mean  | acc={mean_std.loc['mean','acc']:.3f}±{mean_std.loc['std','acc']:.3f} "
              f"| bAcc={mean_std.loc['mean','bacc']:.3f}±{mean_std.loc['std','bacc']:.3f} "
              f"| F1={mean_std.loc['mean','f1']:.3f}±{mean_std.loc['std','f1']:.3f} "
              f"| Rwd={rmean:.4f}±{rstd:.4f}")

def export_csv(rows_df: pd.DataFrame, suffix: str = ""):
    rows_df.sort_values(["method","task"]).to_csv(f"finance_task_results{suffix}.csv", index=False)
    summary = rows_df.groupby("method")[["acc","bacc","f1","reward","cz_frac"]].agg(["mean","std"])
    summary.columns = [f"{a}_{b}" for a,b in summary.columns]
    summary.reset_index().to_csv(f"finance_task_summary{suffix}.csv", index=False)
    print(f"\nSaved CSVs: finance_task_results{suffix}.csv and finance_task_summary{suffix}.csv")

# ========================= Main =========================
def main():
    parser = argparse.ArgumentParser(description="CL-QAS Finance (256D -> TT(4,16,4)->(3,2,2)=12) with Noise")
    parser.add_argument("--csv", nargs="*", default=[], help="CSV files (Close/Adj Close); if none, uses synthetic")
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=32, help="window length for 256-D stack (must be 32)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    # Noise
    parser.add_argument("--p1", type=float, default=0.00, help="single-qubit depolarizing prob")
    parser.add_argument("--p2", type=float, default=0.00, help="two-qubit Pauli error prob")
    parser.add_argument("--pro", type=float, default=0.00, help="readout bit-flip prob")
    parser.add_argument("--suffix", type=str, default="", help="suffix for output CSV names")
    args = parser.parse_args()

    if args.lookback != 32:
        print("[WARN] For 256-D inputs this script expects lookback=32. Continuing anyway…")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)

    NOISE = NoiseConfig(p_depol_1q=args.p1, p_error_2q=args.p2, p_readout=args.pro)
    print(f"[Noise] 1q depol={NOISE.p1:.4%} | 2q err={NOISE.p2:.4%} | readout={NOISE.pro:.2%}")

    # Load prices
    closes = []
    for p in args.csv:
        if os.path.exists(p):
            try:
                closes.append(load_close_from_csv(p))
            except Exception as e:
                print(f"Warning: {p} skipped ({e})")
    if not closes:
        closes = [make_synthetic_close(n=6000, regimes=6, seed=777)]
    close = np.concatenate(closes)

    # Build tasks (256-D inputs)
    tasks = build_finance_tasks(close, n_tasks=args.tasks, lookback=32, horizon=1, min_per_task=500)
    print(f"Prepared {len(tasks)} tasks, input_dim={tasks[0][0].shape[1]} (expected 256)")

    NUM_QUBITS = 12  # TT (3*2*2)
    ALL_ROWS: List[Dict] = []

    # Baseline: Naive-VQC
    print("\n================= Baseline: Naive-VQC =================")
    res_vqc = []
    for t_idx, (X, y) in enumerate(tasks, 1):
        print(f"\n== Task {t_idx}")
        res = run_naive_vqc_on_task(X, y, device=device, epochs=args.epochs, lr=2e-3, batch_size=128,
                                    noise=NOISE, num_qubits=NUM_QUBITS)
        res_vqc.append(res); accumulate_results(ALL_ROWS, "naive-vqc", t_idx, res)

    # Baseline: QAS (no-CL)
    print("\n================= Baseline: QAS (no-CL) =================")
    policy_nc = QASPolicy(num_qubits=NUM_QUBITS, fea_dim=256, d_model=64, nhead=8).to(device)
    res_qas_nc = []; rb_nc = RewardBaseline(beta=0.9)
    for t_idx, (X, y) in enumerate(tasks, 1):
        print(f"\n== Task {t_idx} | method=qas-no-cl")
        res = run_qas_on_task(policy_nc, X, y, device=device, prev_ewc=None, use_cl=False,
                              replay=None, reward_baseline=rb_nc, epochs=args.epochs, lr=2e-3,
                              batch_size=128, noise=NOISE, num_qubits=NUM_QUBITS)
        res_qas_nc.append(res); accumulate_results(ALL_ROWS, "qas-no-cl", t_idx, res)

    # Proposed: CL-QAS
    print("\n================= Proposed: CL-QAS =================")
    policy_cl = QASPolicy(num_qubits=NUM_QUBITS, fea_dim=256, d_model=64, nhead=8).to(device)
    replay = ReplayBuffer(per_class_cap=500)
    res_cl = []; prev_ewc = None; rb_cl = RewardBaseline(beta=0.9)
    for t_idx, (X, y) in enumerate(tasks, 1):
        print(f"\n== Task {t_idx} | method=cl-qas")
        res = run_qas_on_task(policy_cl, X, y, device=device, prev_ewc=prev_ewc, use_cl=True,
                              replay=replay, reward_baseline=rb_cl, epochs=args.epochs, lr=2e-3,
                              batch_size=128, noise=NOISE, num_qubits=NUM_QUBITS)
        res_cl.append(res); accumulate_results(ALL_ROWS, "cl-qas", t_idx, res)
        # refresh EWC on train window only
        Xtr, ytr, _, _, _, _ = time_split(X, y, train_frac=0.8, val_frac=0.1)
        tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=False)
        prev_ewc = EWCLoss(policy_cl, tr_loader, device=device, lambda_ewc=50.0)

    # Pretty print per method
    df_all = pd.DataFrame(ALL_ROWS)
    for method in ["naive-vqc", "qas-no-cl", "cl-qas"]:
        print_method_table(method, df_all)

    # Summary across tasks
    def summarize(arr, key):
        vals = np.array([a[key] for a in arr], dtype=float)
        return vals.mean(), vals.std()

    print("\n================= Summary (Test Metrics across tasks) =================")
    for name, arr in [("naive-vqc", res_vqc), ("qas-no-cl", res_qas_nc), ("cl-qas", res_cl)]:
        acc_m, acc_s = summarize(arr, "acc")
        b_m, b_s = summarize(arr, "bacc")
        f1_m, f1_s = summarize(arr, "f1")
        if name == "naive-vqc":
            print(f"{name:10s}: acc={acc_m:.3f}±{acc_s:.3f} | bAcc={b_m:.3f}±{b_s:.3f} | F1={f1_m:.3f}±{f1_s:.3f}")
        else:
            rew = np.array([a['reward'] for a in arr], dtype=float)
            print(f"{name:10s}: acc={acc_m:.3f}±{acc_s:.3f} | bAcc={b_m:.3f}±{b_s:.3f} | F1={f1_m:.3f}±{f1_s:.3f} | Reward={rew.mean():.4f}±{rew.std():.4f}")

    # Export CSVs
    suffix = f"_{args.suffix}" if args.suffix else ""
    export_csv(df_all, suffix=suffix)

if __name__ == "__main__":
    main()

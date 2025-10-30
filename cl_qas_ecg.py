#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CL-QAS on MIT-BIH ECG (N vs V) with ablation:
  - naive-vqc (fixed hand-crafted circuit, TT-encoding)
  - qas-no-cl  (policy search without EWC / replay, TT-encoding)
  - cl-qas     (policy search with EWC + replay, TT-encoding)
  - cl-qas-no-tt (ablation: CL-QAS without TT-encoding)

High-dim input (256):
- num_qubits = 12
- Input features per beat: 256-sample vector
- TT-encoding maps 256-D -> 12 angles with in_modes=(4,16,4), out_modes=(3,2,2), ranks=(1,2,3,1)
- QASPolicy adapts 256-D features to per-qubit tokens; circuits use ring+extra CZ if policy picks CZ

NoiseModel supports simple scaling approximations for depolarizing and readout errors; set p=0 for noiseless.

Outputs:
- Per-task metrics for all methods
- Mean±Std summaries
- Runtime per task for QAS methods; CL-QAS vs CL-QAS-no-TT ablation includes runtime

Dependencies:
  pip install wfdb torch numpy scikit-learn torchquantum
"""

import time
import random
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import torchquantum as tq
from sklearn.metrics import confusion_matrix, f1_score

try:
    import wfdb
except Exception as e:
    raise ImportError("Please install wfdb: pip install wfdb") from e


# ----------------------------- Repro
def set_seed(seed=1234):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(1234)


# ----------------------------- MIT-BIH helpers
AAMI_N = set(['N', 'L', 'R', 'e', 'j'])
AAMI_V = set(['V', 'E'])

def map_symbol_to_binary(sym):
    if sym in AAMI_N: return 0
    if sym in AAMI_V: return 1
    return None

def pick_single_channel(sig_names):
    names_upper = [s.upper() for s in sig_names]
    return names_upper.index('MLII') if 'MLII' in names_upper else 0

def bandpass(signal, fs=360.0, low=0.5, high=40.0):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    fftx = np.fft.rfft(signal)
    mask = (freqs >= low) & (freqs <= high)
    fftx *= mask
    return np.fft.irfft(fftx, n=n)

# ---------- High-dimensional (256) beat vector ----------
def beat_vector_256(segment, target_len=256):
    """
    Convert a beat segment to a 256-dim vector:
      1) linear resample to target_len
      2) per-beat z-normalize
      3) clip to 5 stds
    """
    x = np.asarray(segment, dtype=np.float32)
    src_t = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
    xr = np.interp(dst_t, src_t, x).astype(np.float32)
    mu, sd = float(xr.mean()), float(xr.std() + 1e-6)
    xr = (xr - mu) / sd
    xr = np.clip(xr, -5.0, 5.0)
    return xr.astype(np.float32)

def extract_record_dataset(record, max_beats=800, window_sec=0.6, min_per_class=10):
    """
    For a record, extract beats around annotations (N/V only), producing:
      X: (n_beats, 256)  y: (n_beats,)
    """
    rec = f"{record}"
    sig, fields = wfdb.rdsamp(rec, pn_dir='mitdb', sampto=None)
    ann = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
    ch_idx = pick_single_channel(fields['sig_name'])
    x = sig[:, ch_idx]
    fs = fields['fs']
    x = bandpass(x, fs=fs, low=0.5, high=40.0)

    half = int(window_sec * fs)
    X, y = [], []
    for samp, sym in zip(ann.sample, ann.symbol):
        label = map_symbol_to_binary(sym)
        if label is None:
            continue
        start, end = samp - half, samp + half
        if start < 0 or end >= len(x):
            continue
        seg = x[start:end]
        feats = beat_vector_256(seg, target_len=256)
        X.append(feats); y.append(label)
        if len(X) >= max_beats:
            break

    if len(X) == 0:
        raise RuntimeError(f"No usable beats for record {record} (N/V only).")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    cls_counts = np.bincount(y, minlength=2)
    if (cls_counts[0] < min_per_class) or (cls_counts[1] < min_per_class):
        raise RuntimeError(f"Record {record}: insufficient class counts {cls_counts.tolist()}")

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mu) / sd

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ----------------------------- Stratified split utilities
def stratified_split_indices(y, train_ratio=0.8, seed=1234):
    rng = np.random.RandomState(seed)
    y_np = y.cpu().numpy()
    idx_pos = np.where(y_np == 1)[0].tolist()
    idx_neg = np.where(y_np == 0)[0].tolist()
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    ntr_pos = int(train_ratio * len(idx_pos))
    ntr_neg = int(train_ratio * len(idx_neg))
    tr = idx_pos[:ntr_pos] + idx_neg[:ntr_neg]
    te = idx_pos[ntr_pos:] + idx_neg[ntr_neg:]
    rng.shuffle(tr); rng.shuffle(te)
    return torch.tensor(tr, dtype=torch.long), torch.tensor(te, dtype=torch.long)

def stratified_subsplit_indices(y, base_indices, sub_train_ratio=0.85, seed=1234):
    sub_y = y[base_indices]
    pos = base_indices[(sub_y == 1).nonzero(as_tuple=True)[0]]
    neg = base_indices[(sub_y == 0).nonzero(as_tuple=True)[0]]
    if len(pos) > 0:
        pos = pos[torch.randperm(len(pos))]
    if len(neg) > 0:
        neg = neg[torch.randperm(len(neg))]
    ntr_pos = int(sub_train_ratio * len(pos))
    ntr_neg = int(sub_train_ratio * len(neg))
    tr = torch.cat([pos[:ntr_pos], neg[:ntr_neg]], dim=0)
    va = torch.cat([pos[ntr_pos:], neg[ntr_neg:]], dim=0)
    if len(tr) > 0: tr = tr[torch.randperm(len(tr))]
    if len(va) > 0: va = va[torch.randperm(len(va))]
    return tr, va


# ----------------------------- Replay buffer (CL only)
class ReplayBuffer:
    def __init__(self, per_class_cap=500):
        self.data = {0: [], 1: []}
        self.cap = per_class_cap

    def add(self, X, y, take_per_class=80):
        for c in [0, 1]:
            idx = (y == c).nonzero(as_tuple=True)[0]
            if len(idx) == 0: continue
            sel = idx[torch.randperm(len(idx))[:min(len(idx), take_per_class)]]
            feats = X[sel].cpu()
            labs = y[sel].cpu()
            self.data[c].extend(list(zip(feats, labs)))
            if len(self.data[c]) > self.cap:
                self.data[c] = self.data[c][-self.cap:]

    def sample_missing_class(self, missing_class, n=80):
        pool = self.data.get(missing_class, [])
        if not pool:
            return None, None
        take = min(n, len(pool))
        import random as pyrand
        sel = pyrand.sample(pool, take)
        feats = torch.stack([t[0] for t in sel])
        labs = torch.stack([t[1] for t in sel])
        return feats, labs


# ----------------------------- Policy (feature-adaptive per-qubit tokens)
class QASPolicy(nn.Module):
    """
    Takes B x fea_dim inputs and produces per-qubit action logits (B, num_qubits, 4).
    Internally adapts feature vectors to num_qubits tokens, then applies a Transformer.
    """
    def __init__(self, num_qubits=12, fea_dim=256, d_model=48, nhead=6, num_layers=2):
        super().__init__()
        self.num_qubits = num_qubits
        self.fea_dim = fea_dim
        self.adapt = nn.Linear(fea_dim, num_qubits)
        self.token_embed = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.randn(1, num_qubits, d_model) * 0.02)
        self.head = nn.Linear(d_model, 4)  # RX, RY, RZ, CZ

    def forward(self, x):
        tok = self.adapt(x).unsqueeze(-1)        # (B, Q, 1)
        h = self.token_embed(tok)                # (B, Q, d_model)
        h = self.encoder(h + self.pos)
        return self.head(h)                      # (B, Q, 4)


# ============================= TT-ENCODING MODULES =============================
class TTMatrix(nn.Module):
    """
    Tensor-Train matrix mapping:
      R^{prod(n_k)} -> R^{prod(m_k)} with cores in R^{r_{k-1} x n_k x m_k x r_k}
    """
    def __init__(self, in_modes, out_modes, tt_ranks):
        super().__init__()
        assert len(in_modes) == len(out_modes), "in_modes and out_modes must have same length"
        self.in_modes  = list(in_modes)
        self.out_modes = list(out_modes)
        self.d = len(in_modes)
        assert tt_ranks[0] == 1 and tt_ranks[-1] == 1, "tt_ranks must start and end with 1"
        assert len(tt_ranks) == self.d + 1, "len(tt_ranks) must be d+1"
        self.tt_ranks = list(tt_ranks)
        cores = []
        for k in range(self.d):
            r0, r1 = self.tt_ranks[k], self.tt_ranks[k+1]
            nk, mk = self.in_modes[k], self.out_modes[k]
            G = nn.Parameter(0.02 * torch.randn(r0, nk, mk, r1))
            cores.append(G)
        self.cores = nn.ParameterList(cores)

    @property
    def in_features(self):
        return int(np.prod(self.in_modes))

    @property
    def out_features(self):
        return int(np.prod(self.out_modes))

    def forward(self, x):
        B = x.shape[0]
        y = x.view(B, *self.in_modes)  # (B, n1,...,nd)
        left = y.unsqueeze(1)          # (B, 1, n1, ..., nd)
        r_prev = 1
        for k in range(self.d):
            G = self.cores[k]  # (r_{k-1}, n_k, m_k, r_k)
            nk, mk = self.in_modes[k], self.out_modes[k]
            left = left.contiguous().view(B, r_prev, nk, -1)  # (B, r_prev, nk, R)
            R = left.shape[-1]
            left_flat = left.view(B, r_prev*nk, R)
            core_flat = G.view(r_prev*nk, mk*self.tt_ranks[k+1])
            out = torch.matmul(left_flat.transpose(1,2), core_flat)   # (B, R, mk*r_k)
            out = out.view(B, R, mk, self.tt_ranks[k+1]).transpose(1,3)  # (B, r_k, mk, R)
            left = out
            r_prev = self.tt_ranks[k+1]
        left = left.squeeze(1)          # (B, mk, R_final)
        left = left.contiguous().view(B, -1)
        return left.view(B, self.out_features)


class TTEncoder(nn.Module):
    """
    TT layer to produce per-qubit angles. Here: 256 -> 12.
    Modes: in=(4,16,4) [=256], out=(3,2,2) [=12], ranks=(1,2,3,1).
    """
    def __init__(
        self,
        in_dim=256,
        out_dim=12,
        in_modes=(4,16,4),
        out_modes=(3,2,2),
        tt_ranks=(1,2,3,1),
        angle_scale=np.pi,
        post_norm=True
    ):
        super().__init__()
        assert int(np.prod(in_modes)) == in_dim, "in_modes must multiply to in_dim"
        assert int(np.prod(out_modes)) == out_dim, "out_modes must multiply to out_dim"
        self.tt = TTMatrix(in_modes, out_modes, tt_ranks)
        self.post_norm = nn.LayerNorm(out_dim) if post_norm else nn.Identity()
        self.angle_scale = float(angle_scale)

    def forward(self, x):
        y = self.tt(x)                    # (B, out_dim)
        y = self.post_norm(y)
        angles = torch.tanh(y) * self.angle_scale
        return angles


# ----------------------------- Noise model (fast expectation scaling)
@dataclass
class NoiseModel:
    p_depol_1q: float = 0.00
    p_depol_2q: float = 0.00
    p_readout:  float = 0.00
    encoder_jitter_sigma: float = 0.00  # radians, training-time jitter

    def clamp_(self):
        self.p_depol_1q = float(np.clip(self.p_depol_1q, 0.0, 1.0))
        self.p_depol_2q = float(np.clip(self.p_depol_2q, 0.0, 1.0))
        self.p_readout  = float(np.clip(self.p_readout,  0.0, 0.5))
        self.encoder_jitter_sigma = max(0.0, float(self.encoder_jitter_sigma))
        return self


# ----------------------------- QNN builder & classifier
def _count_gates(actions, num_qubits, depth):
    actions = actions.detach().cpu()
    oneq_per_layer = int((actions != 3).sum().item())  # RX/RY/RZ
    cz_per_layer = int((actions == 3).sum().item())
    ring_extra = 1 if (num_qubits > 2 and actions[-1].item() == 3) else 0
    cz_per_layer += ring_extra
    oneq_encoding = num_qubits
    n1 = depth * oneq_per_layer + oneq_encoding
    n2 = depth * cz_per_layer
    return n1, n2, oneq_encoding, oneq_per_layer, cz_per_layer


def build_qnn_from_actions(actions, num_qubits=12, depth=2):
    class GeneratedQNN(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.num_qubits = num_qubits
            self.depth = depth
            self.actions = actions.detach().cpu().tolist()

        def _encode(self, qdev, angles):
            for i in range(self.num_qubits):
                tq.RY(has_params=False)(qdev, wires=i, params=angles[:, i])

        def _layer(self, qdev):
            for i, a in enumerate(self.actions):
                if a == 0: tq.RX(has_params=True, trainable=True, init_params=0.05)(qdev, wires=i)
                elif a == 1: tq.RY(has_params=True, trainable=True, init_params=0.05)(qdev, wires=i)
                elif a == 2: tq.RZ(has_params=True, trainable=True, init_params=0.05)(qdev, wires=i)
            for i, a in enumerate(self.actions):
                if a == 3 and i < self.num_qubits - 1:
                    tq.CZ()(qdev, wires=[i, i + 1])
            if self.num_qubits > 2 and self.actions[-1] == 3:
                tq.CZ()(qdev, wires=[self.num_qubits - 1, 0])

        def forward(self, qdev, angles):
            self._encode(qdev, angles)
            for _ in range(self.depth):
                self._layer(qdev)

    qlayer = GeneratedQNN()
    n1, n2, n_enc, n1_per_layer, n2_per_layer = _count_gates(actions, num_qubits, depth)
    gate_stats = {
        "n1_total": n1, "n2_total": n2, "n1_encoding": n_enc,
        "n1_per_layer": n1_per_layer, "n2_per_layer": n2_per_layer,
        "depth": depth, "num_qubits": num_qubits,
    }
    return qlayer, gate_stats


class QNNClassifier(nn.Module):
    def __init__(
        self,
        qlayer: tq.QuantumModule,
        gate_stats: dict,
        num_qubits=12,
        num_classes=2,
        fea_dim=256,
        use_tt_encoding=True,
        tt_in_modes=(4,16,4),
        tt_out_modes=(3,2,2),
        tt_ranks=(1,2,3,1),
        angle_scale=np.pi,
        noise: NoiseModel = NoiseModel(),
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.use_tt = use_tt_encoding
        self.gate_stats = gate_stats
        self.noise = noise.clamp_()

        self.pre = nn.Sequential(
            nn.LayerNorm(fea_dim),
            nn.Linear(fea_dim, fea_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        if self.use_tt:
            self.tt_enc = TTEncoder(
                in_dim=fea_dim,
                out_dim=num_qubits,
                in_modes=tt_in_modes,
                out_modes=tt_out_modes,
                tt_ranks=tt_ranks,
                angle_scale=angle_scale,
                post_norm=True
            )
            self.proj = None
        else:
            self.tt_enc = None
            # lightweight projection to 12 angles if input dim != num_qubits
            self.proj = nn.Linear(fea_dim, num_qubits) if fea_dim != num_qubits else None

        self.q_layer = qlayer
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.fc = nn.Linear(num_qubits, num_classes)

    def forward(self, x):
        x = self.pre(x)
        if self.use_tt and self.tt_enc is not None:
            angles = self.tt_enc(x)              # (B, num_qubits) in radians
        else:
            angles = torch.tanh(x)               # [-1,1]
            if self.proj is not None:
                angles = self.proj(angles)
            angles = torch.tanh(angles) * np.pi  # [-pi, pi]

        # Encoder jitter during training
        if self.noise.encoder_jitter_sigma > 0.0 and self.training:
            angles = angles + torch.randn_like(angles) * self.noise.encoder_jitter_sigma

        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.fc.in_features, bsz=bsz, device=x.device)
        self.q_layer(qdev, angles)
        z_true = self.measure(qdev)  # (B, num_qubits)

        # Fast depolarizing scaling
        n1 = self.gate_stats["n1_total"]
        n2 = self.gate_stats["n2_total"]
        a1 = (1.0 - self.noise.p_depol_1q) ** max(0, n1)
        a2 = (1.0 - self.noise.p_depol_2q) ** max(0, n2)
        z_noisy = z_true * (a1 * a2)

        # Readout symmetric flip
        if self.noise.p_readout > 0.0:
            z_noisy = z_noisy * (1.0 - 2.0 * self.noise.p_readout)

        return self.fc(z_noisy)


# ----------------------------- EWC (for CL)
class EWCLoss:
    def __init__(self, model: nn.Module, dataloader, device='cpu', lambda_ewc=60.0):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fim = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters()}
        self.model.eval()
        for xb, _ in dataloader:
            xb = xb.to(self.device)
            self.model.zero_grad()
            logits = self.model(xb)
            if logits.dim() == 3:  # policy-like
                probs = torch.softmax(logits, dim=-1)
                pm = probs.mean(dim=0)
                dists = [torch.distributions.Categorical(pm[q]) for q in range(pm.size(0))]
                actions = torch.stack([d.sample() for d in dists])
                logp = sum(d.log_prob(a) for d, a in zip(dists, actions))
                (-logp).backward()
            else:
                dummy_y = torch.randint(0, logits.size(-1), (logits.size(0),), device=logits.device)
                nn.functional.cross_entropy(logits, dummy_y).backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fim[n] += (p.grad.detach() ** 2) / len(dataloader)
        return fim

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            loss = loss + (self.fisher[n] * (p - self.params[n])**2).sum()
        return self.lambda_ewc * loss


# ----------------------------- Losses, metrics, sampler
def make_class_weighted_ce(y, device, n_classes=2):
    counts = torch.bincount(y, minlength=n_classes).float()
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return nn.CrossEntropyLoss(weight=weights.to(device)), weights

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
    def forward(self, logits, y):
        ce = nn.functional.cross_entropy(logits, y, weight=self.weight, reduction='none')
        p = torch.softmax(logits, dim=1).gather(1, y.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1-1e-6)
        mod = (1 - p) ** self.gamma
        return (mod * ce).mean()

def maybe_make_sampler(y, n_classes=2):
    counts = torch.bincount(y, minlength=n_classes).float()
    if (counts > 0).sum() < 2:
        return None
    class_weights = 1.0 / (counts + 1e-6)
    sample_weights = class_weights[y]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(y), replacement=True)

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    acc = (np.trace(cm) / np.sum(cm)) if cm.size else 0.0
    with numpy_errstate():
        tpr0 = cm[0,0] / max(1, cm[0].sum())
        tpr1 = cm[1,1] / max(1, cm[1].sum())
        bacc = 0.5 * (tpr0 + tpr1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    return cm, acc, bacc, f1

@contextmanager
def numpy_errstate():
    with np.errstate(divide='ignore', invalid='ignore'):
        yield


# ----------------------------- Policy helpers
def apply_cz_prior(pm, max_expected_cz=0.30, scale=0.5, eps=1e-8):
    expected_cz = pm[:, 3].mean()
    if expected_cz.detach().item() > max_expected_cz:
        pm = pm.clone()
        pm[:, 3] = pm[:, 3] * scale
        pm = pm / (pm.sum(dim=-1, keepdim=True) + eps)
    return pm

def greedy_actions_from_policy(policy, x_batch):
    with torch.no_grad():
        logits = policy(x_batch)
        pm = torch.softmax(logits, dim=-1).mean(dim=0)
        pm = apply_cz_prior(pm)
        actions = torch.argmax(pm, dim=-1)
    return actions

def sample_actions_and_logp(policy, x_batch, eps=1e-8):
    logits = policy(x_batch)
    pm = torch.softmax(logits, dim=-1).mean(dim=0)
    pm = apply_cz_prior(pm)
    pm = pm.clamp_min(eps)
    pm = pm / pm.sum(dim=-1, keepdim=True)

    dists = [torch.distributions.Categorical(pm[q]) for q in range(pm.size(0))]
    actions = torch.stack([d.sample() for d in dists])
    logp = sum(d.log_prob(a) for d, a in zip(dists, actions))
    ent = -(pm * pm.log()).sum(dim=-1).mean()
    uni = torch.full_like(pm, 1.0 / pm.size(-1))
    kl = torch.sum(pm * (pm.log() - uni.log()))
    return actions, logp, ent, kl

def reward_from_metrics(bacc, f1, actions, num_qubits):
    cz_frac = float((actions == 3).sum().item()) / num_qubits
    return 0.6 * bacc + 0.4 * f1 - 0.05 * cz_frac, cz_frac


# ----------------------------- Train / Evaluate
def train_qnn_fixed_arch(qnn, loader, qnn_loss, device='cpu', epochs=30, lr=3e-3):
    qnn.train()
    opt = optim.Adam(qnn.parameters(), lr=lr)
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = qnn(xb)
            loss = qnn_loss(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
        ce = loss_sum / max(1,total)
        acc = correct / max(1,total)
        print(f"    [QNN] Epoch {ep+1:02d} | loss={ce:.4f} | acc={acc:.4f}")

def evaluate_model(qnn, loader, device='cpu'):
    y_true, y_pred = [], []
    qnn.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = qnn(xb)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(yb.cpu().tolist())
    return compute_metrics(y_true, y_pred)


# ----------------------------- Methods: Naive VQC & QAS runners
def run_naive_vqc(X, y, tr_idx, va_idx, te_idx, num_qubits, device,
                  epochs=30, lr=3e-3, batch_size=128, noise: NoiseModel = NoiseModel()):
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    sampler = maybe_make_sampler(ytr, n_classes=2)
    _, w = make_class_weighted_ce(ytr, device=device, n_classes=2)
    w = torch.clamp(w, max=3.0)
    qnn_loss = FocalLoss(weight=w, gamma=2.0)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                              sampler=sampler if sampler is not None else None,
                              shuffle=(sampler is None))
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    actions = torch.tensor([1]*(num_qubits-1) + [3], dtype=torch.long, device=device)
    qlayer, gate_stats = build_qnn_from_actions(actions, num_qubits=num_qubits, depth=2)
    qlayer = qlayer.to(device)

    qnn = QNNClassifier(
        qlayer, gate_stats, num_qubits=num_qubits, num_classes=2, fea_dim=X.shape[1],
        use_tt_encoding=True,
        tt_in_modes=(4,16,4), tt_out_modes=(3,2,2), tt_ranks=(1,2,3,1),
        angle_scale=np.pi,
        noise=noise,
    ).to(device)

    print(f"  [naive-vqc] Training QNN (fixed circuit, TT-encoding 256->12) with noise={noise} ...")
    train_qnn_fixed_arch(qnn, train_loader, qnn_loss, device=device, epochs=epochs, lr=lr)

    cm, acc, bacc, f1 = evaluate_model(qnn, test_loader, device=device)
    print(f"  [naive-vqc] Test | acc={acc:.4f} | bAcc={bacc:.4f} | F1={f1:.4f}\n  Confusion:\n{cm}")
    return {"acc":acc, "bAcc":bacc, "F1":f1, "cm":cm.tolist()}

def reinforce_update(policy, xb_pol, base_reward, ewc_prev=None, device='cpu',
                     lr=2e-3, beta_kl=0.01, K=4, use_ewc=False, entropy_bonus=True):
    policy.train()
    opt = optim.Adam(policy.parameters(), lr=lr)

    logps, ents, kls = [], [], []
    for _ in range(K):
        actions, logp, ent, kl = sample_actions_and_logp(policy, xb_pol)
        logps.append(logp); ents.append(ent); kls.append(kl)

    logp = torch.stack(logps).mean()
    ent  = torch.stack(ents).mean()
    kl   = torch.stack(kls).mean()

    ewc_pen = (ewc_prev.penalty(policy) if (use_ewc and ewc_prev is not None) else 0.0)
    L = -(base_reward * logp) + beta_kl * kl + ewc_pen
    if entropy_bonus:
        L = L - 0.01 * ent

    opt.zero_grad()
    L.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()

def run_qas_on_record(X, y, tr_idx, va_idx, te_idx, policy, device, num_qubits,
                      epochs=30, lr=3e-3, batch_size=128, use_cl=False,
                      replay=None, prev_ewc=None, ewc_lambda_good=60.0, ewc_lambda_weak=24.0,
                      noise: NoiseModel = NoiseModel(),
                      use_tt_encoding=True):
    """
    Run one QAS task with or without TT-encoding. Returns metrics and updated EWC.
    """
    t0 = time.time()

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]
    Xte, yte = X[te_idx], y[te_idx]

    if use_cl:
        counts_tr = torch.bincount(ytr, minlength=2)
        missing = (counts_tr == 0).nonzero(as_tuple=True)[0].tolist()
        for mcls in missing:
            Xbuf, ybuf = replay.sample_missing_class(mcls, n=120) if replay else (None, None)
            if Xbuf is not None:
                Xtr = torch.cat([Xtr, Xbuf], dim=0)
                ytr = torch.cat([ytr, ybuf], dim=0)
                print(f"  [Replay] Added {len(ybuf)} samples for missing class {mcls}")

    sampler = maybe_make_sampler(ytr, n_classes=2)
    _, w = make_class_weighted_ce(ytr, device=device, n_classes=2)
    w = torch.clamp(w, max=3.0)
    qnn_loss = FocalLoss(weight=w, gamma=2.0)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                              sampler=sampler if sampler is not None else None,
                              shuffle=(sampler is None))
    val_loader  = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        xb_small = Xtr[:min(128, len(Xtr))].to(device)
        actions = greedy_actions_from_policy(policy, xb_small)

    qlayer, gate_stats = build_qnn_from_actions(actions, num_qubits=num_qubits, depth=2)
    qlayer = qlayer.to(device)
    qnn = QNNClassifier(
        qlayer, gate_stats, num_qubits=num_qubits, num_classes=2, fea_dim=X.shape[1],
        use_tt_encoding=use_tt_encoding,
        tt_in_modes=(4,16,4), tt_out_modes=(3,2,2), tt_ranks=(1,2,3,1),
        angle_scale=np.pi,
        noise=noise,
    ).to(device)

    label_tt = "TT-encoding" if use_tt_encoding else "no TT"
    print(f"  Training QNN ({label_tt}) with noise={noise} ...")
    train_qnn_fixed_arch(qnn, train_loader, qnn_loss, device=device, epochs=epochs, lr=lr)

    # Test metrics
    cm, acc, bacc, f1 = evaluate_model(qnn, test_loader, device=device)

    # Validation reward for policy update
    cm_v, _, bacc_v, f1_v = evaluate_model(qnn, val_loader, device=device)
    base_reward, cz_frac = reward_from_metrics(bacc_v, f1_v, actions, num_qubits)

    # Policy update (REINFORCE-like)
    xb_pol = Xtr[:min(128, len(Xtr))].to(device)
    reinforce_update(
        policy, xb_pol,
        base_reward=torch.tensor(base_reward, dtype=torch.float32, device=device),
        ewc_prev=prev_ewc, device=device, lr=2e-3, beta_kl=0.01, K=4,
        use_ewc=use_cl, entropy_bonus=True
    )

    next_ewc = prev_ewc
    if use_cl:
        lam = ewc_lambda_good if base_reward >= 0.5 else ewc_lambda_weak
        next_ewc = EWCLoss(policy, DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=False),
                           device=device, lambda_ewc=lam)
        if replay is not None:
            replay.add(Xtr, ytr, take_per_class=100)

    runtime = time.time() - t0
    print(f"  Test | acc={acc:.4f} | bAcc={bacc:.4f} | F1={f1:.4f} | runtime={runtime:.1f}s")
    return {
        "acc": acc,
        "bAcc": bacc,
        "F1": f1,
        "cm": cm.tolist(),
        "reward": float(base_reward),
        "cz_frac": float(cz_frac),
        "runtime": runtime
    }, next_ewc


# ----------------------------- Task selection
def select_tasks(candidate_records, max_tasks=8, min_per_class=10):
    picked = []
    for rec in candidate_records:
        try:
            X, y = extract_record_dataset(rec, max_beats=800, window_sec=0.6, min_per_class=min_per_class)
            picked.append((rec, X, y))
            print(f"[task-select] Record {rec} ok. counts={torch.bincount(y, minlength=2).tolist()}")
            if len(picked) >= max_tasks:
                break
        except Exception as e:
            print(f"[task-select] Skip {rec}: {e}")
    return picked


# ----------------------------- Summaries
def _mean_std(arr):
    a = np.array(arr, dtype=float)
    return float(np.mean(a)), float(np.std(a))

def summarize(name, results, show_reward_cz=True, show_runtime=False):
    if not results:
        print(f"\n{name}: no results")
        return
    accs  = [r["acc"]  for r in results]
    baccs = [r["bAcc"] for r in results]
    f1s   = [r["F1"]   for r in results]
    print(f"\n================= Summary ({name}) over {len(results)} tasks =================")
    for i, r in enumerate(results, 1):
        parts = [f"acc={r['acc']:.3f}", f"bAcc={r['bAcc']:.3f}", f"F1={r['F1']:.3f}"]
        if show_reward_cz and ("reward" in r):
            parts.append(f"reward={r['reward']:.4f}")
        if show_reward_cz and ("cz_frac" in r):
            parts.append(f"cz_frac={r['cz_frac']:.2f}")
        if show_runtime and ("runtime" in r):
            parts.append(f"runtime={r['runtime']:.1f}s")
        print(f"  task{i}: " + " | ".join(parts))
    acc_m, acc_s   = _mean_std(accs)
    bacc_m, bacc_s = _mean_std(baccs)
    f1_m, f1_s     = _mean_std(f1s)
    summary = f"  AVG:   acc={acc_m:.3f}±{acc_s:.3f} | bAcc={bacc_m:.3f}±{bacc_s:.3f} | F1={f1_m:.3f}±{f1_s:.3f}"
    if show_runtime:
        rts = [r.get("runtime", np.nan) for r in results if "runtime" in r]
        if len(rts) > 0:
            rt_m, rt_s = _mean_std(rts)
            summary += f" | runtime={rt_m:.1f}±{rt_s:.1f}s"
    if show_reward_cz:
        rews  = [r.get("reward", np.nan) for r in results if "reward" in r]
        czs   = [r.get("cz_frac", np.nan) for r in results if "cz_frac" in r]
        if len(rews) > 0 and not np.isnan(rews).all():
            rw_m, rw_s = _mean_std(rews)
            summary += f" | reward={rw_m:.4f}±{rw_s:.4f}"
        if len(czs) > 0 and not np.isnan(czs).all():
            cz_m, cz_s = _mean_std(czs)
            summary += f" | cz_frac={cz_m:.2f}±{cz_s:.2f}"
    print(summary)


# ----------------------------- Main
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_qubits = 12
    batch_size = 128
    epochs = 30
    lr = 3e-3

    # Noise defaults (set to 0 for noiseless if desired)
    noise_defaults = NoiseModel(
        p_depol_1q=0.001,
        p_depol_2q=0.001,
        p_readout=0.01,
        encoder_jitter_sigma=0.01,
    )

    candidate_records = [105, 106, 109, 114, 116, 119, 200, 201]
    max_tasks = 8
    picked = select_tasks(candidate_records, max_tasks=max_tasks, min_per_class=10)
    if len(picked) < 2:
        print("Not enough valid tasks found. Please check your PhysioNet access or candidate list.")
        return

    print(f"\n==> Using {len(picked)} tasks: {[rec for rec,_,_ in picked]}")

    fea_dim = picked[0][1].shape[1]  # should be 256

    policy_nocl = QASPolicy(num_qubits=num_qubits, fea_dim=fea_dim, d_model=48, nhead=6).to(device)
    policy_cl   = QASPolicy(num_qubits=num_qubits, fea_dim=fea_dim, d_model=48, nhead=6).to(device)

    replay = ReplayBuffer(per_class_cap=500)
    prev_ewc = None

    sum_naive, sum_qas, sum_cl_tt, sum_cl_nott = [], [], [], []

    for t_idx, (rec, X, y) in enumerate(picked, start=1):
        print(f"\n================= Task {t_idx} | Record {rec} (N vs V) =================")

        tr_idx, te_idx = stratified_split_indices(y, train_ratio=0.8, seed=1000 + t_idx)
        tr2_idx, va_idx = stratified_subsplit_indices(y, base_indices=tr_idx, sub_train_ratio=0.85, seed=2000 + t_idx)

        print("\n================= Baseline: naive-vqc =================")
        res_naive = run_naive_vqc(X, y, tr2_idx, va_idx, te_idx, num_qubits, device,
                                  epochs=epochs, lr=lr, batch_size=batch_size,
                                  noise=noise_defaults)
        sum_naive.append(res_naive)

        print("\n================= Baseline: qas-no-cl =================")
        res_qas, _ = run_qas_on_record(
            X, y, tr2_idx, va_idx, te_idx,
            policy=policy_nocl, device=device, num_qubits=num_qubits,
            epochs=epochs, lr=lr, batch_size=batch_size,
            use_cl=False, replay=None, prev_ewc=None,
            ewc_lambda_good=60.0, ewc_lambda_weak=24.0,
            noise=noise_defaults, use_tt_encoding=True
        )
        sum_qas.append(res_qas)

        print("\n================= Proposed: cl-qas (with TT-encoding) =================")
        res_cl_tt, prev_ewc = run_qas_on_record(
            X, y, tr2_idx, va_idx, te_idx,
            policy=policy_cl, device=device, num_qubits=num_qubits,
            epochs=epochs, lr=lr, batch_size=batch_size,
            use_cl=True, replay=replay, prev_ewc=prev_ewc,
            ewc_lambda_good=60.0, ewc_lambda_weak=24.0,
            noise=noise_defaults, use_tt_encoding=True
        )
        sum_cl_tt.append(res_cl_tt)

        print("\n================= Ablation: cl-qas (without TT-encoding) =================")
        # Note: we DO NOT update prev_ewc from the no-TT run; we ablate architecture only.
        res_cl_nott, _ = run_qas_on_record(
            X, y, tr2_idx, va_idx, te_idx,
            policy=policy_cl, device=device, num_qubits=num_qubits,
            epochs=epochs, lr=lr, batch_size=batch_size,
            use_cl=True, replay=replay, prev_ewc=prev_ewc,
            ewc_lambda_good=60.0, ewc_lambda_weak=24.0,
            noise=noise_defaults, use_tt_encoding=False
        )
        sum_cl_nott.append(res_cl_nott)

    # --------- Summaries
    summarize("naive-vqc", sum_naive, show_reward_cz=False, show_runtime=False)
    summarize("qas-no-cl",  sum_qas,   show_reward_cz=True,  show_runtime=True)
    summarize("cl-qas (with TT-encoding)", sum_cl_tt, show_reward_cz=True, show_runtime=True)
    summarize("cl-qas (without TT-encoding)", sum_cl_nott, show_reward_cz=True, show_runtime=True)

    # Optional: quick paired deltas (TT - noTT)
    if len(sum_cl_tt) == len(sum_cl_nott) and len(sum_cl_tt) > 0:
        d_acc  = [a["acc"]  - b["acc"]  for a,b in zip(sum_cl_tt, sum_cl_nott)]
        d_bacc = [a["bAcc"] - b["bAcc"] for a,b in zip(sum_cl_tt, sum_cl_nott)]
        d_f1   = [a["F1"]   - b["F1"]   for a,b in zip(sum_cl_tt, sum_cl_nott)]
        d_rt   = [a["runtime"] - b["runtime"] for a,b in zip(sum_cl_tt, sum_cl_nott)]
        print("\n=== CL-QAS (TT) minus CL-QAS (no TT) deltas (per task) ===")
        for i,(da,db,df,dr) in enumerate(zip(d_acc,d_bacc,d_f1,d_rt),1):
            print(f" task{i}: Δacc={da:+.3f} | ΔbAcc={db:+.3f} | ΔF1={df:+.3f} | Δruntime={dr:+.1f}s")
        def ms(x): return f"{np.mean(x):+.3f}±{np.std(x):.3f}"
        print(f" AVG Δ: acc={ms(d_acc)} | bAcc={ms(d_bacc)} | F1={ms(d_f1)} | runtime={np.mean(d_rt):+.1f}±{np.std(d_rt):.1f}s")


if __name__ == "__main__":
    main()

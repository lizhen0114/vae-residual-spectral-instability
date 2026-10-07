#!/usr/bin/env python3
"""
Two-stage VAE continuation on a covariance-matched symmetric Gaussian mixture.

Standalone adaptation of vae_two_stage_continuation_lambda_biascorrected_v2.py.
The model, loss, continuation training, active-set classification, collapsed
coordinate marginalization, spectral estimator, extrapolation and bootstrap
are unchanged. Only data generation and configuration/orchestration differ.

Run: python vae_two_stage_continuation_lambda_biascorrected_mixture.py \
         --config experiment_mixture.json
Requirements: torch, numpy, pandas, matplotlib.
"""

import argparse
import json
import math
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, asdict, replace, fields
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt


# ----------------------------- configuration -----------------------------

@dataclass
class Config:
    dataset: str = "symmetric_gaussian_mixture"
    data_seed: int = 1234
    mixture_a: float = 0.5
    rotate_data: bool = True
    n_samples: int = 256_000
    data_dim: int = 8
    latent_dim: int = 4
    spectrum_decay: float = 0.1
    hidden_enc: Tuple[int, int] = (256, 128)
    hidden_dec: Tuple[int, int] = (128, 256)
    batch_size: int = 512
    epochs: int = 80
    lr: float = 1e-3
    seeds: Tuple[int, ...] = tuple(range(10))
    s_values: Tuple[float, ...] = tuple(sorted(set(np.round(np.concatenate([
        np.arange(0.68, 0.7701, 0.01),
        np.arange(0.78, 0.8501, 0.01),
        np.arange(0.87, 0.9301, 0.01),
        np.arange(0.94, 1.0301, 0.01),
    ]), 6).tolist())))
    active_kl_threshold: float = 1e-3
    active_var_tol: float = 2e-2
    eval_points: int = 512
    z_mc: int = 96
    collapsed_mc: int = 64
    collapsed_decode_batch: int = 256
    power_iters: int = 35
    power_tol: float = 1e-6
    importance_chunk: int = 128
    outdir: str = "results_continuation"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    gpu_resident_data: bool = True
    use_tf32: bool = True
    use_amp: bool = True
    compile_model: bool = False


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def configure_cuda(cfg):
    """Performance-oriented defaults for Ada GPUs such as NVIDIA L40S."""
    if not torch.cuda.is_available() or not str(cfg.device).startswith("cuda"):
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.use_tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.use_tf32)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high" if cfg.use_tf32 else "highest")
    except Exception:
        pass


# ----------------------------- data --------------------------------------

def make_symmetric_gaussian_mixture(cfg: Config):
    """Covariance-matched, equally weighted two-component Gaussian mixture.

    v = ones(8)/sqrt(8), G ~ N(0,I), B uniform on {-1,+1}, independent.
    Y = (I - a**2 v v.T)**(1/2) G + a B v; X = Q D**(1/2) Y.
    The population covariance is Q D Q.T for every 0 <= a < 1.
    Only the component along v changes; no dense matrix square root is needed.
    Q and G use the baseline's draw order. At a=0, the default rotated
    dataset exactly reproduces make_synthetic(cfg, seed=cfg.data_seed).
    Sample centering follows the baseline; no empirical whitening is applied.
    """
    if cfg.data_dim != 8:
        raise ValueError('symmetric_gaussian_mixture requires data_dim=8.')
    a = cfg.mixture_a
    if not math.isfinite(a) or not 0 <= a < 1:
        raise ValueError('mixture_a must satisfy 0 <= mixture_a < 1.')
    g = torch.Generator().manual_seed(cfg.data_seed)
    eigvals = torch.exp(-cfg.spectrum_decay * torch.arange(8, dtype=torch.float32))
    Q, _ = torch.linalg.qr(torch.randn(8, 8, generator=g))
    if not cfg.rotate_data:
        Q = torch.eye(8)
    gaussian = torch.randn(cfg.n_samples, 8, generator=g)
    standardized = gaussian
    if a != 0:
        v = torch.ones(8, dtype=gaussian.dtype) / math.sqrt(8)
        signs = torch.randint(0, 2, (cfg.n_samples,), generator=g).to(gaussian.dtype) * 2 - 1
        shift = (math.sqrt(1 - a*a) - 1) * (gaussian @ v) + a * signs
        standardized = gaussian + shift[:, None] * v
    L = Q @ torch.diag(torch.sqrt(eigvals))
    x = standardized @ L.T
    x -= x.mean(0, keepdim=True)
    return x, eigvals.numpy()


# ----------------------------- model -------------------------------------

class MLPVAE(nn.Module):
    def __init__(self, data_dim: int, latent_dim: int,
                 hidden_enc=(256, 128), hidden_dec=(128, 256)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_enc[0]), nn.ReLU(),
            nn.Linear(hidden_enc[0], hidden_enc[1]), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_enc[1], latent_dim)
        self.logvar_head = nn.Linear(hidden_enc[1], latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dec[0]), nn.ReLU(),
            nn.Linear(hidden_dec[0], hidden_dec[1]), nn.ReLU(),
            nn.Linear(hidden_dec[1], data_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h).clamp(-12.0, 8.0)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar


def vae_loss(x, recon, mu, logvar, s: float):
    # Constants independent of parameters omitted.
    rec = 0.5 / s * (x - recon).pow(2).sum(dim=1)
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return (rec + kl_dim.sum(dim=1)).mean(), rec.mean(), kl_dim.mean(dim=0)



# ---------------------- free-continuation helpers ------------------------

@contextmanager
def loader_batches(loader):
    """Close worker queues even when training/statistics raises mid-epoch.

    These short-lived loaders deliberately use persistent_workers=False:
    older PyTorch versions retain worker process handles through atexit when
    persistent workers and pinned memory are enabled together. Recreating that
    combination at every continuation point can exhaust file descriptors.
    PyTorch has no public iterator close API; keep its guarded, idempotent
    shutdown hook isolated here. Single-process iterators need no cleanup.
    """
    iterator = iter(loader)
    try:
        yield iterator
    finally:
        shutdown = getattr(iterator, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()


def resident_batches(x, batch_size, shuffle):
    """Batch an already resident tensor; keep the final incomplete batch.

    A new device-side permutation is drawn once per training epoch. It uses
    PyTorch's device RNG, which is included in continuation checkpoints.
    Sequential statistics use views and do not draw random numbers.
    """
    if shuffle:
        order = torch.randperm(len(x), device=x.device)
        for start in range(0, len(x), batch_size):
            yield (x.index_select(0, order[start:start + batch_size]),)
    else:
        for start in range(0, len(x), batch_size):
            yield (x[start:start + batch_size],)


def training_data_loader(x, cfg, batch_size, shuffle):
    if cfg.gpu_resident_data and x.device.type == 'cuda':
        return None
    return DataLoader(
        TensorDataset(x), batch_size=batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and torch.device(cfg.device).type == 'cuda'),
        persistent_workers=False, drop_last=False,
    )


def epoch_batches(loader, x, batch_size, shuffle):
    if loader is None:
        return nullcontext(resident_batches(x, batch_size, shuffle))
    return loader_batches(loader)


def train_free_step(model, x, cfg, s, epochs, lr=None):
    """Optimize one s point, warm-starting the same unconstrained VAE."""
    device = torch.device(cfg.device)
    model.to(device)
    # train_trajectory supplies the same GPU tensor at every scan point.
    if cfg.gpu_resident_data and device.type == 'cuda':
        x = x.to(device)
    loader = training_data_loader(x, cfg, cfg.batch_size, shuffle=True)
    kwargs = dict(lr=(cfg.lr if lr is None else lr))
    if device.type == "cuda": kwargs["fused"] = True
    try:
        opt = torch.optim.Adam(model.parameters(), **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None); opt = torch.optim.Adam(model.parameters(), **kwargs)
    amp = bool(cfg.use_amp and device.type == "cuda")
    history=[]; model.train()
    for _ in range(epochs):
        # Detached double accumulators preserve the original Python-double
        # reporting precision without reading CUDA scalars every minibatch.
        totals = torch.zeros(2, device=device, dtype=torch.float64)
        n = 0
        with epoch_batches(loader, x, cfg.batch_size, shuffle=True) as batches:
            for (xb,) in batches:
                xb=xb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    recon,mu,lv=model(xb)
                    loss,rec,_=vae_loss(xb,recon,mu,lv,s)
                loss.backward(); opt.step()
                b = xb.shape[0]
                totals.add_(torch.stack((loss.detach(), rec.detach())).double(), alpha=b)
                n += b
        history.append(tuple((totals / n).cpu().tolist()))
    return history

@torch.no_grad()
def free_latent_statistics(model,x,cfg):
    device=torch.device(cfg.device)
    if cfg.gpu_resident_data and device.type == 'cuda':
        x = x.to(device)
    loader = training_data_loader(x, cfg, 4096, shuffle=False)
    sv=torch.zeros(cfg.latent_dim,device=device); sm=torch.zeros_like(sv); sk=torch.zeros_like(sv); n=0
    model.eval()
    with epoch_batches(loader, x, 4096, shuffle=False) as batches:
        for (xb,) in batches:
            xb=xb.to(device,non_blocking=True); mu,lv=model.encode(xb); var=lv.exp()
            kl=.5*(mu.pow(2)+var-1-lv)
            sv+=var.sum(0); sm+=mu.pow(2).sum(0); sk+=kl.sum(0); n+=xb.shape[0]
    return (sv/n).cpu(),(sm/n).cpu(),(sk/n).cpu()

@torch.no_grad()
def _log_q_diag(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    """z [L,A], mu/logvar [B,A] -> log q [L,B]."""
    if mu.shape[1] == 0:
        return torch.zeros(z.shape[0], mu.shape[0], device=mu.device)
    diff = z[:, None, :] - mu[None, :, :]
    return -0.5 * (diff.pow(2) / logvar.exp()[None, :, :] + logvar[None, :, :] + math.log(2*math.pi)).sum(-1)


@torch.no_grad()
def marginal_decode(model, zA, active_mask, cfg, seed):
    """Project the decoder to E_{Z_C~N(0,I)} f(z_A, Z_C), in FP32.

    Average outputs BEFORE forming the residual operator or its Gram matrix.
    Common Gaussian draws across z_A define one reproducible projected decoder;
    they also make the fully collapsed decoder constant across quadrature rows.
    Finite-K integration error is not removed by the B,L extrapolation: check
    convergence separately with collapsed_mc. This projection alone does not
    establish stationarity of the original network.
    """
    if cfg.collapsed_mc < 1 or cfg.collapsed_decode_batch < 1:
        raise ValueError('collapsed_mc and collapsed_decode_batch must be positive.')
    device = zA.device
    mask = active_mask.to(device=device, dtype=torch.bool)
    active_idx = torch.where(mask)[0]
    collapsed_idx = torch.where(~mask)[0]
    n_rows = zA.shape[0]
    # For A empty there is only one projected decoder value.
    points = zA[:1] if active_idx.numel() == 0 else zA
    K = cfg.collapsed_mc if collapsed_idx.numel() else 1
    generator = torch.Generator().manual_seed(seed + 111)
    samples = torch.randn(K, collapsed_idx.numel(), generator=generator).to(device)
    outputs = []
    with torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(points), cfg.collapsed_decode_batch):
            za = points[start:start + cfg.collapsed_decode_batch].float()
            n = len(za)
            sample_chunk = max(1, cfg.collapsed_decode_batch // n)
            total = None
            for k in range(0, K, sample_chunk):
                eps = samples[k:k + sample_chunk]
                z = torch.zeros(n, len(eps), cfg.latent_dim,
                                device=device, dtype=torch.float32)
                z[:, :, active_idx] = za[:, None, :]
                z[:, :, collapsed_idx] = eps[None, :, :]
                decoded = model.decode(z.reshape(-1, cfg.latent_dim)).float()
                subtotal = decoded.reshape(n, len(eps), -1).sum(dim=1)
                total = subtotal if total is None else total + subtotal
            outputs.append(total / K)
    result = torch.cat(outputs, dim=0)
    return result.expand(n_rows, -1) if active_idx.numel() == 0 else result


@torch.no_grad()
def estimate_lambda_max(model, x: torch.Tensor, active_mask: torch.Tensor,
                        cfg: Config, seed: int = 0):
    """
    GPU-batched estimator of Lambda_max(C_A), optimized for L40S.

    T has shape [L,D,B], T[l,i,b] = w[l,b] R[b,i; z_l].
    It remains on GPU.  Power iteration applies
        C h = (B/L) sum_l T_l^T (T_l h)
    using batched einsum, eliminating the previous CPU/Python-loop bottleneck.

    The residual uses the standard-normal z_C-averaged decoder, not f(z_A,0).
    Spectrum arithmetic stays float32 even when VAE training uses BF16 AMP.
    """
    seed_all(seed + 100003)
    device = torch.device(cfg.device)
    model.eval()

    B = min(cfg.eval_points, x.shape[0])
    gen = torch.Generator().manual_seed(seed + 77)
    idx = torch.randperm(x.shape[0], generator=gen)[:B]
    xb = x[idx].to(device, non_blocking=True).float()

    # Force FP32 for the stability estimator.
    with torch.autocast(device_type=device.type, enabled=False):
        mu_all, lv_all = model.encode(xb)
        mu_all, lv_all = mu_all.float(), lv_all.float()

    active_idx = torch.where(active_mask.to(device))[0]
    A = int(active_idx.numel())
    L = cfg.z_mc

    owner = torch.randint(0, B, (L,),
                          generator=torch.Generator().manual_seed(seed + 88)).to(device)
    if A > 0:
        muA, lvA = mu_all[:, active_idx], lv_all[:, active_idx]
        eps = torch.randn(L, A,
                          generator=torch.Generator().manual_seed(seed + 99)).to(device)
        zA = muA[owner] + torch.exp(0.5 * lvA[owner]) * eps
    else:
        muA, lvA = mu_all[:, :0], lv_all[:, :0]
        zA = torch.empty(L, 0, device=device)

    # Marginalize collapsed standard-normal coordinates before building T.
    f_projected = marginal_decode(model, zA, active_mask, cfg, seed)
    T_chunks = []
    for st in range(0, L, cfg.importance_chunk):
        en = min(L, st + cfg.importance_chunk)
        zc = zA[st:en]
        C = en - st
        logq = _log_q_diag(zc, muA, lvA).float()
        w = torch.softmax(logq, dim=1)  # [C,B]

        f_l = f_projected[st:en]                         # [C,D]
        residual = xb.unsqueeze(0) - f_l.unsqueeze(1)      # [C,B,D]
        T_chunks.append((w[:, :, None] * residual).permute(0,2,1).contiguous())

    T = torch.cat(T_chunks, dim=0)  # [L,D,B], FP32 CUDA

    def apply_C(h):
        # y[l,d] = sum_b T[l,d,b] h[b]
        y = torch.einsum("ldb,b->ld", T, h)
        # out[b] = (B/L) sum_ld T[l,d,b] y[l,d]
        return (B / L) * torch.einsum("ldb,ld->b", T, y)

    h = torch.randn(B, device=device, dtype=torch.float32)
    h -= h.mean()
    h /= h.norm().clamp_min(1e-12)
    prev = None
    for _ in range(cfg.power_iters):
        y = apply_C(h)
        y -= y.mean()
        h_new = y / y.norm().clamp_min(1e-12)
        Ch = apply_C(h_new)
        lam = float(torch.dot(h_new, Ch).item())
        if prev is not None and abs(lam-prev) <= cfg.power_tol*max(1.0,abs(prev)):
            h = h_new
            break
        h, prev = h_new, lam
    lam = float(torch.dot(h, apply_C(h)).item())

    # Exact same stationarity diagnostic, now vectorized.
    ones = torch.ones(B, device=device)
    mean_r = torch.einsum("ldb,b->ld", T, ones)
    stationarity_rms = float(torch.sqrt(mean_r.pow(2).mean()).item())
    residual_scale = float(torch.sqrt((T.pow(2).sum(dim=2)*B).mean().clamp_min(1e-30)).item())

    return {
        "lambda_max_A": lam,
        "active_dim_for_spectrum": A,
        "stationarity_rms": stationarity_rms,
        "stationarity_relative": stationarity_rms / residual_scale,
    }



# -------------------------- free continuation ----------------------------

def soft_mass(s,lam):
    return (1+s-math.sqrt((1-s)**2+4*lam))/(2*s)

def ascending_grid(s_min,s_max,ds):
    vals=[]; s=float(s_min)
    while s <= float(s_max)+1e-10:
        vals.append(round(s,10)); s+=float(ds)
    return tuple(vals)

def _checkpoint_path(out, seed, k, s):
    ck = out / f"seed_{seed}"
    ck.mkdir(parents=True, exist_ok=True)
    return ck / f"step_{k:03d}_s{s:.6g}.pt"


def _atomic_save(path, writer):
    """A completed checkpoint is the commit marker for a continuation point."""
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    writer(temp)
    os.replace(temp, path)


def _rng_snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and len(state['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def _checkpoint_rows(state, checkpoint, out, previous_s, old=None):
    seed, k, s = int(state['seed']), int(state['continuation_index']), float(state['s'])
    active = np.asarray(state['active_mask'], dtype=bool)
    old = old or {}
    row = dict(seed=seed, continuation_index=k, s=s,
               previous_s=previous_s, warm_started=int(k > 0),
               n_active=int(active.sum()), n_collapsed=int(len(active)-active.sum()),
               final_loss=state.get('final_loss', old.get('final_loss', float('nan'))),
               final_reconstruction_term=state.get('final_reconstruction_term',
                   old.get('final_reconstruction_term', float('nan'))),
               checkpoint=str(checkpoint.relative_to(out)))
    lat = []
    order = np.argsort(-np.asarray(state['mean_sigma2']), kind='stable')
    for rank, j in enumerate(order, 1):
        lat.append(dict(seed=seed, continuation_index=k, s=s, latent_rank=rank,
                        original_latent=int(j), mean_sigma2=float(state['mean_sigma2'][j]),
                        mean_mu2=float(state['mean_mu2'][j]), mean_kl=float(state['mean_kl'][j]),
                        active=int(active[j])))
    return row, lat


def _write_training_tables(out, rows, latent_rows):
    if rows:
        _atomic_save(out/'trajectory.csv', lambda p: pd.DataFrame(rows).sort_values(
            ['seed', 'continuation_index']).to_csv(p, index=False))
        _atomic_save(out/'latent_stats.csv', lambda p: pd.DataFrame(latent_rows).sort_values(
            ['seed', 'continuation_index', 'latent_rank']).to_csv(p, index=False))


def train_trajectory(cfg, s_min, s_max, ds, init_epochs, step_epochs, resume=False, resume_config=None):
    """Resume at completed s-point boundaries; Adam is recreated at each point."""
    configure_cuda(cfg)
    out=Path(cfg.outdir); out.mkdir(parents=True,exist_ok=True)
    s_values=ascending_grid(s_min,s_max,ds)
    rows=[]; latent_rows=[]; restored={}
    old_rows={}
    if resume:
        try:
            old = pd.read_csv(out/'trajectory.csv')
            old_rows={(int(r['seed']),int(r['continuation_index'])):r for r in old.to_dict('records')}
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, KeyError, ValueError):
            print('[RESUME] Missing/incomplete trajectory.csv; rebuilding from checkpoints.', flush=True)
        # Validate all completed prefixes before updating any training outputs.
        validator=MLPVAE(cfg.data_dim,cfg.latent_dim,cfg.hidden_enc,cfg.hidden_dec)
        for seed in cfg.seeds:
            folder=out/f'seed_{seed}'
            files=set(folder.glob('step_*.pt'))
            expected=[folder/f'step_{k:03d}_s{s:.6g}.pt' for k,s in enumerate(s_values)]
            extra=files-set(expected)
            if extra:
                raise ValueError(f'Checkpoint grid mismatch: {sorted(str(p) for p in extra)}')
            gap=False; last=None
            for k,(s,path) in enumerate(zip(s_values,expected)):
                if path not in files:
                    gap=True; continue
                if gap:
                    raise ValueError(f'Non-contiguous completed checkpoints before {path}; refusing to overwrite later results.')
                try:
                    state=torch.load(path,map_location='cpu',weights_only=False)
                    if (int(state['seed'])!=int(seed) or int(state['continuation_index'])!=k
                            or not math.isclose(float(state['s']),s,rel_tol=0,abs_tol=1e-8)):
                        raise ValueError('checkpoint metadata does not match filename/grid')
                    for key in ('active_mask','mean_sigma2','mean_mu2','mean_kl'):
                        if np.asarray(state[key]).shape != (cfg.latent_dim,):
                            raise ValueError(f'invalid {key} shape')
                    if 'model' not in state: raise ValueError('missing model weights')
                    validator.load_state_dict(state['model'])
                    row,lat=_checkpoint_rows(state,path,out,s_values[k-1] if k else float('nan'),
                                            old_rows.get((int(seed),k)))
                except Exception as exc:
                    raise RuntimeError(f'Cannot resume from {path}; existing files preserved: {exc}') from exc
                rows.append(row); latent_rows.extend(lat);last=state
            restored[int(seed)]=last
        # Detect CSV entries whose checkpoint is missing, rather than silently discarding them.
        complete={(r['seed'],r['continuation_index']) for r in rows}
        missing=set(old_rows)-complete
        if missing:
            raise ValueError(f'trajectory.csv references missing checkpoints: {sorted(missing)}')
        _write_training_tables(out,rows,latent_rows)
        if resume_config is not None:
            _atomic_save(out/'config.json',lambda p: p.write_text(json.dumps(resume_config,indent=2)))
    x,data_eigs=load_dataset_for_estimation(cfg)
    np.savetxt(out/'data_covariance_eigenvalues.csv',data_eigs,delimiter=',',
               header='eigenvalue',comments='')

    # Only ~7.8 MiB for the default data. One host-to-device copy for the
    # whole training trajectory, shared by all scan points and training seeds.
    # Keep x on CPU for the unchanged frozen-checkpoint spectrum estimator.
    training_x = x.to(cfg.device) if (cfg.gpu_resident_data and
                                      torch.device(cfg.device).type == 'cuda') else x

    for seed in cfg.seeds:
        seed_all(int(seed))
        model=MLPVAE(cfg.data_dim,cfg.latent_dim,cfg.hidden_enc,cfg.hidden_dec).to(cfg.device)
        last=restored.get(int(seed)); start=0
        if last is not None:
            model.load_state_dict(last['model'])
            start=int(last['continuation_index'])+1
            if start >= len(s_values):
                print(f'[RESUME] seed={seed}: all {start} points complete; skipping training.',flush=True)
                continue
            if 'rng_state' in last:
                _restore_rng(last['rng_state'])
            else:
                # Legacy checkpoints have model/statistics but no RNG snapshot.
                seed_all(int(seed)+1_000_003*start)
                print('[RESUME] Legacy checkpoint: RNG state unavailable; weights restored, '
                      'future randomness is reproducibly reseeded.',flush=True)
            print(f'[RESUME] seed={seed}: {start} completed points; next index={start}, s={s_values[start]:.6g}',flush=True)
        for k in range(start,len(s_values)):
            s=s_values[k]
            print(f"\n[TRAIN] seed={seed}, s={s:.6g}, "
                  f"{'initialize' if k==0 else 'warm-start'}",flush=True)
            hist=train_free_step(model,training_x,cfg,float(s),init_epochs if k==0 else step_epochs)
            mean_var,mean_mu2,mean_kl=free_latent_statistics(model,training_x,cfg)
            active=(mean_kl > cfg.active_kl_threshold)
            ckpt=_checkpoint_path(out,seed,k,s)
            state=dict(model=model.state_dict(),s=float(s),seed=int(seed),continuation_index=k,
                       active_mask=active.numpy(),mean_sigma2=mean_var.numpy(),
                       mean_mu2=mean_mu2.numpy(),mean_kl=mean_kl.numpy(),
                       final_loss=hist[-1][0],final_reconstruction_term=hist[-1][1],
                       rng_state=_rng_snapshot())
            _atomic_save(ckpt,lambda p: torch.save(state,p))
            row,lat=_checkpoint_rows(state,ckpt,out,s_values[k-1] if k else float('nan'))
            rows.append(row);latent_rows.extend(lat)
            _write_training_tables(out,rows,latent_rows)
    return x,data_eigs


def load_dataset_for_estimation(cfg):
    if cfg.dataset == 'symmetric_gaussian_mixture':
        return make_symmetric_gaussian_mixture(cfg)
    raise ValueError(cfg.dataset)


def _weighted_extrapolation(cell_summary):
    """
    Weighted fit:
        Lambda(B,L) = Lambda_inf + a/B + c/L
    using inverse-variance weights from repeat-level SEs.
    Returns intercept, coefficients, chi2/dof, and condition number.
    """
    g=cell_summary.copy()
    X=np.column_stack([np.ones(len(g)),1.0/g.B.to_numpy(float),1.0/g.L.to_numpy(float)])
    y=g.lambda_mean.to_numpy(float)
    se=g.lambda_se.to_numpy(float)
    good=np.isfinite(se)&(se>0)
    fallback=np.nanmedian(se[good]) if good.any() else 1.0
    se=np.where(good,se,fallback)
    w=1.0/np.maximum(se,1e-12)**2
    XtW=X.T*w
    normal=XtW@X
    coef=np.linalg.lstsq(normal,XtW@y,rcond=None)[0]
    pred=X@coef
    resid=y-pred
    chi2=float(np.sum((resid/se)**2))
    dof=max(1,len(y)-X.shape[1])
    return dict(lambda_inf=float(coef[0]),a_over_B=float(coef[1]),
                c_over_L=float(coef[2]),chi2_dof=chi2/dof,
                fit_rms=float(np.sqrt(np.mean(resid**2))),
                condition_number=float(np.linalg.cond(normal)))


def _bootstrap_extrapolation(cells, n_boot, seed):
    """
    Hierarchical bootstrap over estimator repeats independently within each (B,L)
    cell, followed by the same weighted extrapolation.  This propagates the
    Monte-Carlo uncertainty into Lambda_inf without extrapolating each noisy
    repeat separately.
    """
    rng=np.random.default_rng(seed)
    keys=sorted(cells[['B','L']].drop_duplicates().itertuples(index=False,name=None))
    vals={(B,L):cells[(cells.B==B)&(cells.L==L)].lambda_max_A.to_numpy(float)
          for B,L in keys}
    boots=[]
    for _ in range(n_boot):
        rows=[]
        for B,L in keys:
            v=vals[(B,L)]
            vb=rng.choice(v,size=len(v),replace=True)
            rows.append(dict(B=B,L=L,lambda_mean=float(vb.mean()),
                             lambda_se=float(vb.std(ddof=1)/math.sqrt(len(vb)))
                             if len(vb)>1 else 1.0))
        fit=_weighted_extrapolation(pd.DataFrame(rows))
        if np.isfinite(fit['lambda_inf']):
            boots.append(fit['lambda_inf'])
    a=np.asarray(boots,float)
    if len(a)==0:
        return dict(se=float('nan'),q025=float('nan'),q975=float('nan'))
    return dict(se=float(a.std(ddof=1)) if len(a)>1 else float('nan'),
                q025=float(np.quantile(a,.025)),q975=float(np.quantile(a,.975)))


def estimate_saved_trajectory(cfg, n_lambda_repeats,
                              bias_extrapolation=True,
                              b_scales=(0.25,0.5,1.0),
                              l_scales=(0.25,0.5,1.0),
                              bootstrap_reps=500):
    """
    Phase 2, publication-oriented version.

    For each frozen VAE checkpoint:
      1. evaluate Lambda independently on a B x L resolution grid;
      2. repeat every grid cell n_lambda_repeats times;
      3. form mean +/- SE at each cell;
      4. fit the cell means, NOT individual repeats, to
             Lambda(B,L)=Lambda_inf+a/B+c/L
         by inverse-variance weighted least squares;
      5. bootstrap repeats within each grid cell to obtain uncertainty on
         Lambda_inf.

    The population fully-collapsed limit has Lambda=1 for this dataset.
    This is a reference, not the exact finite-sample value, and is never
    used to determine the correction.
    """
    configure_cuda(cfg)
    out=Path(cfg.outdir)
    traj_path=out/'trajectory.csv'
    if not traj_path.exists():
        raise FileNotFoundError(f"{traj_path} not found; run training first.")
    traj=pd.read_csv(traj_path)
    x,data_eigs=load_dataset_for_estimation(cfg)
    cell_rows=[]; fit_rows=[]
    model=MLPVAE(cfg.data_dim,cfg.latent_dim,cfg.hidden_enc,cfg.hidden_dec).to(cfg.device)

    Bvals=sorted(set(min(cfg.eval_points,max(64,int(round(cfg.eval_points*q)))) for q in b_scales))
    Lvals=sorted(set(min(cfg.z_mc,max(32,int(round(cfg.z_mc*q)))) for q in l_scales))
    if cfg.eval_points not in Bvals: Bvals.append(cfg.eval_points)
    if cfg.z_mc not in Lvals: Lvals.append(cfg.z_mc)
    Bvals=sorted(set(Bvals)); Lvals=sorted(set(Lvals))

    for _,row in traj.iterrows():
        train_seed=int(row.seed); k=int(row.continuation_index); s=float(row.s)
        n_active=int(row.n_active); n_collapsed=int(row.n_collapsed)
        state=torch.load(out/str(row.checkpoint),map_location=cfg.device,
                         weights_only=False)
        model.load_state_dict(state['model']); model.eval()
        active=torch.as_tensor(state['active_mask'],dtype=torch.bool)

        if n_collapsed==0:
            print(f"[LAMBDA] seed={train_seed}, s={s:.6g}: fully active -> N/A",flush=True)
            continue

        this_rows=[]
        grid=[(cfg.eval_points,cfg.z_mc)] if not bias_extrapolation else [
            (B,L) for B in Bvals for L in Lvals
        ]
        for B,L in grid:
            local_cfg=replace(cfg,eval_points=int(B),z_mc=int(L))
            for rep in range(n_lambda_repeats):
                est_seed=10_000_000*train_seed+100_000*k+1_000*B+10*L+rep
                print(f"[LAMBDA] train_seed={train_seed}, s={s:.6g}, "
                      f"B={B}, L={L}, repeat={rep+1}/{n_lambda_repeats}",flush=True)
                spec=estimate_lambda_max(model,x,active,local_cfg,seed=est_seed)
                lam=float(spec['lambda_max_A'])
                d=dict(train_seed=train_seed,continuation_index=k,s=s,
                       n_active=n_active,n_collapsed=n_collapsed,
                       B=int(B),L=int(L),collapsed_mc=cfg.collapsed_mc,
                       decoder_projection="standard_normal_mean",estimator_repeat=rep,
                       estimator_seed=est_seed,lambda_max_A=lam,
                       stationarity_rms=spec['stationarity_rms'],
                       stationarity_relative=spec['stationarity_relative'])
                cell_rows.append(d); this_rows.append(d)
            pd.DataFrame(cell_rows).to_csv(out/'lambda_grid_repeats.csv',index=False)

        cells=pd.DataFrame(this_rows)
        cs=cells.groupby(['B','L'],as_index=False).agg(
            lambda_mean=('lambda_max_A','mean'),
            lambda_sd=('lambda_max_A','std'),
            lambda_n=('lambda_max_A','count'),
            stationarity_relative_mean=('stationarity_relative','mean'))
        cs['lambda_se']=cs.lambda_sd/np.sqrt(cs.lambda_n)
        cs['collapsed_mc']=cfg.collapsed_mc
        cs['decoder_projection']='standard_normal_mean'
        cs['train_seed']=train_seed; cs['continuation_index']=k; cs['s']=s
        # append checkpoint-level cell summary incrementally
        cell_summary_path=out/'lambda_grid_summary.csv'
        if cell_summary_path.exists():
            old=pd.read_csv(cell_summary_path)
            old=old[~((old.train_seed==train_seed)&(old.continuation_index==k))]
            pd.concat([old,cs],ignore_index=True).to_csv(cell_summary_path,index=False)
        else:
            cs.to_csv(cell_summary_path,index=False)

        if bias_extrapolation:
            fit=_weighted_extrapolation(cs)
            boot=_bootstrap_extrapolation(cells,bootstrap_reps,
                                           seed=987654+1000*train_seed+k)
            lamcorr=fit['lambda_inf']
        else:
            q=cs[(cs.B==cfg.eval_points)&(cs.L==cfg.z_mc)].iloc[0]
            lamcorr=float(q.lambda_mean)
            fit=dict(lambda_inf=lamcorr,a_over_B=float('nan'),c_over_L=float('nan'),
                     chi2_dof=float('nan'),fit_rms=float('nan'),
                     condition_number=float('nan'))
            boot=dict(se=float(q.lambda_se),q025=lamcorr-1.96*q.lambda_se,
                      q975=lamcorr+1.96*q.lambda_se)

        rawq=cs[(cs.B==cfg.eval_points)&(cs.L==cfg.z_mc)].iloc[0]
        raw=float(rawq.lambda_mean)
        exact=float(data_eigs[0]) if (cfg.dataset=='symmetric_gaussian_mixture' and n_active==0) else float('nan')
        physical_ok=bool(np.isfinite(lamcorr) and lamcorr>=0)
        mass=(soft_mass(s,lamcorr) if physical_ok and ((1-s)**2+4*lamcorr)>=0
              else float('nan'))

        fit_rows.append(dict(
            train_seed=train_seed,continuation_index=k,s=s,
            n_active=n_active,n_collapsed=n_collapsed,
            collapsed_mc=cfg.collapsed_mc,decoder_projection="standard_normal_mean",
            lambda_raw_full_mean=raw,lambda_raw_full_se=float(rawq.lambda_se),
            lambda_inf=lamcorr,lambda_inf_bootstrap_se=boot['se'],
            lambda_inf_ci95_low=boot['q025'],lambda_inf_ci95_high=boot['q975'],
            estimated_bias_raw_minus_inf=raw-lamcorr,
            delta_A_corrected=s-lamcorr,
            delta_A_corrected_se=boot['se'],
            m_perp_corrected=mass,
            physical_lambda_nonnegative=int(physical_ok),
            fit_a_over_B=fit['a_over_B'],fit_c_over_L=fit['c_over_L'],
            fit_chi2_dof=fit['chi2_dof'],fit_rms=fit['fit_rms'],
            fit_condition_number=fit['condition_number'],
            lambda_exact_calibration=exact,
            corrected_calibration_error=(lamcorr-exact if np.isfinite(exact) else float('nan'))
        ))
        pd.DataFrame(fit_rows).to_csv(out/'spectral_summary.csv',index=False)

    return pd.DataFrame(cell_rows),pd.DataFrame(fit_rows)

def make_two_stage_plot(cfg):
    out=Path(cfg.outdir)
    traj=pd.read_csv(out/'trajectory.csv')
    lat=pd.read_csv(out/'latent_stats.csv')
    spec_path=out/'spectral_summary.csv'
    spec=pd.read_csv(spec_path) if spec_path.exists() else pd.DataFrame()

    for seed in sorted(traj.seed.unique()):
        tq=traj[traj.seed==seed].sort_values('s')
        lq=lat[lat.seed==seed]
        sq=spec[spec.train_seed==seed].sort_values('s') if not spec.empty else pd.DataFrame()
        fig,axes=plt.subplots(4,1,figsize=(7.4,10.5),sharex=True)
        axes[0].step(tq.s,tq.n_active,where='post')
        axes[0].set_ylabel(r'$N_{\rm active}$')
        axes[0].set_ylim(-.2,cfg.latent_dim+.2)

        for rank in range(1,cfg.latent_dim+1):
            q=lq[lq.latent_rank==rank].sort_values('s')
            axes[1].plot(q.s,q.mean_sigma2,marker='.',label=fr'$\sigma_{rank}^2$')
        axes[1].axhline(1,ls='--',lw=1)
        axes[1].set_ylabel(r'ordered $\langle\sigma_j^2\rangle$')
        axes[1].legend(ncol=cfg.latent_dim,fontsize=8)

        if not sq.empty:
            axes[2].errorbar(sq.s,sq.lambda_inf,yerr=sq.lambda_inf_bootstrap_se,
                             marker='o',ms=3,capsize=2,
                             label=r'$\bar\Lambda_{\infty}^{(A)}\pm{\rm SE}$')
            axes[2].plot(sq.s,sq.s,ls='--',lw=1,label=r'$s$')
            axes[2].legend()
            axes[3].errorbar(sq.s,sq.delta_A_corrected,yerr=sq.delta_A_corrected_se,
                             marker='o',ms=3,capsize=2)
        axes[2].set_ylabel('spectral scale')
        axes[3].axhline(0,ls='--',lw=1)
        axes[3].set_ylabel(r'$s-\bar\Lambda_{\max}^{(A)}$')
        axes[3].set_xlabel(r"$s=\sigma'^2$")
        fig.tight_layout()
        fig.savefig(out/f'two_stage_summary_seed{seed}.png',dpi=220)
        fig.savefig(out/f'two_stage_summary_seed{seed}.pdf')
        plt.close(fig)


# ------------------------- JSON entry point -------------------------------

RUN_DEFAULTS = dict(
    s_min=0.65, s_max=1.05, ds=0.001, init_epochs=120, step_epochs=30,
    n_lambda_repeats=16, bias_extrapolation=True,
    b_scales=[0.25, 0.5, 1.0], l_scales=[0.25, 0.5, 1.0],
    bootstrap_reps=500, resume=False, train_only=False, estimate_only=False,
)
MODE = 'two_stage_free_continuation_then_frozen_lambda'


def _reject_nonfinite(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


def load_json_config(path):
    raw = json.loads(Path(path).read_text(), parse_constant=_reject_nonfinite)
    if not isinstance(raw, dict):
        raise ValueError('Configuration must be a JSON object.')
    defaults = asdict(Config())
    defaults['outdir'] = 'results_mixture_a05_zc128'
    defaults.update(RUN_DEFAULTS)
    unknown = set(raw) - set(defaults) - {'mode'}
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    if 'mode' in raw and raw['mode'] != MODE:
        raise ValueError('Unsupported saved experiment mode.')
    for key, value in raw.items():
        if key == 'mode':
            continue
        example = defaults[key]
        if isinstance(example, bool):
            valid = type(value) is bool
        elif isinstance(example, int):
            valid = type(value) is int
        elif isinstance(example, float):
            valid = type(value) in (int, float) and math.isfinite(value)
        elif isinstance(example, (list, tuple)):
            valid = isinstance(value, list) and bool(value)
            if valid:
                element_type = type(example[0])
                valid = all(type(v) is int for v in value) if element_type is int else all(
                    type(v) in (int, float) and math.isfinite(v) for v in value)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f'Invalid type/value for {key}: {value!r}')
        defaults[key] = value
    return defaults


def validate_config(options):
    positive = ('n_samples', 'data_dim', 'latent_dim', 'batch_size', 'epochs',
                'eval_points', 'z_mc', 'collapsed_mc', 'collapsed_decode_batch',
                'power_iters', 'importance_chunk', 'init_epochs', 'step_epochs',
                'n_lambda_repeats', 'bootstrap_reps', 'lr', 'ds', 's_min', 'power_tol')
    for key in positive:
        if options[key] <= 0:
            raise ValueError(f'{key} must be positive.')
    if options['dataset'] != 'symmetric_gaussian_mixture' or options['data_dim'] != 8:
        raise ValueError('This experiment requires dataset=symmetric_gaussian_mixture and data_dim=8.')
    if not 0 <= options['mixture_a'] < 1:
        raise ValueError('mixture_a must satisfy 0 <= mixture_a < 1.')
    for key in ('spectrum_decay', 'active_kl_threshold', 'active_var_tol', 'num_workers'):
        if options[key] < 0:
            raise ValueError(f'{key} must be nonnegative.')
    for key in ('hidden_enc', 'hidden_dec'):
        if len(options[key]) != 2 or min(options[key]) < 1:
            raise ValueError(f'{key} must contain two positive layer widths.')
    if len(set(options['seeds'])) != len(options['seeds']) or min(options['seeds']) < 0:
        raise ValueError('seeds must contain distinct nonnegative integers.')
    if options['data_seed'] < 0:
        raise ValueError('data_seed must be nonnegative.')
    if options['s_max'] < options['s_min']:
        raise ValueError('s_max must be >= s_min.')
    if options['train_only'] and options['estimate_only']:
        raise ValueError('train_only and estimate_only are mutually exclusive.')
    if options['resume'] and options['estimate_only']:
        raise ValueError('resume continues training; use estimate_only separately.')
    if options['eval_points'] > options['n_samples']:
        raise ValueError('eval_points must not exceed n_samples (including extrapolation resolutions).')
    for key in ('b_scales', 'l_scales'):
        if any(not 0 < v <= 1 for v in options[key]):
            raise ValueError(f'{key} entries must lie in (0, 1].')
    if options['bias_extrapolation']:
        if options['n_lambda_repeats'] < 2 or options['bootstrap_reps'] < 2:
            raise ValueError('Bias extrapolation requires at least two repeats and bootstrap replicates.')
        for key, size, floor in [('b_scales', 'eval_points', 64), ('l_scales', 'z_mc', 32)]:
            resolutions = {options[size]} | {
                min(options[size], max(floor, round(options[size] * q))) for q in options[key]}
            if len(resolutions) < 2:
                raise ValueError(f'{key} must generate at least two distinct resolutions.')
    if not options['outdir'].strip():
        raise ValueError('outdir must not be empty.')
    device = torch.device(options['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; set device to cpu for a local smoke test.')


def run_experiment(options):
    validate_config(options)
    cfg_values = {f.name: options[f.name] for f in fields(Config)}
    for key in ('hidden_enc', 'hidden_dec', 'seeds', 's_values'):
        cfg_values[key] = tuple(cfg_values[key])
    cfg = Config(**cfg_values)
    out = Path(cfg.outdir)
    saved_path = out / 'config.json'
    dump = dict(options, mode=MODE)
    resume, estimate_only = options['resume'], options['estimate_only']
    if resume or estimate_only:
        if not saved_path.exists():
            raise ValueError(f'Resuming/measurement requires existing {saved_path}.')
        saved = json.loads(saved_path.read_text())
        # Measurement resolution/device may change; the learned trajectory and
        # the regenerated dataset must still describe the saved experiment.
        keys = ('dataset', 'data_seed', 'mixture_a', 'rotate_data',
                'n_samples', 'data_dim', 'latent_dim', 'spectrum_decay',
                'hidden_enc', 'hidden_dec', 'batch_size', 'lr', 'seeds',
                'active_kl_threshold', 'active_var_tol', 'use_amp', 'use_tf32',
                's_min', 'ds', 'init_epochs', 'step_epochs')
        for key in keys:
            if key not in saved or json.dumps(saved[key]) != json.dumps(dump[key]):
                raise ValueError(f'Saved configuration mismatch for {key}; existing outputs preserved.')
        if options['s_max'] < saved['s_max'] - 1e-10:
            raise ValueError('Cannot shorten the saved scan; s_max may be extended on resume.')
    elif out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty. Use resume/estimate_only or a new outdir.')
    out.mkdir(parents=True, exist_ok=True)
    if not resume and not estimate_only:
        _atomic_save(saved_path, lambda p: p.write_text(json.dumps(dump, indent=2) + '\n'))
    if not estimate_only:
        train_trajectory(cfg, options['s_min'], options['s_max'], options['ds'],
                         options['init_epochs'], options['step_epochs'],
                         resume=resume, resume_config=dump if resume else None)
        _atomic_save(saved_path, lambda p: p.write_text(json.dumps(dump, indent=2) + '\n'))
    if not options['train_only']:
        _atomic_save(out / 'measurement_config.json',
                     lambda p: p.write_text(json.dumps(dump, indent=2) + '\n'))
        estimate_saved_trajectory(cfg, options['n_lambda_repeats'],
                                  options['bias_extrapolation'],
                                  tuple(options['b_scales']), tuple(options['l_scales']),
                                  options['bootstrap_reps'])
        make_two_stage_plot(cfg)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path,
                   default=Path(__file__).with_name('experiment_mixture.json'),
                   help='JSON configuration (default: experiment_mixture.json beside script).')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--train-only', action='store_true', default=None)
    modes.add_argument('--estimate-only', action='store_true', default=None)
    p.add_argument('--resume', action='store_true', default=None)
    p.add_argument('--outdir', help='Optional override; relative paths use the working directory.')
    p.add_argument('--device', help='Optional override, e.g. cpu or cuda:0.')
    a = p.parse_args(argv)
    try:
        options = load_json_config(a.config)
        for key in ('resume', 'outdir', 'device'):
            value = getattr(a, key)
            if value is not None:
                options[key] = value
        if a.train_only:
            options.update(train_only=True, estimate_only=False)
        if a.estimate_only:
            options.update(estimate_only=True, train_only=False, resume=False)
        validate_config(options)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    return options


def main():
    run_experiment(parse_args())


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Function-space mean-channel perturbations of projected VAE checkpoints.

No optimization, parameter Hessian, forced zeros, or eigenvalue extrapolation.
See coupled_curvature_README.md for the background and sampling conventions.
"""
import argparse
import csv
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from vae_two_stage_continuation_lambda_biascorrected_mixture import (
    Config, MLPVAE, load_dataset_for_estimation,
)


def generator(seed):
    return torch.Generator(device='cpu').manual_seed(seed)


def normal(shape, seed, device):
    return torch.randn(shape, generator=generator(seed), dtype=torch.float64).to(device)


def log_q(z, mu, lv):
    return -0.5 * (((z[:, None] - mu[None]) ** 2 / lv.exp()[None])
                   + lv[None] + math.log(2 * math.pi)).sum(-1)


@torch.no_grad()
def project_decoder(model, z, mask, eps_c, max_batch):
    """One fixed, common-random-number approximation f_A(z_A), in float64."""
    values = []
    for start in range(0, len(z), max_batch):
        za = z[start:start + max_batch]
        total = torch.zeros(len(za), model.decoder[-1].out_features,
                            device=z.device, dtype=z.dtype)
        for k in range(0, len(eps_c), max(1, max_batch // len(za))):
            ec = eps_c[k:k + max(1, max_batch // len(za))]
            full = torch.zeros(len(za), len(ec), len(mask), device=z.device, dtype=z.dtype)
            full[:, :, mask] = za[:, None]
            full[:, :, ~mask] = ec[None]
            total += model.decode(full.reshape(-1, len(mask))).reshape(len(za), len(ec), -1).sum(1)
        values.append(total / len(eps_c))
    return torch.cat(values)


def sample_joint(mu, lv, n, seed):
    owners = torch.randint(len(mu), (n,), generator=generator(seed)).to(mu.device)
    z = mu[owners] + (lv[owners] / 2).exp() * normal((n, mu.shape[1]), seed + 1, mu.device)
    return owners, z


def top_mode(T, iterations, tolerance, seed):
    """Top eigenvector of P C P; empirical <psi^2> = 1 and <psi> = 0."""
    L, D, B = T.shape
    flat = T.reshape(L * D, B)

    def apply(h):
        y = (B / L) * (flat.T @ (flat @ h))
        return y - y.mean()

    best = None
    for restart in range(2):
        h = normal((B,), seed + restart, T.device)
        h -= h.mean()
        h /= h.norm()
        for iteration in range(iterations):
            y = apply(h)
            if y.norm() <= 1e-20:
                raise ValueError('Residual operator has no resolved positive mode.')
            h = y / y.norm()
            ch = apply(h)
            lam = h @ ch
            residual = (ch - lam * h).norm() / lam.abs().clamp_min(1e-20)
            if residual <= tolerance:
                break
        candidate = (float(lam), h * math.sqrt(B), float(residual), iteration + 1)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def energy_difference(residual, chi, psi, u, v, t, s):
    """Direct reconstruction-loss difference, integrating z_c moments exactly.

    z_c = xi + t*v*psi. Two nodes xi=+/-1 integrate this quadratic
    integrand exactly under N(0,1); this is NOT a Gaussian approximation.
    KL change for this coordinate is exactly (t*v*psi)^2/2.
    """
    base = residual.square().sum(-1)
    plus = residual - t * u * (1 + t * v * psi)[:, None] * chi
    minus = residual - t * u * (-1 + t * v * psi)[:, None] * chi
    delta_rec = ((plus.square().sum(-1) + minus.square().sum(-1)) / 2 - base).mean()
    return delta_rec / (2 * s) + (t * v * psi).square().mean() / 2


def matrix(s, a, b, p=1.0):
    return np.array([[a / s, -b / s], [-b / s, p]])


@torch.no_grad()
def measure(model, x, mask, cfg, options, seed, checkpoint_s):
    device = next(model.parameters()).device
    B, L = options['data_points'], options['mode_latent_samples']
    indices = torch.randperm(len(x), generator=generator(seed))[:B]
    xb = x[indices].to(device=device, dtype=torch.float64)
    mu_all, lv_all = model.encode(xb)
    mu, lv = mu_all[:, mask], lv_all[:, mask]
    ec = normal((options['collapsed_mc'], int((~mask).sum())), seed + 10, device)
    _, z = sample_joint(mu, lv, L, seed + 20)
    chunk = options['chunk_size']
    tensors = []
    for start in range(0, L, chunk):
        zz = z[start:start + chunk]
        f = project_decoder(model, zz, mask, ec, options['decode_batch_size'])
        w = log_q(zz, mu, lv).softmax(1)
        tensors.append((w[:, :, None] * (xb[None] - f[:, None])).permute(0, 2, 1))
    T = torch.cat(tensors)
    lam, psi, eig_res, iterations = top_mode(T, options['power_iters'], options['power_tol'], seed + 30)
    conditional_mean_rms = float((T.sum(-1).square().sum(-1).mean()).sqrt())
    del T, tensors
    # Same empirical data measure, independent latent quadrature. This avoids
    # inventing an out-of-sample extension of the eigenvector psi(x_b).
    owners, zv = sample_joint(mu, lv, options['validation_samples'], seed + 40)
    residuals, partners = [], []
    for start in range(0, len(zv), chunk):
        zz = zv[start:start + chunk]
        f = project_decoder(model, zz, mask, ec, options['decode_batch_size'])
        w = log_q(zz, mu, lv).softmax(1)
        r_all = xb[None] - f[:, None]
        chi = torch.einsum('lb,lbd,b->ld', w, r_all, psi) / math.sqrt(lam)
        residuals.append(xb[owners[start:start + chunk]] - f)
        partners.append(chi)
    r, chi = torch.cat(residuals), torch.cat(partners)
    pv = psi[owners]
    a = float(chi.square().sum(-1).mean())
    b = float((pv * (r * chi).sum(-1)).mean())
    p = float(pv.square().mean())
    quartic_moment = float((pv.square() * chi.square().sum(-1)).mean())
    # Independent joint samples estimate all entries, including encoder metric p.
    edge = b * b / (a * p)
    probes = [('checkpoint', checkpoint_s)] + [(f'lambda_factor_{factor:g}', lam * factor)
        for factor in options['probe_s_factors']]
    rows = []
    for probe_label, s in probes:
        theoretical = matrix(s, 1, math.sqrt(lam))
        eigenvalues, vectors = np.linalg.eigh(theoretical)
        u, v = vectors[:, 0]
        measured = matrix(s, a, b, p)
        directional = float(np.array([u, v]) @ measured @ np.array([u, v]))
        for t in options['amplitudes']:
            dp = float(energy_difference(r, chi, pv, u, v, t, s))
            dm = float(energy_difference(r, chi, pv, u, v, -t, s))
            fd = (dp + dm) / t ** 2
            quartic_bias = t ** 2 * u ** 2 * v ** 2 * quartic_moment / s
            rows.append(dict(probe_label=probe_label, probe_s=s, is_checkpoint_s=probe_label == 'checkpoint',
                amplitude=t, lambda_mode=lam, predicted_mass=eigenvalues[0],
                direction_u=u, direction_v=v, delta_F_plus=dp, delta_F_minus=dm,
                curvature_fd=fd, curvature_fd_quartic_removed=fd-quartic_bias,
                curvature_joint_moments=directional, measured_block_min=np.linalg.eigvalsh(measured)[0],
                decoder_norm2=a, encoder_norm2=p, cross_correlation=b,
                frozen_background_block_edge=edge, conditional_residual_rms=conditional_mean_rms,
                eigen_residual=eig_res, power_iterations=iterations,
                mode_converged=eig_res <= options['power_tol']))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(options, config_dir):
    required = {'results_dir', 'output_dir', 'training_seed', 'checkpoint_s', 'device',
        'data_points', 'mode_latent_samples', 'validation_samples', 'collapsed_mc',
        'chunk_size', 'decode_batch_size', 'power_iters', 'power_tol', 'repeats',
        'measurement_seed', 'amplitudes', 'probe_s_factors'}
    if set(options) != required:
        raise ValueError(f'Config keys: missing={required-set(options)}, unknown={set(options)-required}')
    for name in ('data_points', 'mode_latent_samples', 'validation_samples', 'collapsed_mc',
                 'chunk_size', 'decode_batch_size', 'power_iters', 'repeats'):
        if not isinstance(options[name], int) or options[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('amplitudes', 'probe_s_factors', 'checkpoint_s'):
        if not options[name] or any(not math.isfinite(v) or v <= 0 for v in options[name]):
            raise ValueError(f'{name} must be a nonempty list of positive finite values')
    if not math.isfinite(options['power_tol']) or options['power_tol'] <= 0:
        raise ValueError('power_tol must be positive and finite')
    root = (config_dir / options['results_dir']).resolve()
    out = (config_dir / options['output_dir']).resolve()
    if out == root or root in out.parents:
        raise ValueError('Use a separate output directory, outside the original results directory.')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is nonempty; choose another output_dir to preserve measurements.')
    source_config = json.loads((root / 'config.json').read_text())
    cfg = Config(**{k: v for k, v in source_config.items() if k in {f.name for f in fields(Config)}})
    if options['data_points'] > cfg.n_samples:
        raise ValueError('data_points exceeds the dataset size')
    device = options['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use device=cpu or auto.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with (root / 'trajectory.csv').open(newline='') as f:
        trajectory = [r for r in csv.DictReader(f) if int(r['seed']) == options['training_seed']]
    selected = []
    for s in options['checkpoint_s']:
        hits = [r for r in trajectory if math.isclose(float(r['s']), s, abs_tol=1e-9, rel_tol=0)]
        if len(hits) != 1:
            raise ValueError(f'Expected exactly one checkpoint at s={s}; found {len(hits)}')
        selected.append(hits[0])
    x, _ = load_dataset_for_estimation(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'measurement_config.json').write_text(json.dumps(dict(options=options,
        source_config=source_config, resolved_results_dir=str(root), resolved_device=device,
        torch_version=torch.__version__, convention='negative ELBO; projected function-space mean channel; float64'), indent=2))
    rows, skipped = [], []
    for row in selected:
        checkpoint = root / row['checkpoint']
        # The project's own trusted training checkpoints include numpy/RNG metadata.
        with checkpoint.open('rb') as f:
            state = torch.load(f, map_location='cpu', weights_only=False)
        mask = torch.as_tensor(state['active_mask'], device=device, dtype=torch.bool)
        if mask.numel() != cfg.latent_dim:
            raise ValueError('Invalid checkpoint active mask')
        if mask.all():
            skipped.append({'checkpoint': row['checkpoint'], 'reason': 'no collapsed coordinate'})
            print(f"SKIP s={row['s']}: all coordinates active", flush=True)
            continue
        model = MLPVAE(cfg.data_dim, cfg.latent_dim, cfg.hidden_enc, cfg.hidden_dec)
        model.load_state_dict(state['model'])
        model = model.to(device=device, dtype=torch.float64).eval()
        model.requires_grad_(False)
        for repeat in range(options['repeats']):
            print(f"MEASURE s={row['s']} |A|={int(mask.sum())} repeat={repeat+1}/{options['repeats']}", flush=True)
            seed = options['measurement_seed'] + 100000 * int(row['continuation_index']) + 1000 * repeat
            local = measure(model, x, mask, cfg, options, seed, float(row['s']))
            metadata = dict(checkpoint=row['checkpoint'], checkpoint_s=float(row['s']),
                training_seed=options['training_seed'], repeat=repeat, measurement_seed=seed,
                n_active=int(mask.sum()), collapsed_coordinate=int(torch.where(~mask)[0][0]))
            rows.extend({**metadata, **r} for r in local)
            write_csv(out / 'coupled_curvature.csv', rows)
            if not local[0]['mode_converged']:
                print('WARNING: mode iteration has not met tolerance; increase power_iters.', flush=True)
        del model, state
    (out / 'skipped.json').write_text(json.dumps(skipped, indent=2))
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        columns = ['lambda_mode', 'predicted_mass', 'curvature_fd', 'curvature_fd_quartic_removed',
                   'curvature_joint_moments', 'measured_block_min', 'frozen_background_block_edge',
                   'decoder_norm2', 'encoder_norm2', 'cross_correlation', 'conditional_residual_rms', 'eigen_residual']
        # Repeat-specific probe variances share a fixed dimensionless factor.
        summary = df.groupby(['checkpoint', 'amplitude', 'probe_label'])[columns + ['probe_s']].agg(['mean', 'sem', 'count'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary.to_csv(out / 'summary.csv')
    print(f'Outputs: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = Path(args.config).resolve()
    run(json.loads(path.read_text()), path.parent)


if __name__ == '__main__':
    main()

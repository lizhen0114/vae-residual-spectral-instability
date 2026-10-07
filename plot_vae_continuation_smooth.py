#!/usr/bin/env python3
"""Standalone scatter + smooth-trend variant of plot_vae_continuation.py.

Panel (b) joins original points with straight segments. Panels (c), (d) display
small original points and Gaussian local-linear smooth curves, without error
bars or confidence intervals. Smoothed curves are visual guides,
not new measurements or interpolants constrained to pass through every point.
Spectral curves are segmented at active-set changes, missing or flagged points.
The original plotting script and its outputs are preserved.

Requires numpy, pandas, matplotlib; no scipy, torch or LaTeX installation.
Example:
  python plot_vae_continuation_smooth.py --results results_two_stage_l40s --seed 0
Use --smooth-width 0.25 for stronger smoothing (default: 0.15).
Use --show-data-eigenvalues to mark the largest max(|A|) data covariance
eigenvalues on panel (a), with a labeled top axis and dash-dot reference lines.
Panel (a) removes local unit-count reversals over at most 0.01 in s; use
--collapse-flicker-width to change this span (0 preserves raw counts).
The resulting collapse boundary is placed at the start of the flicker region.
Use --zero-at-collapse to constrain panel (d)'s smooth guides to zero at the
displayed collapse boundaries. This is a visual constraint, not a measurement;
raw scatter points remain unchanged, and missing/flagged gaps remain broken.
"""

import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAMBDA = r'\widehat{\Lambda}_*^{(A)}'
VARIANCE = r'\langle\sigma_j^2\rangle_{x}'

# Restrained, high-contrast print palette; not an official journal style.
COLORS = {
    'active': '#332288',
    'spectrum': '#0077BB',
    'margin': '#BB5566',
    'flagged': '#994F00',
    'reference': '#444444',
    'boundary': '#777777',
    'pca': '#AA3377',
}
LATENT_COLORS = ['#4477AA', '#EE6677', '#228833', '#AA3377',
                 '#66CCEE', '#CCBB44', '#BBBBBB', '#332288']
PAPER_STYLE = {
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14,
    'axes.labelsize': 19,
    'axes.labelpad': 9,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.linewidth': 1.3,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.prop_cycle': plt.cycler(color=LATENT_COLORS),
}


def select_seed(frame, seed):
    if frame.empty:
        return frame.copy()
    key = 'train_seed' if 'train_seed' in frame else 'seed'
    return frame.loc[frame[key] == seed].sort_values('s').copy()


def fit_flags(spec, chi2_max, relative_rms_max):
    lam = spec['lambda_inf'].to_numpy(float)
    flagged = ~np.isfinite(lam) | (lam < 0)
    if 'physical_lambda_nonnegative' in spec:
        flagged |= spec['physical_lambda_nonnegative'].to_numpy(float) == 0
    # NaN fit diagnostics are normal when extrapolation was disabled.
    if chi2_max is not None and 'fit_chi2_dof' in spec:
        chi = spec['fit_chi2_dof'].to_numpy(float)
        flagged |= np.isfinite(chi) & (chi > chi2_max)
    if relative_rms_max is not None and 'fit_rms' in spec:
        rms = spec['fit_rms'].to_numpy(float)
        scale = np.maximum(np.abs(lam), np.abs(spec['s'].to_numpy(float)))
        flagged |= np.isfinite(rms) & (rms > relative_rms_max * np.maximum(scale, 1e-12))
    return flagged


def smooth_curve(x, y, width):
    """Gaussian local-linear regression on a dense grid, without extrapolation.

    The bandwidth is width times the segment's x range. Local-linear smoothing
    reproduces affine functions, so smoothing s-lambda is consistent with
    s minus the smoothed lambda when nodes and segments are identical.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return x, y
    if np.any(np.diff(x) <= 0):
        raise ValueError('Smooth curves require strictly increasing s values.')
    grid = np.linspace(x[0], x[-1], max(100, 12 * len(x)))
    if len(x) == 2:
        return grid, np.interp(grid, x, y)
    bandwidth = max(width * (x[-1] - x[0]), np.median(np.diff(x)) * .5)
    u = (x[None, :] - grid[:, None]) / bandwidth
    weights = np.exp(-.5 * u**2)
    s0 = weights.sum(1)
    s1 = (weights * u).sum(1)
    s2 = (weights * u**2).sum(1)
    t0 = weights @ y
    t1 = (weights * u) @ y
    determinant = s0 * s2 - s1**2
    result = np.divide(s2 * t0 - s1 * t1, determinant,
                       out=t0 / s0, where=determinant > 1e-14)
    return grid, result


def smooth_with_zero_endpoints(x, y, width, left=False, right=False):
    """Local endpoint correction of the visual guide, with exact zero anchors."""
    gx, gy = smooth_curve(x, y, width)
    ends = ([0] if left else []) + ([-1] if right else [])
    if ends:
        bandwidth = max(width * (gx[-1] - gx[0]), np.median(np.diff(x)) * .5)
        basis = np.exp(-.5 * ((gx[:, None] - gx[ends]) / bandwidth)**2)
        gy -= basis @ np.linalg.solve(basis[ends], gy[ends])
        gy[ends] = 0.0
    return gx, gy


def draw_smooth_segments(ax, x, y, valid, branches, color, width, zero_at=()):
    # Split at every missing point and branch boundary; do not bridge gaps.
    start = None
    for i in range(len(x) + 1):
        split = (i == len(x) or not valid[i] or
                 (start is not None and branches[i] != branches[i-1]))
        if split and start is not None:
            if i - start >= 2:
                sx, sy = x[start:i], y[start:i]
                left = x[start] in zero_at
                right = x[i-1] in zero_at
                # Reach the next collapse boundary only when its measurement is
                # valid: never extend over missing or flagged observations.
                if i < len(x) and valid[i] and x[i] in zero_at:
                    sx, sy = np.r_[sx, x[i]], np.r_[sy, 0.0]
                    right = True
                gx, gy = smooth_with_zero_endpoints(sx, sy, width, left, right)
                ax.plot(gx, gy, color=color, lw=2.1, zorder=2)
            start = None
        if i < len(x) and valid[i] and start is None:
            start = i


def draw_estimate(ax, x, y, flagged, branches, color, label, width, zero_at=()):
    finite = np.isfinite(y)
    draw_smooth_segments(ax, x, y, finite & ~flagged, branches, color, width, zero_at)
    for flag in (False, True):
        keep = finite & (flagged == flag)
        if not keep.any():
            continue
        c = COLORS['flagged'] if flag else color
        ax.plot(x[keep], y[keep], linestyle='none', marker='o', ms=1.8, markeredgewidth=.65,
                markerfacecolor='none' if flag else c, markeredgecolor=c,
                alpha=.95, label='Flagged estimate' if flag else label, zorder=4)


def remove_count_flicker(x, counts, window):
    """Merge local unit-amplitude reversals at the first transition.

    A cluster spans at most `window` in s and visits only two adjacent counts.
    Apply its final count from the first transition onward; a return to the
    original count erases the entire excursion. Sustained changes are retained.
    """
    counts = np.asarray(counts).copy()
    changes = np.flatnonzero(np.diff(counts) != 0) + 1
    k = 0
    while k < len(changes):
        first = changes[k]
        end = k
        low = high = counts[first-1]
        while end < len(changes):
            pos = changes[end]
            lo, hi = min(low, counts[pos]), max(high, counts[pos])
            if x[pos] - x[first] > window + 1e-12 or hi - lo > 1:
                break
            low, high = lo, hi
            end += 1
        if window > 0 and end - k >= 2:
            counts[first:changes[end-1]] = counts[changes[end-1]]
        k = max(k+1, end)
    return counts


def active_branches(traj, lat):
    """Split spectral curves on membership changes, mark count changes at new s.

    The count is n[i] on [s[i], s[i+1]); a transition occurs at s[i+1].
    Membership-only swaps do not introduce extra count/collapse boundary lines.
    """
    states = {}
    if {'original_latent', 'active'}.issubset(lat.columns):
        for s, rows in lat.groupby('s'):
            states[s] = tuple(sorted(rows.loc[rows.active > 0, 'original_latent'].astype(int)))
    keys = [states.get(s, int(n)) for s, n in zip(traj.s, traj.n_active)]
    changes = np.array([False] + [a != b for a, b in zip(keys[:-1], keys[1:])])
    counts = traj.n_active.to_numpy()
    count_changes = np.r_[False, counts[1:] != counts[:-1]]
    return np.cumsum(changes), traj.s.to_numpy(float)[count_changes]


def experiment_title(config, custom):
    if custom:
        return custom
    if config.get('dataset') == 'synthetic':
        return 'Synthetic Gaussian VAE'
    if config.get('dataset') == 'fashion_mnist':
        return 'Fashion-MNIST VAE'
    return 'VAE continuation'


def mark_data_eigenvalues(ax, root, count):
    path = root / 'data_covariance_eigenvalues.csv'
    if count == 0:
        return
    if not path.exists():
        warnings.warn(f'{root}: covariance eigenvalues unavailable; skipping top-axis reference.')
        return
    frame = pd.read_csv(path)
    eig = frame['eigenvalue'] if 'eigenvalue' in frame else frame.iloc[:, 0]
    eig = eig.to_numpy(float)
    if not np.isfinite(eig).all() or (eig < 0).any():
        raise ValueError(f'{path}: covariance eigenvalues must be finite and nonnegative.')
    if len(eig) < count:
        warnings.warn(f'{root}: only {len(eig)} covariance eigenvalues available; requested {count}.')
    eig = np.sort(eig)[::-1][:count]
    # Keep the measured scan range; reference lines must not change shared limits.
    limits = ax.get_xlim()
    visible = (eig >= limits[0]) & (eig <= limits[1])
    if not visible.all():
        warnings.warn(f'{root}: eigenvalues outside the plotted scan range are not shown: {eig[~visible]}.')
    ticks, labels = [], []
    for value in np.unique(eig[visible]):
        ranks = np.flatnonzero(eig == value) + 1
        symbol = ','.join(rf'\lambda_{{{rank}}}' for rank in ranks)
        ticks.append(value)
        labels.append('$' + symbol + '$')
        ax.axvline(value, color=COLORS['pca'], ls='-.', lw=1.4, alpha=.9, zorder=1)
    ax.set_xlim(limits)
    if ticks:
        top = ax.secondary_xaxis('top')
        top.set_xticks(ticks, labels)
        top.tick_params(direction='in', colors='black', labelsize=12)
        ax.tick_params(top=False)


def plot_seed(root, seed, args):
    traj = select_seed(pd.read_csv(root / 'trajectory.csv'), seed)
    lat = select_seed(pd.read_csv(root / 'latent_stats.csv'), seed)
    if traj.empty or lat.empty:
        raise ValueError(f'{root}: missing trajectory or latent rows for seed {seed}')
    if traj.s.duplicated().any():
        raise ValueError(f'{root}: duplicate trajectory s values for seed {seed}')
    spec_path = root / 'spectral_summary.csv'
    spec = select_seed(pd.read_csv(spec_path), seed) if spec_path.exists() else pd.DataFrame()
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    branches, transitions = active_branches(traj, lat)
    x = traj.s.to_numpy(float)
    displayed_counts = remove_count_flicker(x, traj.n_active.to_numpy(), args.collapse_flicker_width)
    count_changes = np.r_[False, np.diff(displayed_counts) != 0]
    transitions = x[count_changes]
    collapses = x[np.r_[False, np.diff(displayed_counts) < 0]]
    # Retain raw membership boundaries for spectral estimates, and also split
    # at displayed critical points so endpoint constraints are branch-local.
    branches = np.cumsum(np.r_[False, np.diff(branches) != 0] | count_changes)
    changed = np.count_nonzero(displayed_counts != traj.n_active.to_numpy())
    if changed:
        print(f'{root}, seed {seed}: removed unit-count flicker at {changed} plotted checkpoints.')
    latent_dim = int((traj.n_active + traj.n_collapsed).max())

    fig, axes = plt.subplots(4, 1, figsize=(8.2, 10.6), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.8, 1.8, 1.8]})
    a0, a1, a2, a3 = axes
    # Right-continuous: keep n[i] throughout [x[i], x[i+1]).
    a0.step(x, displayed_counts, where='post', lw=2.3, color=COLORS['active'])
    a0.set_ylabel(r'$|A|$')
    a0.set_ylim(-.25, latent_dim + .4)
    a0.set_yticks(np.arange(latent_dim + 1))
    key = 'latent_rank' if args.latent_order == 'rank' else 'original_latent'
    for j, rows in lat.groupby(key, sort=True):
        rows = rows.sort_values('s')
        label = f'Rank {int(j)}' if key == 'latent_rank' else rf'$j={int(j) + 1}$'
        # a1.plot(rows.s, rows.mean_sigma2, 'o-', ms=1.7, markeredgewidth=.4,
        #         lw=1.1, alpha=.9, label=label)
        a1.plot(rows.s, rows.mean_sigma2, 'o-', ms=1.7, markeredgewidth=.4,
                        lw=1.65, alpha=1.0)
    a1.axhline(1, color=COLORS['reference'], ls='--', lw=1.3)
    a1.set_ylabel(r'$' + VARIANCE + '$')
#    a1.legend(ncol=min(4, latent_dim), fontsize=8, loc='best', framealpha=.9)
#    a1.set_title('Descending rank at each checkpoint' if key == 'latent_rank'
#                 else 'Fixed latent coordinates', loc='right', fontsize=8, color='.35', pad=6)

    #a2.plot(x, x, '--', color='.35', lw=1.2, label=r'$\sigma^{\prime 2}$')
    a2.plot(x, x, '--', color=COLORS['reference'], lw=1.5, label=None)
    a2.set_ylabel(r'$' + LAMBDA + '$')
    a3.axhline(0, color=COLORS['reference'], ls='--', lw=1.3)
    a3.set_ylabel(r'$\sigma^{\prime 2}-' + LAMBDA + '$')
    note = 'No spectral estimates available'
    if not spec.empty:
        if spec.s.duplicated().any():
            raise ValueError(f'{root}: duplicate spectral s values for seed {seed}')
        if not spec.s.isin(traj.s).all():
            raise ValueError(f'{root}: spectral checkpoints do not match the trajectory')
        # Align to the full trajectory so unmeasured points break connecting lines.
        aligned = spec.set_index('s').reindex(x)
        aligned['s'] = x
        measured = np.isin(x, spec.s.to_numpy(float))
        lam = aligned.lambda_inf.to_numpy(float)
        flags = fit_flags(aligned, args.chi2_max, args.relative_fit_rms_max)
        label = r'$' + LAMBDA + '$ estimates'
        #draw_estimate(a2, x, lam, flags, branches, '#c77720', label, args.smooth_width)
        draw_estimate(a2, x, lam, flags, branches, COLORS['spectrum'], None, args.smooth_width)
        draw_estimate(a3, x, x - lam, flags, branches, COLORS['margin'], None, args.smooth_width,
                      zero_at=collapses if args.zero_at_collapse else ())
        unavailable = int(np.sum(measured & ~np.isfinite(lam)))
        flagged = int(np.sum(measured & flags & np.isfinite(lam)))
        print(f'{root}, seed {seed}: {measured.sum()} measured checkpoints, '
              f'{flagged} flagged, {unavailable} nonfinite estimates')
        if unavailable:
            a2.text(.98, .04, f'{unavailable} nonfinite estimates (not drawable)',
                    ha='right', transform=a2.transAxes, fontsize=8)
        projection = set(spec.decoder_projection.dropna()) if 'decoder_projection' in spec else set()
        if projection == {'standard_normal_mean'}:
            ks = ', '.join(str(int(k)) for k in sorted(spec.collapsed_mc.dropna().unique()))
            note = r'Decoder averaged over $Z_C\sim\mathcal{N}(0,I)$' + f'; K = {ks}'
        else:
            note = 'Decoder projection: legacy / unspecified' if not projection else 'Mixed decoder projection metadata'
        if args.show_pca:
            eig_path = root / 'data_covariance_eigenvalues.csv'
            if eig_path.exists():
                eig = pd.read_csv(eig_path).iloc[:, 0].to_numpy(float)
                a2.axhline(eig[0], color=COLORS['pca'], ls='-.', lw=1.3,
                            label=r'$\lambda_{\max}(\Sigma_X)$ (reference)')
            else:
                warnings.warn(f'{root}: covariance eigenvalues unavailable; skipping reference.')
    else:
        for ax in (a2, a3):
            ax.text(.5, .5, 'No spectral estimates for this seed\n(run the estimation phase)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10, color='.4')
#    a2.legend(fontsize=8, loc='best', framealpha=.9)
    for letter, ax in zip('abcd', axes):
        for sc in transitions:
            ax.axvline(sc, ls=':', color=COLORS['boundary'], lw=1.15, alpha=.8)
        ax.grid(color='#AAAAAA', alpha=.20, linewidth=.55)
        ax.text(-.11, 1.02, f'({letter})', transform=ax.transAxes, fontsize=15, fontweight='bold')
        ax.tick_params(direction='in', top=True, right=True)
    if getattr(args, 'show_data_eigenvalues', False):
        mark_data_eigenvalues(a0, root, int(traj.n_active.max()))
    a3.set_xlabel(r'$\sigma^{\prime 2}$')
#    fig.suptitle(f'{experiment_title(config, args.title)} — seed {seed}', fontsize=13, y=.992)
#    fig.text(.5, .014, note + '\nDots: original estimates; (b): straight segments; (c,d): smoothed guides. No uncertainty intervals.',
#             ha='center', va='bottom', fontsize=8, color='.35')
    fig.tight_layout(rect=(0, .053, 1, .995), h_pad=1.2)
    prefix = root / (args.out_prefix or 'continuation_summary_smooth')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        output = prefix.with_name(f'{prefix.name}_seed{seed}.{ext}')
        fig.savefig(output, dpi=args.dpi, bbox_inches='tight')
        print(f'Saved: {output}')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, nargs='+', required=True, help='One or more result directories.')
    p.add_argument('--seed', type=int, nargs='+', help='Selected seeds; default: all seeds in each trajectory.')
    p.add_argument('--out-prefix', help='Output prefix within each result directory; seed suffix is added.')
    p.add_argument('--title', help='Override the dataset title inferred from config.json.')
    p.add_argument('--smooth-width', type=float, default=0.15,
                   help='Gaussian bandwidth / segment span; larger gives smoother curves.')
    p.add_argument('--collapse-flicker-width', type=float, default=0.01,
                   help='Maximum s span of unit-count reversals removed in (a); 0 disables (default: 0.01).')
    p.add_argument('--zero-at-collapse', action='store_true',
                   help='Constrain panel (d) smooth guides to zero at collapse boundaries; raw points remain unchanged.')
    p.add_argument('--latent-order', choices=['rank', 'index'], default='rank')
    p.add_argument('--chi2-max', type=float, default=10., help='Flag finite fit chi-square/dof above this value.')
    p.add_argument('--relative-fit-rms-max', type=float,
                   help='Optional scale-relative fit RMS threshold: RMS/max(|lambda|,|s|).')
    p.add_argument('--show-pca', action='store_true', help='Show leading data covariance eigenvalue as a reference.')
    p.add_argument('--show-data-eigenvalues', action='store_true',
                   help='Mark the largest max(|A|) data covariance eigenvalues on panel (a): top-axis labels and dash-dot lines.')
    p.add_argument('--dpi', type=int, default=220)
    return p.parse_args()


def main():
    args = parse_args()
    if not np.isfinite(args.smooth_width) or args.smooth_width <= 0:
        raise ValueError('--smooth-width must be finite and positive.')
    if not np.isfinite(args.collapse_flicker_width) or args.collapse_flicker_width < 0:
        raise ValueError('--collapse-flicker-width must be finite and nonnegative.')
    plt.rcParams.update(PAPER_STYLE)
    for root in args.results:
        seeds = args.seed if args.seed is not None else sorted(pd.read_csv(root / 'trajectory.csv').seed.unique())
        for seed in seeds:
            plot_seed(root, int(seed), args)


if __name__ == '__main__':
    main()

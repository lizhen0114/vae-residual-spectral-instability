#!/usr/bin/env python3
"""Curvature scans grouped by active dimension; repeat means +/- one SEM.

Uses the typography and palette of plot_vae_continuation_smooth.py.
No smoothing, forced zeros, or pooling of distinct checkpoint backgrounds.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_vae_continuation_smooth import PAPER_STYLE, COLORS, LATENT_COLORS


def stats(values):
    values = np.asarray(values, float)
    return float(values.mean()), (float(values.std(ddof=1) / np.sqrt(len(values)))
                                 if len(values) > 1 else float('nan'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results_coupled_curvature_a05'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amplitude', type=float, default=0.01)
    parser.add_argument('--corrected', action='store_true', help='Subtract the known quartic finite-amplitude term.')
    parser.add_argument('--exclude-unconverged', action='store_true', help='Exclude entire repeats failing mode tolerance.')
    parser.add_argument('--output', type=Path, help='Output prefix (without extension).')
    parser.add_argument('--layout', choices=['active', 'overview'], default='active',
                        help='active: compact panels by |A|; overview: original three-panel figure')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    d = pd.read_csv(args.results / 'coupled_curvature.csv')
    d = d[(d.training_seed == args.seed) & np.isclose(d.amplitude, args.amplitude, rtol=1e-9, atol=1e-12)].copy()
    if d.empty:
        parser.error('No measurements for this training seed and amplitude.')
    d['converged'] = d.mode_converged.astype(str).str.lower().map({'true': True, 'false': False})
    if d.converged.isna().any():
        parser.error('Unrecognized mode_converged values.')
    failed = d.loc[~d.converged, ['checkpoint', 'repeat']].drop_duplicates()
    if args.exclude_unconverged:
        d = d[d.converged].copy()
    if d.empty:
        parser.error('No measurements remain after filtering.')
    if d.duplicated(['checkpoint', 'repeat', 'probe_label']).any():
        parser.error('Duplicate measurements for checkpoint/repeat/probe.')
    ykey = 'curvature_fd_quartic_removed' if args.corrected else 'curvature_fd'
    required = ['checkpoint_s', 'probe_s', 'lambda_mode', 'predicted_mass', ykey,
                'frozen_background_block_edge']
    if not np.isfinite(d[required].to_numpy(float)).all() or (d.lambda_mode <= 0).any():
        parser.error('Measurements must be finite with positive lambda_mode.')
    rows = []
    for (checkpoint, label), g in d.groupby(['checkpoint', 'probe_label'], sort=False):
        row = dict(checkpoint=checkpoint, checkpoint_s=g.checkpoint_s.iloc[0],
                   n_active=int(g.n_active.iloc[0]), probe_label=label, repeats=len(g),
                   unconverged_repeats=int((~g.converged).sum()))
        quantities = {'s_ratio': g.probe_s / g.lambda_mode, 'theory': g.predicted_mass,
                      'measured': g[ykey], 'edge_shift_percent':
                      100 * (g.frozen_background_block_edge / g.lambda_mode - 1)}
        for name, values in quantities.items():
            row[name], row[name + '_sem'] = stats(values)
        rows.append(row)
    values = pd.DataFrame(rows)
    cp = values[values.probe_label == 'checkpoint'].sort_values('checkpoint_s')
    if cp.empty:
        parser.error('Missing checkpoint probes.')
    plt.rcParams.update(PAPER_STYLE)
    if args.layout == 'active':
        scans = values[values.probe_label != 'checkpoint']
        counts = sorted(scans.n_active.unique(), reverse=True)
        if not counts:
            parser.error('No fixed-background scan points.')
        ncols = min(2, len(counts))
        nrows = (len(counts) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(8.2, 3.05*nrows),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, count, letter in zip(axes.flat, counts, 'abcdefghijklmnopqrstuvwxyz'):
            part = scans[scans.n_active == count]
            for i, (checkpoint, g) in enumerate(part.groupby('checkpoint', sort=True)):
                g = g.sort_values('s_ratio')
                color = LATENT_COLORS[i % len(LATENT_COLORS)]
                # marker = ['o', 's', '^', 'D'][i % 4]
                marker = 'o'
                # flag = ' *' if g.unconverged_repeats.max() else ''
                flag = ''
                ax.plot(g.s_ratio, g.theory, '--', color= "#0000009D", lw=1.35)
                ax.errorbar(g.s_ratio, g.measured, yerr=g.measured_sem,
                            fmt=marker, color=color, ms=4, capsize=2,
                            elinewidth=.9, mfc='white', zorder=3,
                            label=fr'$\sigma_0^{{\prime 2}}={g.checkpoint_s.iloc[0]:g}$'+flag)
            ax.axvline(1, color=COLORS['boundary'], ls=':', lw=.9)
            ax.axhline(0, color=COLORS['reference'], ls=':', lw=.9)
            ax.tick_params(direction='in', top=True, right=True)
            ax.set_xticks([.9, .95, 1., 1.05, 1.1])
            ax.text(.04, .94, fr'({letter}) $|A|={count}$', transform=ax.transAxes,
                    va='top', fontsize=14)
            ax.legend(loc='lower right', fontsize=10, frameon=False)
        for ax in list(axes.flat)[len(counts):]:
            ax.set_visible(False)
        fig.supxlabel(r'$\sigma^{\prime 2}/\lambda_{\mathrm{mode}}$', fontsize=19, y=.02)
        fig.supylabel(r'$m^2_{\mu,*,-},\quad \kappa(t)$', fontsize=19, x=.015)
        fig.legend(handles=[Line2D([], [], color=COLORS['reference'], marker='o',
                                   mfc='white', ls='none', label='Free-energy difference'),
                            Line2D([], [], color=COLORS['reference'], ls='--',
                                   label='Spectral prediction')],
                   loc='upper center', ncol=2, frameon=False, fontsize=12)
        fig.subplots_adjust(left=.12, right=.98, bottom=.13, top=.90, hspace=.13, wspace=.10)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.6), gridspec_kw={'height_ratios': [1, 1.25, 1]})
        a, b, c = axes
        # No connections between separate stationary backgrounds in panels a/c.
        for key, color, marker, label in [('theory', COLORS['spectrum'], 's', 'Spectral prediction'),
                                         ('measured', COLORS['margin'], 'o', 'Free-energy difference')]:
            a.errorbar(cp.checkpoint_s, cp[key], yerr=cp[key+'_sem'], fmt=marker,
                       color=color, mfc='white' if key == 'theory' else color, ms=6,
                       capsize=3, elinewidth=1.1, label=label, zorder=3)
        a.set_ylabel(r'$m^2_{\mu,*,-},\quad \kappa(t)$')
        a.set_xlabel(r'$\sigma_0^{\prime 2}$')
        a.legend(frameon=False, fontsize=11, loc='best')
        for i, row in enumerate(cp.itertuples()):
            g = values[(values.checkpoint == row.checkpoint) & (values.probe_label != 'checkpoint')].sort_values('s_ratio')
            if g.empty:
                continue
            color = LATENT_COLORS[i % len(LATENT_COLORS)]
            # All curves connect measured grid points only, with no smoothing.
            b.plot(g.s_ratio, g.theory, '--', color=color, lw=1.1, alpha=.65)
            b.errorbar(g.s_ratio, g.measured, yerr=g.measured_sem, color=color,
                       fmt='o-', ms=3.5, lw=1.0, capsize=2,
                       label=fr'${row.checkpoint_s:g}\;({row.n_active})$')
        b.axvline(1, color=COLORS['boundary'], lw=1, ls=':')
        b.set_xlabel(r'$\sigma^{\prime 2}/\lambda_{\mathrm{mode}}$')
        b.set_ylabel(r'$m^2_{\mu,*,-},\quad \kappa(t)$')
        background_legend = b.legend(title=r'$\sigma_0^{\prime 2}\; (|A|)$', ncol=4, fontsize=10,
                 title_fontsize=11, frameon=False, loc='upper left', columnspacing=1.2)
        b.add_artist(background_legend)
        b.legend(handles=[Line2D([], [], color=COLORS['reference'], marker='o', lw=1,
                                 label='Free-energy difference'),
                          Line2D([], [], color=COLORS['reference'], ls='--', lw=1,
                                 label='Spectral prediction')], loc='lower right',
                 fontsize=10, frameon=False)
        # Make space above measured curves for the background legend.
        lo, hi = b.get_ylim()
        b.set_ylim(lo, hi + .38 * (hi-lo))
        c.errorbar(cp.checkpoint_s, cp.edge_shift_percent, yerr=cp.edge_shift_percent_sem,
                   fmt='o', color=COLORS['active'], ms=6, capsize=3, elinewidth=1.1)
        c.set_ylabel(r'$\Delta_{\mathrm{edge}}\;[\%]$')
        c.set_xlabel(r'$\sigma_0^{\prime 2}$')
        flagged = cp[cp.unconverged_repeats > 0]
        if not flagged.empty:
            c.scatter(flagged.checkpoint_s, flagged.edge_shift_percent, marker='x', s=85,
                      color=COLORS['flagged'], zorder=4, label='Includes unconverged mode')
            c.legend(fontsize=10, frameon=False, loc='best')
        for ax, letter in zip(axes, 'abc'):
            ax.axhline(0, color=COLORS['reference'], ls=':', lw=1, zorder=0)
            ax.tick_params(direction='in', top=True, right=True)
            ax.text(-.13, 1.02, f'({letter})', transform=ax.transAxes,
                    fontsize=15, fontweight='bold')
        for ax in (a,c):
            ax.set_xlim(cp.checkpoint_s.min()-.018, cp.checkpoint_s.max()+.018)
        fig.subplots_adjust(left=.19, right=.97, bottom=.07, top=.98, hspace=.52)
    prefix = args.output or args.results / (f'coupled_curvature_{args.layout}_seed{args.seed}' +
                                            ('_corrected' if args.corrected else ''))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(str(prefix)+'.'+extension, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    values.to_csv(str(prefix)+'_values.csv', index=False)
    metadata = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    metadata.update(curvature_column=ykey, error_bars='one SEM across independent measurement repeats',
                    failed_mode_repeats=len(failed), smoothing=False,
                    panel_b='solid: direct finite difference; dashed: spectral prediction; fixed backgrounds',
                    panel_c='paired edge/lambda ratio per repeat, then mean and SEM')
    Path(str(prefix)+'_plot_config.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved {prefix}.pdf / .png / _values.csv / _plot_config.json')
    print(f'Mode repeats failing tolerance: {len(failed)}; excluded={args.exclude_unconverged}')
    if (values.repeats < 2).any():
        print('WARNING: SEM unavailable for groups with fewer than two repeats.')


if __name__ == '__main__':
    main()

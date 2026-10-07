#!/usr/bin/env python3
"""Active count and collapsed-sector mass squared; optional parameter Hessian.

Uses the style and EXACT Gaussian local-linear smoothing implementation from
plot_vae_continuation_smooth.py. Smooth the measured masses, not lambda first.
Do not join across active-set changes, missing checkpoints or flagged points.
Panel (c) is lambda_min of the Euclidean PARAMETER Hessian, not the paper's
function-space mass. Enable this panel with --show-hessian (off by default).
Unit-count flicker is removed at its first transition (s span <= 0.01).
--show-data-eigenvalues adds black lambda labels and reference lines to (a).
--zero-at-collapse constrains (b)'s visual guide, not the raw measurements.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_vae_continuation_smooth import (
    PAPER_STYLE, COLORS, select_seed, active_branches, fit_flags, draw_estimate,
    remove_count_flicker, mark_data_eigenvalues,
)


def mean_sector_mass(s, lam):
    """m^2_{mu,*,-}, rationalized to avoid cancellation near zero.

    Equals (1+s-sqrt((1-s)^2+4*lambda))/(2*s). Negative masses are
    retained; negative residual eigenvalues are invalid and become NaN.
    """
    s,lam=np.broadcast_arrays(np.asarray(s,float),np.asarray(lam,float))
    result=np.full(s.shape,np.nan)
    good=np.isfinite(s)&(s>0)&np.isfinite(lam)&(lam>=0)
    a,b=s[good],lam[good]
    result[good]=2*(a-b)/(a*(1+a+np.sqrt((1-a)**2+4*b)))
    return result


def align_measurements(frame, traj, seed, name):
    frame=select_seed(frame,seed)
    if frame.empty:
        return frame
    if frame.s.duplicated().any():
        raise ValueError(f'Duplicate {name} s values for seed {seed}.')
    if not frame.s.isin(traj.s).all():
        raise ValueError(f'{name} s values do not match trajectory.')
    merged=frame.merge(traj[['s','continuation_index','n_active','n_collapsed']],on='s',
                       suffixes=('','_trajectory'),validate='one_to_one')
    for key in ('continuation_index','n_active','n_collapsed'):
        if key+'_trajectory' in merged and not (merged[key]==merged[key+'_trajectory']).all():
            raise ValueError(f'{name} {key} does not match trajectory.')
    aligned=frame.set_index('s').reindex(traj.s.to_numpy(float))
    aligned['s']=traj.s.to_numpy(float)
    return aligned


def plot_seed(root, seed, args):
    traj=select_seed(pd.read_csv(root/'trajectory.csv'),seed)
    lat=select_seed(pd.read_csv(root/'latent_stats.csv'),seed)
    if traj.empty or lat.empty:
        raise ValueError(f'No trajectory/latent statistics for seed {seed}.')
    if traj.s.duplicated().any():
        raise ValueError(f'Duplicate trajectory s values for seed {seed}.')
    x=traj.s.to_numpy(float)
    branches,transitions=active_branches(traj,lat)
    displayed_counts=remove_count_flicker(x,traj.n_active.to_numpy(),
                                         getattr(args,'collapse_flicker_width',.01))
    count_changes=np.r_[False,np.diff(displayed_counts)!=0]
    transitions=x[count_changes]
    collapses=x[np.r_[False,np.diff(displayed_counts)<0]]
    branches=np.cumsum(np.r_[False,np.diff(branches)!=0] | count_changes)
    show_hessian=getattr(args,'show_hessian',False)
    latent_dim=int((traj.n_active+traj.n_collapsed).max())
    fig,axes=plt.subplots(3 if show_hessian else 2,1,
                          figsize=(8.2,8.3 if show_hessian else 5.8),sharex=True,
                          gridspec_kw={'height_ratios':[1,1.8,1.8] if show_hessian else [1,1.8]})
    diagnostics=pd.DataFrame(dict(seed=seed,continuation_index=traj.continuation_index.to_numpy(),
                                  s=x,n_active=traj.n_active.to_numpy(),branch=branches))
    diagnostics['n_active_display']=displayed_counts
    a,b=axes[:2]
    c=axes[2] if show_hessian else None
    a.step(x,displayed_counts,where='post',lw=2.3,color=COLORS['active'])
    a.set_ylabel(r'$|A|$')
    a.set_ylim(-.25,latent_dim+.4)
    a.set_yticks(np.arange(latent_dim+1))
    b.set_ylabel(r'$m^2_{\mu,*,-}$')
    if show_hessian:
        c.set_ylabel(r'$\lambda_{\min}(H_{\theta,\phi})$')
        c.text(.98,.96,'Full parameter Hessian',transform=c.transAxes,
               ha='right',va='top',fontsize=10,color='.3')
    for ax in axes[1:]:
        ax.axhline(0,color=COLORS['reference'],ls='--',lw=1.3)
    spec_path=root/'spectral_summary.csv'
    spec=align_measurements(pd.read_csv(spec_path),traj,seed,'spectrum') if spec_path.exists() else pd.DataFrame()
    if not spec.empty:
        lam=spec.lambda_inf.to_numpy(float)
        masses=mean_sector_mass(x,lam)
        masses[traj.n_collapsed.to_numpy()==0]=np.nan
        flags=fit_flags(spec,args.chi2_max,args.relative_fit_rms_max)|~np.isfinite(masses)
        draw_estimate(b,x,masses,flags,branches,COLORS['spectrum'],None,args.smooth_width,
                      zero_at=collapses if getattr(args,'zero_at_collapse',False) else ())
        diagnostics['lambda_inf']=lam
        diagnostics['mass_mu_squared']=masses
        diagnostics['mass_flagged']=flags.astype(int)
        if not np.isfinite(masses).any():
            b.text(.5,.5,'No valid collapsed-sector mass estimates',transform=b.transAxes,
                   ha='center',va='center',fontsize=10,color='.4')
    else:
        b.text(.5,.5,'No collapsed-sector spectral measurements',transform=b.transAxes,
               ha='center',va='center',fontsize=10,color='.4')
    if show_hessian:
        hessian_path=root/args.hessian_file
        hs=align_measurements(pd.read_csv(hessian_path),traj,seed,'Hessian') if hessian_path.exists() else pd.DataFrame()
        if not hs.empty:
            kinds=set(hs.hessian_kind.dropna()) if 'hessian_kind' in hs else set()
            metrics=set(hs.parameter_metric.dropna()) if 'parameter_metric' in hs else set()
            if kinds!={'parameter'} or metrics!={'euclidean'}:
                raise ValueError('Panel (c) requires explicitly identified Euclidean parameter Hessian measurements.')
            values=hs.lambda_min_parameter.to_numpy(float)
            required={'converged','ritz_residual','residual_tolerance'}
            if not required.issubset(hs):
                raise ValueError('Hessian CSV lacks convergence diagnostics.')
            residual=hs.ritz_residual.to_numpy(float)
            tolerance=hs.residual_tolerance.to_numpy(float)
            flags=(~np.isfinite(values)|~np.isfinite(residual)|~np.isfinite(tolerance)|
                   (residual>tolerance)|(hs.converged.to_numpy(float)!=1))
            draw_estimate(c,x,values,flags,branches,COLORS['margin'],None,args.smooth_width)
            diagnostics['lambda_min_parameter']=values
            diagnostics['hessian_flagged']=flags.astype(int)
            print(f'{root}, seed {seed}: {np.isfinite(values).sum()} Hessian measurements, '
                  f'{np.sum(flags & np.isfinite(values))} flagged (excluded from smoothing).')
            if not np.isfinite(values).any():
                c.text(.5,.5,'No finite parameter Hessian estimates',transform=c.transAxes,
                       ha='center',va='center',fontsize=10,color='.4')
        else:
            c.text(.5,.5,'Parameter Hessian not measured\n(run measure_vae_parameter_hessian.py)',
                   transform=c.transAxes,ha='center',va='center',fontsize=10,color='.4')
    for letter,ax in zip('abc',axes):
        for sc in transitions:
            ax.axvline(sc,ls=':',color=COLORS['boundary'],lw=1.15,alpha=.8)
        ax.grid(color='#AAAAAA',alpha=.20,linewidth=.55)
        ax.text(-.11,1.02,f'({letter})',transform=ax.transAxes,fontsize=15,fontweight='bold')
        ax.tick_params(direction='in',top=True,right=True)
    if getattr(args,'show_data_eigenvalues',False):
        mark_data_eigenvalues(a,root,int(traj.n_active.max()))
    axes[-1].set_xlabel(r'$\sigma^{\prime 2}$')
    if args.show_notes:
        fig.text(.5,.012,'Dots: estimates; curves: segmented local-linear smoothing.\n'
                 'Open orange dots: flagged.' +
                 (' Panels (b) and (c) use different metrics.' if show_hessian else '') +
                 (' Zero endpoints are imposed visual constraints.' if getattr(args,'zero_at_collapse',False) else ''),
                 ha='center',fontsize=8,color='.35')
    fig.tight_layout(rect=(0,.06,1,.995),h_pad=1.2)
    prefix=root/args.out_prefix
    prefix.parent.mkdir(parents=True,exist_ok=True)
    try:
        for extension in ('pdf','png'):
            path=prefix.with_name(f'{prefix.name}_seed{seed}.{extension}')
            fig.savefig(path,dpi=args.dpi,bbox_inches='tight')
            print(f'Saved: {path}')
        diagnostics.to_csv(prefix.with_name(f'{prefix.name}_seed{seed}_values.csv'),index=False)
    finally:
        plt.close(fig)
    return diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,nargs='+',required=True)
    p.add_argument('--seed',type=int,nargs='+')
    p.add_argument('--hessian-file',type=Path,default=Path('parameter_hessian.csv'))
    p.add_argument('--out-prefix',default='continuation_masses_smooth')
    p.add_argument('--smooth-width',type=float,default=.15)
    p.add_argument('--collapse-flicker-width',type=float,default=.01,
                   help='Maximum s span of unit-count reversals; collapse at first transition. 0 disables.')
    p.add_argument('--zero-at-collapse',action='store_true',
                   help='Constrain panel (b) smooth guides to zero at collapse boundaries.')
    p.add_argument('--show-data-eigenvalues',action='store_true',
                   help='Mark largest max(|A|) covariance eigenvalues on panel (a).')
    p.add_argument('--show-hessian',action='store_true',help='Restore optional panel (c).')
    p.add_argument('--chi2-max',type=float,default=10.)
    p.add_argument('--relative-fit-rms-max',type=float)
    p.add_argument('--dpi',type=int,default=220)
    p.add_argument('--show-notes',action='store_true')
    args=p.parse_args()
    if not np.isfinite(args.smooth_width) or args.smooth_width<=0:
        p.error('smooth-width must be finite and positive.')
    if not np.isfinite(args.collapse_flicker_width) or args.collapse_flicker_width<0:
        p.error('collapse-flicker-width must be finite and nonnegative.')
    plt.rcParams.update(PAPER_STYLE)
    for root in args.results:
        seeds=args.seed if args.seed is not None else sorted(pd.read_csv(root/'trajectory.csv').seed.unique())
        for seed in seeds:
            plot_seed(root,int(seed),args)


if __name__=='__main__':
    main()

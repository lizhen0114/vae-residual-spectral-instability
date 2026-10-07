# VAE Continuation and Coupled Curvature Experiments

This repository contains the scripts and configurations used for the numerical simulations of variational autoencoder (VAE) latent-coordinate collapse on a covariance-matched symmetric Gaussian mixture. The workflow combines continuation training, spectral estimation on frozen checkpoints, and direct free-energy curvature measurements under coupled encoder–decoder perturbations.

All data are generated locally. No external datasets or pretrained models are required.

## Files

| File | Purpose |
| --- | --- |
| `experiment_mixture.json` | Data, model, continuation, and spectral-estimation settings. |
| `vae_two_stage_continuation_lambda_biascorrected_mixture.py` | Train the continuation trajectory, save checkpoints, and estimate the residual spectrum with finite-sample bias extrapolation. |
| `coupled_curvature_mixture.json` | Checkpoint selection and sampling settings for coupled curvature measurements. |
| `measure_vae_coupled_curvature.py` | Measure function-space mean-channel curvature on projected, frozen checkpoints. |
| `plot_vae_continuation_smooth.py` | Plot active-coordinate counts, posterior variances, residual spectral estimates, and stability margins. |
| `plot_vae_masses_smooth.py` | Plot active-coordinate counts and the spectral prediction for the mean-channel mass squared. |
| `plot_vae_coupled_curvature.py` | Compare direct curvature measurements with spectral predictions. |
| `requirements.txt` | Python dependencies. |

Keep these files in the same directory: the curvature measurement script imports the training module, and both the mass and curvature plotting scripts import utilities from `plot_vae_continuation_smooth.py`.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependencies are PyTorch (`torch>=2.0`), NumPy, pandas, and Matplotlib. The supplied training configuration selects `cuda` and requires a CUDA-enabled PyTorch installation and a compatible GPU. CPU execution is supported through `--device cpu`, but the full configuration is computationally intensive. Plotting requires only NumPy, pandas, and Matplotlib; no LaTeX installation is needed.

Run the commands below from the directory containing the scripts and JSON files. For headless training, set `MPLBACKEND=Agg` if needed.

## Experiment definition

The data distribution is an equally weighted, symmetric two-component Gaussian mixture in eight dimensions. Let

$$
v=\frac{1}{\sqrt{8}}(1,\ldots,1)^\top,\qquad
G\sim\mathcal N(0,I_8),\qquad
B\in\{-1,+1\},\quad \Pr(B=\pm1)=\tfrac12.
$$

For mixture parameter $0\leq a<1$, the samples are constructed as

$$
Y=(I_8-a^2vv^\top)^{1/2}G+aBv,\qquad
X=QD^{1/2}Y,
$$

where $Q$ is a seeded orthogonal rotation and
$D_{jj}=\exp[-0.1(j-1)]$, $j=1,\ldots,8$, under the supplied configuration. The population covariance is $QDQ^\top$ for every $a$. At $a=0$, the distribution reduces to a Gaussian. Generated samples are centered by their sample mean without empirical whitening.

The model is an MLP VAE with a four-dimensional latent space, a standard normal prior, a diagonal Gaussian encoder, and ReLU hidden layers. Encoder widths are `256, 128`; decoder widths are `128, 256`. Training minimizes the negative ELBO, up to parameter-independent constants:

$$
\mathcal F_s=
\mathbb E_{x,z\sim q_\phi(z\mid x)}
\left[\frac{\|x-f_\theta(z)\|^2}{2s}\right]
+\mathbb E_x\operatorname{KL}\!\left(q_\phi(z\mid x)\,\|\,\mathcal N(0,I)\right).
$$

Here `s` is the observation variance, labeled $\sigma^{\prime 2}$ in the figures. A coordinate is classified as active when its mean KL contribution exceeds `active_kl_threshold` (`0.001`). This classification is diagnostic; training does not clamp collapsed coordinates.

The supplied `experiment_mixture.json` uses:

| Setting | Value |
| --- | --- |
| Mixture parameter `mixture_a` | `0.5` |
| Dataset size / data seed | `256000` / `1234` |
| Training seeds | `[0]` |
| Continuation grid | `0.65` to `1.05`, inclusive, in steps of `0.001` (401 checkpoints) |
| Training epochs | 120 at the first point; 30 at each subsequent point |
| Batch size / learning rate | `4096` / `0.001` |
| Spectral data resolutions | `1024, 2048, 4096` |
| Spectral latent-sample resolutions | `256, 512, 1024` |
| Repeats per resolution pair | `16` |
| Collapsed-coordinate Monte Carlo samples | `128` |
| Bootstrap replicates | `500` |
| Output directory | `results_mixture_a05_zc128` |

## Training and spectral estimation

Run the complete two-stage experiment:

```bash
python vae_two_stage_continuation_lambda_biascorrected_mixture.py \
  --config experiment_mixture.json
```

The first stage increases `s` along the configured grid, initializing each model from the preceding checkpoint. Adam is recreated at each continuation point. The second stage freezes the saved models, averages the decoder over collapsed coordinates under their standard normal prior, and estimates the leading residual-operator eigenvalue for checkpoints with at least one collapsed coordinate.

With `bias_extrapolation: true`, the spectral estimator fits

$$
\widehat\Lambda(B,L)=\Lambda_\infty+\frac{c_B}{B}+\frac{c_L}{L},
$$

using inverse-variance weights across the data and latent-sample resolutions. Bootstrap resampling supplies uncertainty estimates for the intercept `lambda_inf`. Fully active checkpoints have no collapsed-sector spectral estimate.

The stages can also be run separately:

```bash
# Stage 1: train and save checkpoints.
python vae_two_stage_continuation_lambda_biascorrected_mixture.py \
  --config experiment_mixture.json --train-only

# Stage 2: measure the saved trajectory.
python vae_two_stage_continuation_lambda_biascorrected_mixture.py \
  --config experiment_mixture.json --estimate-only
```

To continue interrupted training, add `--resume`. Resumption starts after the last completed continuation point; it does not restore a partially completed epoch. A plain run requires an empty output directory. Use `--outdir NEW_DIRECTORY` for a separate experiment. Relative training output paths are resolved from the current working directory.

`--estimate-only` recomputes spectral measurements and updates the measurement tables; it does not resume an interrupted spectral pass. Preserve previous measurements separately if they are needed for comparison.

Generate the continuation and mass figures after spectral estimation:

```bash
python plot_vae_continuation_smooth.py \
  --results results_mixture_a05_zc128 --seed 0

python plot_vae_masses_smooth.py \
  --results results_mixture_a05_zc128 --seed 0
```

The mass plot evaluates

$$
m^2_{\mu,*,-}(s,\Lambda)=
\frac{1+s-\sqrt{(1-s)^2+4\Lambda}}{2s}
$$

using `lambda_inf`. Negative masses are retained; invalid negative spectral estimates are excluded.

## Direct coupled curvature measurements

This measurement requires the training checkpoints and training metadata. It can run after `--train-only`; the bias-extrapolated spectral tables are not required.

```bash
python measure_vae_coupled_curvature.py \
  --config coupled_curvature_mixture.json

python plot_vae_coupled_curvature.py \
  --results results_coupled_curvature_a05_dense \
  --seed 0 --amplitude 0.01 --layout active
```

The supplied configuration selects training seed `0` and checkpoints at `s = 0.75, 0.84, 0.93, 1.02`. Each checkpoint uses four measurement repeats, 4096 data points, 1024 latent samples for mode estimation, and 16384 independent latent validation samples on the same empirical data set. Decoder projection uses 128 collapsed-coordinate samples. Calculations use float64 with TF32 disabled; `device: "auto"` selects CUDA when available.

For each frozen background, the script constructs a centered leading residual mode and a coupled decoder/encoder mean perturbation. It compares the spectral prediction with the symmetric free-energy difference

$$
\kappa(t)=\frac{\mathcal F(+t)+\mathcal F(-t)-2\mathcal F(0)}{t^2}.
$$

Measurements use amplitudes `0.01, 0.02, 0.04, 0.08`, both at the checkpoint variance and at 21 probe variances from `0.90` to `1.10` times the repeat-specific `lambda_mode`. The model and projected background stay fixed during this probe scan. `lambda_mode` is estimated for this measurement and is distinct from the extrapolated `lambda_inf` used in the continuation plots. This is a function-space curvature calculation; it does not compute the neural-network parameter Hessian or retrain at the probe variances.

Add `--corrected` to the plotting command to subtract the analytically known quartic finite-amplitude contribution. Add `--exclude-unconverged` to remove repeats whose mode iteration fails the configured tolerance. The default plot includes those repeats and reports their count in the console and plot metadata. Error bars represent one standard error across measurement repeats, rather than across training seeds. `--layout overview` additionally shows checkpoint comparisons and the estimated block-edge shift.

In the curvature JSON, `results_dir` and `output_dir` are resolved relative to the JSON file's directory. The output directory must be empty and outside the training results directory. If training uses another output path, update `results_dir` accordingly. Requested checkpoint values must exist in the saved trajectory. Checkpoints without a collapsed coordinate are recorded in `skipped.json`.

## Outputs

Training and spectral outputs are written to `results_mixture_a05_zc128/` by default:

| Output | Contents |
| --- | --- |
| `config.json` | Resolved training configuration. |
| `seed_0/step_*.pt` | Model weights, coordinate statistics, active masks, and RNG states at each continuation point. |
| `trajectory.csv` | Continuation indices, variances, active counts, losses, and checkpoint paths. |
| `latent_stats.csv` | Per-coordinate posterior variance, squared mean, KL, activity, and rank. |
| `data_covariance_eigenvalues.csv` | Prescribed population covariance eigenvalues. |
| `measurement_config.json` | Configuration used for spectral estimation. |
| `lambda_grid_repeats.csv` | Individual spectral estimates at each resolution pair. |
| `lambda_grid_summary.csv` | Means and standard errors by resolution pair. |
| `spectral_summary.csv` | Extrapolated eigenvalues, bootstrap intervals, stability margins, and fit diagnostics. |
| `two_stage_summary_seed0.{pdf,png}` | Summary figures generated by the experiment script. |
| `continuation_summary_smooth_seed0.{pdf,png}` | Continuation figures from the dedicated plotting script. |
| `continuation_masses_smooth_seed0.{pdf,png}` | Mean-channel mass figures. |
| `continuation_masses_smooth_seed0_values.csv` | Mass-plot values and diagnostic flags. |

Curvature outputs are written to `results_coupled_curvature_a05_dense/`:

| Output | Contents |
| --- | --- |
| `measurement_config.json` | Measurement settings, source configuration, resolved device, and PyTorch version. |
| `coupled_curvature.csv` | Individual measurements, predictions, moment diagnostics, and convergence information. |
| `summary.csv` | Repeat means, standard errors, and counts by checkpoint, amplitude, and probe. |
| `skipped.json` | Checkpoints skipped because all coordinates are active. |
| `coupled_curvature_active_seed0.{pdf,png}` | Default curvature comparison figures. |
| `coupled_curvature_active_seed0_values.csv` | Aggregated values used in the plot. |
| `coupled_curvature_active_seed0_plot_config.json` | Plot settings and convergence-filter metadata. |

## Plotting and reproducibility notes

The continuation and mass plots show Gaussian local-linear smooth guides with default `--smooth-width 0.15`. Spectral and mass guides are broken at active-set changes, missing values, and flagged estimates. The variance panel connects the original points with straight segments. By default, short unit-count reversals over a span of at most `0.01` in `s` are removed from the displayed active count; `--collapse-flicker-width 0` preserves the raw counts.

The optional `--zero-at-collapse` flag imposes zero endpoints on the smooth guides without changing the measured points. Smoothing and endpoint constraints are display choices, not additional measurements. The coupled curvature plot does not smooth its measurements or force zero crossings. The optional parameter-Hessian panel in the mass plot requires separately supplied Hessian data; its measurement script is outside this release.

The supplied configuration uses one training seed. Measurement repeats and bootstrap intervals quantify estimator variability for the frozen models, not variability across independently trained models. Seeds and configurations are recorded, but exact numerical agreement can depend on hardware, PyTorch/CUDA versions, and floating-point execution. Dependencies are not version-locked; record the environment alongside results when reproducing a run.

For a small local check, copy the experiment JSON, choose a new `outdir`, reduce the data size, continuation range, and epoch counts, and run with `--device cpu --train-only`. Reduced settings check the workflow but do not reproduce the supplied experiment. If spectral estimation is enabled, keep `eval_points <= n_samples` and ensure the resolution scales generate at least two distinct resolutions in each dimension when bias extrapolation is enabled.

Each script provides a `--help` option for its command-line interface.

# SIC counterfactuals

> General optimizer and plotting examples. For reported task-specific commands, use [the active guide](../README.md).

For the paired full-trial Mahalanobis typicality study, source-only
threshold calibration, complete optimization archives, and offline manuscript
tables/figures, see [typicality workflow](../typicality/README.md). Its entry point is
`python -m eegproc_xai_cfo.model_explainability.typicality.runner`.

Counterfactual optimization and plotting utilities.

| File | Owns |
| --- | --- |
| `arguments.py` | CLI arguments and their validation; no TensorFlow imports. |
| `loss.py` | `CounterfactualLoss`: target, latent, decoded, physiological, and central loss methods. |
| `vcsc_calibration_dreamer.npz` | Measured DREAMER calibration curves used by the VCSC term. |
| `optimizer.py` | Full-trial latent gradient descent with the saved SIC classifier and decoders. |
| `runner.py` | Model/data loading, trial selection, printed diagnostics, and result files. |
| `plotting.py` | Shared counterfactual NPZ loading and channel-label helpers. |
| `heatmap.py` | Channel-by-time decoded counterfactual difference heatmap. |
| `topography.py` | Side-by-side whole-trial scalp maps for each EEG band. |
| `training_monitor.py` | Static or live epoch × target probability × difference trajectory. |

## Install in EEGProc-XAI-CFO

Keep these `.py` files together in `src/eegproc_xai_cfo/model_explainability/counterfactuals/`.
The new runner does not import the old `counterfactual_losses.py` or
`counterfactual_state.py`. Leave those old files in place if other code still
imports them; this rewrite is not a compatibility layer for their APIs.
No change to `cross_val.py`, SIC training, or the SIC model is required. The
runner supports SICModelv11's independent reconstructions and SICModelv15's
learned joint reconstruction.

Use the existing EEGProc-XAI-CFO environment. Tests here used Python 3.12,
TensorFlow CPU 2.20.0, Keras 3.15.1, and NumPy.

## Local smoke test

From the EEGProc-XAI-CFO repository root with its virtual environment activated, run:

```bash
PYTHONPATH=src python -m eegproc_xai_cfo.model_explainability.counterfactuals.runner --help
```

This verifies that Python can import TensorFlow and the counterfactual modules
and build the command-line interface. It does not load a checkpoint, process
data, create output files, or run optimization.

## Run from existing raw-data files

From the EEGProc-XAI-CFO repository root, replace the checkpoint/data paths below.
The window, sampling-rate, normalization, label, and filtering settings must
match the original training run. These example values are not auto-detected.

```bash
PYTHONPATH=src python -m eegproc_xai_cfo.model_explainability.counterfactuals.runner \
  --model /path/to/loso_target_0_zero_shot.keras \
  --raw-eeg-npy /path/to/dreamer_eeg.npy \
  --raw-labels-npy /path/to/dreamer_labels.npy \
  --dataset dreamer \
  --label-dimension valence \
  --fs 128 --window-sec 1 --window-overlap 0 \
  --window-normalization global_rms \
  --label-threshold-mode global --median-label 3 \
  --subject-id 0 --trial-id 0 \
  --target-probability 0.8 \
  --learning-rate 0.01 --max-steps 20 \
  --target-weight 1 --latent-weight 0.1 --decoded-weight 0.1 \
  --physiological-weight 0 \
  --log-every 1 \
  --out-dir runs/counterfactuals/subject0_trial0_smoke
```

Add `--remove-median-label` if that trial exclusion was used in training.
Use `--trial-ids 0 1 2` for a subset, or omit trial selection to process all
trials of the specified subject. The default target is the opposite of the
original predicted class for a binary model. Set `--target-class 0` or `1`
explicitly if desired. Multiclass models require an explicit target.

For SICModelv15's single learned joint reconstruction, add:

```bash
  --model-module eegproc_xai_cfo.deep_learning.joint_architectures.SICModelv15.sic_model \
  --decoder-mode joint
```

Joint mode requires a v15 checkpoint trained with the decoder and both encoder
branches enabled. It runs both branch decoders and applies the checkpoint's
learned convex fusion weight; it does not train or recalibrate that weight.

Raw mode reuses `load_sic_training_data` and `_group_windows_into_trials` from
the matching SIC training module. It does not call any training function or
estimate a new MI adjacency. The default model registration module is
`eegproc_xai_cfo.deep_learning.joint_architectures.SICModelv11.sic_model`; override
`--model-module` if your package path differs. Raw mode expects its sibling
`sic_model_train` module.

The current training metadata does not contain every preprocessing setting,
so raw mode requires explicit `--label-dimension`, `--fs`, `--window-sec`, and
`--window-normalization`. The remaining preprocessing options are printed by
`--help` and saved in `settings.json`; check them against the training command.

## Run from prepared trials

An NPZ must contain:

| Key | Shape | Meaning |
| --- | --- | --- |
| `features` | `(N, W, T, F)` | Preprocessed, normalized, chronological trial windows. |
| `subject_ids` | `(N,)` | Integer subject ID for each trial. |
| `trial_ids` | `(N,)` | Integer trial ID, unique within each subject. |
| `labels` (optional) | `(N,)` | Integer true labels, recorded only as metadata. |

For example, after the existing SIC data loading/grouping steps:

```python
np.savez_compressed(
    "prepared_trials.npz",
    features=X,
    subject_ids=subjects,
    trial_ids=trials,
    labels=y,
)
```

```bash
PYTHONPATH=src python -m eegproc_xai_cfo.model_explainability.counterfactuals.runner \
  --model /path/to/loso_target_0_zero_shot.keras \
  --trials-npz prepared_trials.npz \
  --subject-id 0 --trial-id 0 \
  --max-steps 200 --out-dir runs/counterfactuals/subject0_trial0
```

Prepared inputs are not normalized or filtered again. No padding, cropping,
mask inference, or averaging is introduced. The inspected SIC trainer groups
equal numbers of real windows per trial and its classifier has no padding
mask. An NPZ containing a `window_mask` with invalid windows is rejected.

## Slurm: all 18 trials for subject 0

Submit the 18-task array with the subject 0 SICModelv15 checkpoint:

```bash
sbatch \
  --export=ALL,MODEL_PATH=/absolute/path/to/subject_0_sic_v15.keras \
  src/eegproc_xai_cfo/model_explainability/counterfactuals/slurm/run_joint_counterfactuals_user_0.sh
```

The array indices are exactly `0-17`; each task processes the matching DREAMER
trial for subject 0 in joint-decoder mode. Raw data defaults to
`datasets/dreamer_eeg.npy` and `datasets/dreamer_labels.npy`. To use a prepared
trial file instead, also export `TRIALS_NPZ=/absolute/path/prepared_trials.npz`.
The optimization settings can be overridden through the environment variables
listed near the top of the script. In particular, set
`PHYSIOLOGICAL_WEIGHT` to make VCSC part of the optimized objective; its
default is `0.0`, which computes and reports VCSC as a diagnostic only. The
VCSC distance and z-score settings are also exposed as environment variables.

Every task also creates the joint heatmap, signed topography, and optimization
trajectory. Outputs are grouped under:

```text
XAI_runs/subject_0_joint_counterfactuals_<array-job-id>/
├── all_metrics.json
├── all_metrics.json.lock
├── trial_00/subject_0_trial_0/
├── ...
└── trial_17/subject_0_trial_17/
```

On exit, every task runs the locked, atomic metrics collector. The last
successful task therefore leaves `all_metrics.json` containing each complete
trial's result metrics, complete optimization history, settings, artifact list,
and cross-trial numeric summaries. `complete` is true only after all 18 trials
are present. Failed or unfinished trials remain listed in `missing_trial_ids`.

To refresh or validate the combined file manually:

```bash
PYTHONPATH=src python -m \
  eegproc_xai_cfo.model_explainability.counterfactuals.aggregate_metrics \
  XAI_runs/subject_0_joint_counterfactuals_<array-job-id> \
  --subject-id 0 --expected-trials 18 --require-complete
```

## Model-agnostic workflow

The adapter-based workflow now lives in [`../model_agnostic/`](../model_agnostic/README.md). Its APIs and commands are documented there.

## Outputs

The output directory must be new or empty; existing results are never
overwritten. Completed trials are saved individually.

| File | Contents |
| --- | --- |
| `settings.json` | Arguments, loss weights, model path, input shape, selected trials, environment version. |
| `subject_<id>_trial_<id>/history.csv` | Step 0 and each finite evaluated step: total/raw/weighted losses, selected reconstruction-path MSEs, probabilities, prediction, success, gradient norm. |
| `subject_<id>_trial_<id>/result.json` | Original and optimized latent predictions, optional latent typicality; reconstruction fidelity, distances, losses, VCSC, step and timing. |
| `subject_<id>_trial_<id>/counterfactual.npz` | `x`, `z`, `z_prime`, decoded baseline/counterfactual signals, and original/optimized classification embeddings; joint mode uses `<path>=joint`. |
| `results.json` | Completed trial summaries, updated after each trial. |
| `summary.json` | Latent success, class-flip, distance, probability-change, and per-output VCSC diagnostics. |


## Quick start: generate the graphs

Run these commands from the EEGProc-XAI-CFO repository root after activating the same
Python environment used for the counterfactual run:

```bash
cd /path/to/extracted-source
source venv/bin/activate
```

Each completed trial directory contains the two plotting inputs:

```text
runs/counterfactuals/YOUR_RUN/subject_0_trial_0/
├── counterfactual.npz   # difference heatmap and scalp topography
└── history.csv          # three-dimensional optimization trajectory
```

Replace `YOUR_RUN` and the subject/trial numbers in the commands below with an
actual completed result directory.

### 1. Counterfactual difference heatmap

```bash
PYTHONPATH=src python -m eegproc_xai_cfo.model_explainability.counterfactuals.heatmap \
  runs/counterfactuals/YOUR_RUN/subject_0_trial_0/counterfactual.npz \
  --branch joint \
  --sampling-rate 128
```

This interprets the 42 features as 14 electrodes × 3 bands and creates three
stacked heatmap sections: Theta, Alpha, and Beta. Each section contains the
same 14 channel rows, its own peak-channel summary, and the same time axis. It
shows RMS decoded change in one-second bins by default, avoiding the dense
phase-driven striping produced by plotting every signed EEG sample. Each band
has its own labeled robust color scale so lower-amplitude band structure stays
visible. It opens the graph and saves
`counterfactual_joint_counterfactual_difference_heatmap.png` beside the NPZ.
The default difference is
`x_prime_joint - x_reconstructed_joint`, which isolates the intervention
from decoder reconstruction error. Use `--reference input` only to include
that reconstruction error. Use `--time-bin-seconds 0.5` for finer temporal
resolution, `--measure mean-absolute` for average magnitude, or
`--measure signed-mean` when the direction of a binned change is meaningful.
Add `--shared-scale` when direct cross-band color comparison is more important
than seeing lower-amplitude within-band structure.

### 2. Whole-trial scalp topographies by band

```bash
PYTHONPATH=src python -m eegproc_xai_cfo.model_explainability.counterfactuals.topography \
  runs/counterfactuals/YOUR_RUN/subject_0_trial_0/counterfactual.npz \
  --branch joint
```

This interprets the 42 features as 14 electrodes × 3 bands, opens three
side-by-side scalp maps (Theta, Alpha, and Beta), prints the peak channel for
each band, and saves
`counterfactual_joint_difference_topography.png` beside the NPZ. The default
maps show the signed mean counterfactual difference over the complete trial on
a zero-centered red/blue scale. Positive and negative electrodes therefore
remain distinguishable. Each band has its own symmetric color range, preserving
spatial contrast within lower-amplitude bands. Colors should be compared within
a band, not between bands; the adjacent colorbars retain the actual values.
Add `--shared-scale` only when direct cross-band magnitude comparison is
desired. Use `--measure mean-absolute` to restore unsigned activity magnitude.

For 14-channel data, the DREAMER order is automatic: AF3, F7, F3, FC5, T7,
P7, O1, O2, P8, T8, FC6, F4, F8, AF4. The scalp map currently requires these
positioned DREAMER/Emotiv channel names. Mean absolute activity is used so
positive and negative EEG values do not cancel only when
`--measure mean-absolute` is selected; the default signed mean intentionally
retains cancellation and direction.

The default flattened feature order is channel-major: Theta/Alpha/Beta for
AF3, then Theta/Alpha/Beta for F7, and so on. If the saved features instead
contain all channels for Theta followed by all channels for Alpha and Beta, add
`--feature-order band-major`. Override the labels, if needed, with for example
`--band-names Delta Theta Alpha`.

### 3. Training/optimization trajectory

```bash
PYTHONPATH=src python -m eegproc_xai_cfo.model_explainability.counterfactuals.training_monitor \
  runs/counterfactuals/YOUR_RUN/subject_0_trial_0/history.csv
```

This opens the three-dimensional graph and saves
`history_counterfactual_training_trajectory.png` beside the CSV. Its axes are:

- x: optimization step, displayed as epoch
- y: `decoded`, the selected reconstruction-path MSE relative to the original
  reconstruction R(z) in new runs (older runs used the observed input)
- z: `target_probability`, displayed as `target_p` and zoomed to its observed
  variation so small changes remain visible

The current runner writes `history.csv` after a trial finishes, so this command
plots the completed trajectory. The plotting code contains a TODO for coloring
the trajectory by validity later.

Add `--full-probability-range` to restore a fixed y-axis from 0 to 1.

### Common options

- Joint-mode outputs use `--branch joint`.
- Branch-mode outputs use `--branch gcn_gru` or `--branch bilstm`.
- When the NPZ contains multiple reconstruction paths, `--branch` is required.
- Use `--band-names NAME1 NAME2 NAME3` to change the three band labels.
- Use `--feature-order band-major` if features are grouped by band rather than channel.
- Add `--no-show` to save the PNG without opening a graph window.
- Add `--output figures/my_plot.png` to choose a different output path.
- Append `--help` to any plotting command to see all available options.

## Verification

Unit-test scripts are not included in this review repository. See the [provenance guide](../../../../reproducibility/README.md) for the recorded-run and metric checks. Full experiments require DREAMER and the matching checkpoints.

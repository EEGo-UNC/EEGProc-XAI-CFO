# EEGProc-XAI-CFO

This frozen research repository provides EEG preprocessing, feature extraction, subject-independent emotion recognition, and model explainability. The experiments center on the **Subject Invariant Calibrator (SIC)** for DREAMER valence and arousal classification.

SIC combines a GCN-GRU branch and a BiLSTM branch, preserves the order of one-second EEG windows in a trial-level recurrent classifier, and uses a variational classifier head. The training code supports leave-one-subject-out (LOSO) evaluation, first-order MLDG subject generalization, target-subject calibration, reconstruction, and ablation studies. Version 15 adds a learned convex fusion of the two EEG reconstructions. The explainability package studies latent counterfactuals, typicality, physiological checks, and matched ablations.

## Reported experiments

Start with [recorded-run provenance and commands](reproducibility/README.md).

## Where to start

| Area | Guide |
| --- | --- |
| Reported SIC v15 arousal training and two-GPU folds | [SICModelv15/README.md](src/eegproc_xai_cfo/deep_learning/joint_architectures/SICModelv15/README.md) |
| Reported SIC v11 valence training | [SICModelv11/README.md](src/eegproc_xai_cfo/deep_learning/joint_architectures/SICModelv11/README.md) |
| Counterfactual explainability and four-arm ablations | [model_explainability/README.md](src/eegproc_xai_cfo/model_explainability/README.md) |
| Core preprocessing and feature extraction | [`src/eegproc_xai_cfo/`](src/eegproc_xai_cfo/) and [local documentation](docs/source/index.rst) |

## Setup

Use Python 3.12 and run commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install tensorflow
```

The package metadata installs the core dependencies; TensorFlow is also required for SIC training and counterfactuals. For NVIDIA GPUs, use a TensorFlow/CUDA installation compatible with your machine. The training and explainability guides show `PYTHONPATH=src python -m ...` commands, so they also work from a source checkout.

To load a saved `.keras` model created before or after the package rename, use `eegproc_xai_cfo.checkpoints.load_model`. The training and explainability commands use this loader where they consume existing checkpoints.

The DREAMER experiments require the dataset, which is not included here. Download [DREAMER from Kaggle](https://www.kaggle.com/datasets/phhasian0710/dreamer) separately and follow the [CSV preparation prerequisites](src/eegproc_xai_cfo/deep_learning/joint_architectures/README.md). Place the EEG and label arrays at `datasets/dreamer_eeg.npy` and `datasets/dreamer_labels.npy`, or replace those paths in the examples. Use the same label threshold, one-second windows, 128 Hz sampling, and `global_rms` normalization for training and explainability of a saved model.

## Compute and outputs

Training supports multiple GPUs and parallel computing. SIC v11 can train separate LOSO folds concurrently on different GPUs. SIC v15 can additionally split one MLDG fold across two GPUs while running other folds in parallel. The counterfactual ablations can run subject shards as independent Python processes on separate GPUs. See each guide for the corresponding flags and output layout.

Training writes run configurations, metrics, and LOSO checkpoint manifests under the chosen `--out-dir`. Explainability runs consume those frozen checkpoints and write study manifests, per-trial results, and reports under their own `--out-dir`. Keep the model version, task, preprocessing, and checkpoint manifest matched when comparing runs.

## License

This project is licensed under GPLv2; see [LICENSE](LICENSE).

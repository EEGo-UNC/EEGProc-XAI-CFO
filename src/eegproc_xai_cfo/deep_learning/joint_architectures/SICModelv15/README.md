# SICModelv15: reported arousal configuration

The [fixed arousal JSON](best_arousal_hyperparameters.json) matches configuration 1 in the designated arousal `hyperparameter_search_summary.csv`. Full run identity, scores, saved configurations are in the [reproduction guide](../../../../../reproducibility/README.md).

## Recorded settings

The source budget is **10 epochs × 12 MLDG episodes per epoch**, with **1 calibration epochs**. The configuration uses 12 meta-train / 6 meta-test subjects, 2 trials per subject, `vc_alpha=2.0`, `vc_beta=0.6`, `vc_lambda=0.05`, `vc_logit_scale=64`, reconstruction weight 0.6, and subject loss weight 0.2. Layer-width lists have `fixed` wrappers so the CLI expands this file to exactly one model.

## Reproduce

Follow [DREAMER preparation](../README.md), then run from the repository root:

```bash
bash reproducibility/train_arousal.sh
```

This is an effective command reconstructed from saved training settings, not an original shell-history transcript. Inspect the script for every option. The saved model and dataset must remain matched when running [counterfactuals](../../../model_explainability/README.md).

The recorded hardware layout is `--n-jobs 2 --gpus-per-fold 2 --gpu-ids 0 1 2 3 --cpus-per-worker 4`. For a one-GPU development run, append `--n-jobs 1 --gpus-per-fold 1 --gpu-ids 0`; for a single-target smoke check also append `--target-subjects 0`. Hardware and smoke adaptations are not the original full run. Two-GPU MLDG requires trial-level classification and `gcn_use_batch_norm=false`.

Each run writes a timestamped directory with settings, metrics, `configuration_0001/loso_zero_shot_models.json`, and frozen source checkpoints under `configuration_0001/loso_zero_shot_models/`.
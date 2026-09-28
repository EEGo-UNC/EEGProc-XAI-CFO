# SICModelv11: reported valence configuration

The [fixed valence JSON](best_valence_hyperparameters.json) matches configuration 1 in the designated valence `hyperparameter_search_summary.csv`. Full run identity, scores, saved configurations are in the [reproduction guide](../../../../../reproducibility/README.md).

## Reproduce

Follow [DREAMER preparation](../README.md), then run from the repository root:

```bash
bash reproducibility/train_valence.sh
```

This is an effective command reconstructed from saved training settings, not an original shell-history transcript. Inspect the script for every option. The saved model and dataset must remain matched when running [counterfactuals](../../../model_explainability/README.md).

The recorded hardware layout is `--n-jobs 4 --gpu-ids 0 1 2 3 --cpus-per-worker 2`. For a one-GPU development run, append `--n-jobs 1 --gpu-ids 0`; for a single-target smoke check also append `--target-subjects 0`. These are execution adaptations.

Each run writes a timestamped directory with settings, metrics, `configuration_0001/loso_zero_shot_models.json`, and frozen source checkpoints under `configuration_0001/loso_zero_shot_models/`.
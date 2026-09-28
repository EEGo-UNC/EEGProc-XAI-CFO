# Reported-run provenance

These records identify the two runs designated for the reported DREAMER results. The supplied CSV excerpts were compared with the saved run CSVs and match exactly. Each CSV contains **one configuration**, `configuration_0001`. The effective commands below were reconstructed from saved configuration files and counterfactual manifests; the original shell invocation and the earlier model-selection history were not recorded here.

## Recorded experiments

| Setting | Arousal | Valence |
| --- | --- | --- |
| Model | SIC v15 | SIC v11 |
| Training suite | 1599318 | 65452590 |
| Source epochs | **10** | **3** |
| MLDG episodes per epoch | **12** | **20** |
| Meta-train / meta-test subjects | 12 / 6 | 8 / 4 |
| Trials per episode subject | 2 | 3 |
| Calibration epochs | 1 | 20 |
| Recorded ranking level | LOSO zero-shot | LOSO zero-shot |
| Recorded ranking metric | Balanced accuracy, maximize | Balanced accuracy, maximize |
| Ranking score | 0.5606704853827617 | 0.24907571198509296 |
| Counterfactual target probability | **0.8** | **0.5** |
| Decoder | Learned joint v15 | Fixed joint v11 |
| Counterfactual eligible trials | 57 | 92 |


## Effective training commands

From the repository root, after [preparing DREAMER](../src/eegproc_xai_cfo/deep_learning/joint_architectures/README.md):

```bash
bash reproducibility/train_arousal.sh
bash reproducibility/train_valence.sh
```

These scripts make the saved training options explicit and write new timestamped runs under `runs/reproduction/`. They retain the original parallelism: two two-GPU folds for arousal, four one-GPU folds for valence. Additional CLI arguments can override input/output locations or hardware. A one-GPU arousal adaptation is `bash reproducibility/train_arousal.sh --n-jobs 1 --gpus-per-fold 1 --gpu-ids 0`; this changes execution hardware and is not evidence of bitwise reproduction. All JSON layer-width arrays use `fixed` wrappers so each file expands to one model.

For counterfactual commands, use [the active guide](../src/eegproc_xai_cfo/model_explainability/README.md) or `python reproducibility/reproduce_counterfactuals.py --help`. The latter reads the saved manifest arguments, substitutes local paths, and covers all 23 subjects by default. `--dry-run` prints the effective commands without loading TensorFlow. Use the zero-shot checkpoints from the matching task and model version. The snapshots record the historical model filenames and checkpoint hashes; model files and raw data are not distributed.
# Counterfactual paper evaluation

The supported workflow is implemented in `eegproc_xai_cfo.model_explainability.typicality.runner`. The former readiness-gate runbook and placeholder launchers have been retired.

1. Follow the [DREAMER preparation guide](../deep_learning/joint_architectures/README.md).
2. Train or obtain matching **v15 arousal** and **v11 valence** zero-shot checkpoints using the [recorded configurations](../../../reproducibility/README.md).
3. Run the four objectives using the [active guide](README.md). Recorded targets are 0.8 for arousal and 0.5 for valence; v11 uses fixed joint GCN-GRU weight 0.49751.
4. Aggregate completed folds:

```bash
PYTHONPATH=src python -m eegproc_xai_cfo.model_explainability.typicality.results runs/cfo_arousal_four_arm/fold_* runs/cfo_valence_four_arm/fold_* --out-dir runs/cfo_combined_report
```

Check `fold_status.csv` and `results.json`: each intended task must have subjects 0–22 and completed folds before calling a population report complete. Supplying a subset only aggregates that subset. All eligible trials, including failed targets, remain in the report. Validity is evaluated in the optimized classification embedding; it does not claim decoded signals were re-encoded and classified successfully. Missing physiological checks remain unavailable.

`delta_dec` and `e_rec` are **RMSE diagnostics**; raw endpoint fields and optimization losses remain MSE. See the [metric provenance audit](../../../reproducibility/README.md#reconstruction-metrics-retain-rmse-in-the-reported-diagnostics) before editing manuscript equations or numbers. All 596 final-report endpoints were checked.

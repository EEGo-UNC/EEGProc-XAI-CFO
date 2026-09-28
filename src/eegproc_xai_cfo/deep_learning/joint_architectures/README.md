# Prepare DREAMER and run SIC

Run these commands from the repository root after [installing the package](../../../../README.md). The converter is `eegproc_xai_cfo.deep_learning.prepare_datasets`; there is no root-level `prepare_datasets.py`.

## Dataset download

Download DREAMER separately from [Kaggle](https://www.kaggle.com/datasets/phhasian0710/dreamer). No raw DREAMER data files or prepared dataset arrays are included in this repository or review ZIP.

## Create dreamer_joined.csv

The included [join_dreamer.py](../join_dreamer.py) follows the supplied `open_datasets/join_dreamer.py`: it keeps every EEG row, attaches ECG on subject/trial/segment/sample index, and attaches ratings on subject/trial. Explicit paths replace the original hard-coded Downloads folder. Duplicate ECG or label keys are rejected to prevent multiplying EEG rows.

Place `dreamer_eeg.csv`, `dreamer_ecg.csv`, and `dreamer_labels.csv` together, then run:

```bash
PYTHONPATH=src python -m eegproc_xai_cfo.deep_learning.join_dreamer --input_dir /absolute/path/to/exported_csvs --output_dir datasets/raw
PYTHONPATH=src python -m eegproc_xai_cfo.deep_learning.prepare_datasets --dataset dreamer --dreamer_dir datasets/raw --output_dir datasets --verify
```

The first command writes `datasets/raw/dreamer_joined.csv`; the second produces the training arrays. Only the helper code is distributed, not its input or output data files.

## Existing joined CSV input

```bash
PYTHONPATH=src python -m eegproc_xai_cfo.deep_learning.prepare_datasets --dataset dreamer --dreamer_dir /absolute/path/to/dreamer_raw --output_dir datasets --verify
```

The directory must contain `dreamer_joined.csv` directly. Required columns are `subject_id`, `trial_id`, `segment`, `sample_idx`, `valence`, `arousal`, and EEG channels in this order: `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`. Stimulus sample indices must be contiguous and unique, with one rating per trial. Baseline rows are ignored.

## Outputs and preprocessing

| Output | Shape | Contents |
| --- | --- | --- |
| `datasets/dreamer_eeg.npy` | `(23, 18, 42, 7680)` | float32, subject × trial × channel-band × sample |
| `datasets/dreamer_labels.npy` | `(23, 18, 2)` | float32 raw ratings, `[valence, arousal]` |


## Reported configurations

Use [the reproduction guide](../../../../reproducibility/README.md), [v15 arousal](SICModelv15/README.md), and [v11 valence](SICModelv11/README.md).
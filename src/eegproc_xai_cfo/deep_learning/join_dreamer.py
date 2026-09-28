"""Create dreamer_joined.csv from the DREAMER EEG, ECG, and label CSVs.

Adapted from the supplied open_datasets/join_dreamer.py. This joins existing
CSV exports; it does not extract the original DREAMER.mat dataset.
"""

import argparse
from pathlib import Path

import pandas as pd


def join_dreamer(input_dir, output_dir):
    """Keep every EEG row, attach matching ECG samples, then trial labels."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    eeg = pd.read_csv(input_dir / "dreamer_eeg.csv")
    ecg = pd.read_csv(input_dir / "dreamer_ecg.csv")
    labels = pd.read_csv(input_dir / "dreamer_labels.csv")

    merged = eeg.merge(
        ecg,
        on=["subject_id", "trial_id", "segment", "sample_idx"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        labels,
        on=["subject_id", "trial_id"],
        how="left",
        validate="many_to_one",
    )
    if merged[["valence", "arousal"]].isna().any().any():
        raise ValueError("Every EEG trial must have valence and arousal labels")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "dreamer_joined.csv"
    merged.to_csv(output, index=False)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Folder containing dreamer_eeg.csv, dreamer_ecg.csv, and dreamer_labels.csv",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(f"Saved to: {join_dreamer(args.input_dir, args.output_dir)}")


if __name__ == "__main__":
    main()

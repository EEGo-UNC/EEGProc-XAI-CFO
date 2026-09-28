"""Replay the saved task-specific CFO protocol with local data/checkpoints."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def commands(args):
    here = Path(__file__).resolve().parent
    study = json.loads((here / args.task / "counterfactual_study_fold_00.json").read_text())
    values = study["arguments"].copy()
    provenance = json.loads((here / "provenance.json").read_text())
    config = args.config_dir or Path(provenance["tasks"][args.task]["training_run"]) / "configuration_0001"
    values.update(models_json=str(config / "loso_zero_shot_models.json"),
                  model_dir=str(config / "loso_zero_shot_models"))
    values["data_config"] = {**values["data_config"], "raw_eeg_npy": str(args.eeg),
                             "raw_labels_npy": str(args.labels)}
    for subject in args.subjects:
        command = [sys.executable, "-m", "eegproc_xai_cfo.model_explainability.typicality.runner"]
        for key, value in values.items():
            if key == "subjects" or value is None:
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif isinstance(value, dict):
                command += [flag, json.dumps(value)]
            elif isinstance(value, list):
                command += [flag, *map(str, value)]
            else:
                command += [flag, str(value)]
        command += ["--subjects", str(subject), "--out-dir", str(args.out_dir / f"fold_{subject:02d}")]
        if args.resume:
            command.append("--resume")
        yield command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("arousal", "valence"), required=True)
    parser.add_argument("--config-dir", type=Path, help="Matching training configuration_0001 directory")
    parser.add_argument("--eeg", type=Path, default=Path("datasets/dreamer_eeg.npy"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/dreamer_labels.npy"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", type=int, choices=range(23), default=list(range(23)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        parser.error("Run this command from the repository root so input and output paths are unambiguous")
    env = {**os.environ, "PYTHONPATH": str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "MPLBACKEND": "Agg"}
    for command in commands(args):
        print("PYTHONPATH=src MPLBACKEND=Agg " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()

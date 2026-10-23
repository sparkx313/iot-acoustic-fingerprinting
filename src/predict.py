"""Classify one WAV file or every WAV file under a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURE_NAMES, extract_features

MODEL_FORMAT_VERSION = 1


def collect_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".wav":
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            path for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        )
        if files:
            return files
    raise ValueError("Input must be a WAV file or a folder containing WAV recordings.")


def predict(args: argparse.Namespace) -> int:
    if args.output.resolve() == args.model.resolve():
        raise ValueError("Predictions output must not overwrite the model artifact.")
    if not args.model.is_file():
        raise ValueError(f"Model artifact does not exist: {args.model}")
    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported acoustic model artifact format.")
    if bundle.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model feature schema does not match this code version.")

    files = collect_audio_files(args.input)
    if args.output.resolve() in {path.resolve() for path in files}:
        raise ValueError("Predictions output must not overwrite an input WAV file.")
    rows = [extract_features(path) for path in files]
    features = pd.DataFrame(rows, columns=FEATURE_NAMES)
    model = bundle["pipeline"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    result = pd.DataFrame(
        {
            "file": [str(path) for path in files],
            "predicted_class": predictions,
        }
    )
    for index, label in enumerate(bundle["class_labels"]):
        result[f"probability_{label}"] = probabilities[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(files)} WAV recording(s); wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify local WAV recordings with a saved model.")
    parser.add_argument("--input", required=True, type=Path, help="WAV file or folder of WAV files.")
    parser.add_argument("--model", required=True, type=Path, help="Trusted artifact from src.train.")
    parser.add_argument("--output", type=Path, default=Path("acoustic_predictions.csv"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return predict(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

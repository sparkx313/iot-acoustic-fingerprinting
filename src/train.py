"""Train and evaluate a local classifier from labeled WAV folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_NAMES, extract_features

MODEL_FORMAT_VERSION = 1


def collect_training_files(root: Path) -> list[tuple[Path, str]]:
    if not root.is_dir():
        raise ValueError(f"Training directory does not exist: {root}")
    labeled_files = [
        (audio_file, class_dir.name)
        for class_dir in sorted(root.iterdir())
        if class_dir.is_dir()
        for audio_file in sorted(class_dir.rglob("*"))
        if audio_file.is_file() and audio_file.suffix.lower() == ".wav"
    ]
    if not labeled_files:
        raise ValueError("No WAV files found; put labeled recordings in one subfolder per class.")
    counts = pd.Series([label for _, label in labeled_files]).value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        raise ValueError("Training requires at least two WAV recordings in each of two or more class folders.")
    return labeled_files


def build_pipeline() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )


def train(args: argparse.Namespace) -> int:
    if args.model.resolve() == args.metrics.resolve():
        raise ValueError("Model and metrics output paths must differ.")
    labeled_files = collect_training_files(args.data_dir)
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    for audio_file, label in labeled_files:
        try:
            rows.append(extract_features(audio_file))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read training recording {audio_file}: {exc}") from exc
        labels.append(label)
    features = pd.DataFrame(rows, columns=FEATURE_NAMES)
    labels_array = np.asarray(labels)
    classes = sorted(set(labels))

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels_array,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels_array,
        )
    except ValueError as exc:
        raise ValueError(f"Could not create a stratified holdout: {exc}") from exc

    evaluation_model = build_pipeline()
    evaluation_model.fit(x_train, y_train)
    predicted = evaluation_model.predict(x_test)
    evaluation: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "class_labels": classes,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=classes).tolist(),
    }

    final_model = build_pipeline()
    final_model.fit(features, labels_array)
    bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "pipeline": final_model,
        "feature_names": FEATURE_NAMES,
        "class_labels": classes,
        "recordings_used": len(labeled_files),
    }
    report = {
        "recordings_total": len(labeled_files),
        "class_counts": {label: labels.count(label) for label in classes},
        "holdout_fraction": float(args.test_size),
        "seed": int(args.seed),
        "holdout_metrics": evaluation,
    }
    for path in (args.model, args.metrics):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    args.metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {len(y_test)} held-out recording(s); balanced accuracy: {evaluation['balanced_accuracy']:.4f}")
    print(f"Saved fitted model to {args.model}")
    print(f"Saved holdout report to {args.metrics}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an offline acoustic classifier from labeled WAV recordings."
    )
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder with one class subfolder per label.")
    parser.add_argument("--model", type=Path, default=Path("acoustic_model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("acoustic_metrics.json"))
    parser.add_argument("--test-size", type=float, default=0.2, help="Stratified holdout fraction (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return train(args)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

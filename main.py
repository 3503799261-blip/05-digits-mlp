"""Train a tiny neural network on the offline scikit-learn digits dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
PROJECT = Path(__file__).resolve().parent.name


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--sample-index", type=int, default=0, help="Index inside the held-out test set")
    args = parser.parse_args()
    data = load_digits()
    train_ids, test_ids = train_test_split(
        np.arange(len(data.target)), test_size=0.25, stratify=data.target, random_state=SEED
    )
    if not 0 <= args.sample_index < len(test_ids):
        parser.error(f"--sample-index must be in [0, {len(test_ids) - 1}]")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    x_train, x_test = data.data[train_ids], data.data[test_ids]
    y_train, y_test = data.target[train_ids], data.target[test_ids]
    models = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=SEED)),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,), activation="relu", alpha=0.001,
                max_iter=300, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=SEED,
            ),
        ),
    }
    metrics = {}
    predictions = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions[name] = model.predict(x_test)
        metrics[name] = scores(y_test, predictions[name])
    pred = predictions["mlp"]
    network = models["mlp"].named_steps["mlpclassifier"]
    report = {
        "project": PROJECT, "seed": SEED,
        "data_source": "sklearn.datasets.load_digits; 1797 offline 8x8 handwritten-digit images; UCI Optical Recognition of Handwritten Digits",
        "split": {"train": len(train_ids), "test": len(test_ids), "stratified": True},
        "metrics": metrics,
        "training": {"epochs": int(network.n_iter_), "hidden_units": 64, "early_stopping": True},
        "limitations": "Random image split is not a writer-independent split; test labels are used only for final evaluation and error analysis.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_sample_id", "true_label", "mlp_prediction", "logistic_prediction"])
        writer.writerows(zip(test_ids, y_test, pred, predictions["logistic_regression"]))
    fig, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=axis, colorbar=False, cmap="Blues")
    axis.set_title("Digits MLP: held-out confusion matrix")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    failures = np.flatnonzero(pred != y_test)[:12]
    fig, axes = plt.subplots(3, 4, figsize=(9, 6))
    for axis in axes.flat:
        axis.axis("off")
    if not len(failures):
        axes.flat[0].text(0.0, 0.5, "No errors on this test split")
    for axis, idx in zip(axes.flat, failures):
        axis.imshow(x_test[idx].reshape(8, 8), cmap="gray_r")
        axis.set_title(f"true={y_test[idx]}, pred={pred[idx]}")
    fig.suptitle("Misclassified test images")
    fig.tight_layout()
    fig.savefig(output / "error_examples.png", dpi=150)
    plt.close(fig)
    idx = args.sample_index
    probabilities = models["mlp"].predict_proba(x_test[idx:idx + 1])[0]
    example = {
        "test_index": idx, "original_sample_id": int(test_ids[idx]), "true_label": int(y_test[idx]),
        "prediction": int(pred[idx]), "class_probabilities": probabilities.tolist(),
        "note": "Neural-network probabilities are not separately calibrated.",
    }
    (output / "example_prediction.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(3, 3))
    axis.imshow(x_test[idx].reshape(8, 8), cmap="gray_r")
    axis.set_title(f"True: {y_test[idx]} / MLP: {pred[idx]}")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "example_prediction.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()

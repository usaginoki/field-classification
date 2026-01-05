"""Analyze misclassified examples from Logistic Regression model."""

import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.dataset import load_image
from src.feature_extraction import FeatureCache


def main():
    # Configuration (must match main.py run)
    cache_dir = Path("cache/features")
    model_path = Path("output/LogisticRegression/best_model.joblib")
    output_path = Path("output/LogisticRegression/misclassified.csv")

    # Same parameters as main.py
    holdout_fraction = 0.2
    random_seed = 42

    # Load cached features
    cache = FeatureCache(cache_dir)
    # Cache key for full dataset with vision_encoder mode
    cache_key = cache.get_cache_key(
        "Qwen/Qwen3-VL-8B-Instruct",
        "mean",
        "full_vision_encoder",
    )

    print(f"Loading features from cache key: {cache_key}")
    features, labels, filenames, metadata = cache.load(cache_key)
    print(f"Loaded {len(labels)} samples with {features.shape[1]} features")

    # Recreate the same train/test split
    X_train, X_test, y_train, y_test, _, test_filenames = train_test_split(
        features,
        labels,
        filenames,
        test_size=holdout_fraction,
        random_state=random_seed,
        stratify=labels,
    )
    print(f"Test set: {len(y_test)} samples")

    # Load the saved model
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Find misclassified examples
    misclassified_mask = y_pred != y_test
    n_errors = misclassified_mask.sum()
    print(f"Found {n_errors} misclassified examples out of {len(y_test)} ({100*n_errors/len(y_test):.2f}%)")

    # Prepare output data
    rows = []
    for i in range(len(y_test)):
        if misclassified_mask[i]:
            rows.append({
                "filename": test_filenames[i],
                "true_class": int(y_test[i]),
                "predicted_class": int(y_pred[i]),
                "prob_class_0": float(y_proba[i, 0]),
                "prob_class_1": float(y_proba[i, 1]),
            })

    # Sort by confidence (how wrong the model was)
    # Higher probability for wrong class = more confident mistake
    rows.sort(key=lambda r: r[f"prob_class_{r['predicted_class']}"], reverse=True)

    # Save to CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_class", "predicted_class", "prob_class_0", "prob_class_1"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved misclassified examples to: {output_path}")

    # Print summary
    print("\nMost confident mistakes:")
    for row in rows[:5]:
        print(f"  {Path(row['filename']).name}: true={row['true_class']}, pred={row['predicted_class']}, "
              f"P(0)={row['prob_class_0']:.3f}, P(1)={row['prob_class_1']:.3f}")

    # Convert misclassified images to PNG
    png_dir = Path("output/LogisticRegression/misclassified_images")
    png_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting {len(rows)} misclassified images to PNG...")
    for row in rows:
        src_path = Path(row["filename"])
        # Include prediction info in filename
        png_name = f"true{row['true_class']}_pred{row['predicted_class']}_{src_path.stem}.png"
        dst_path = png_dir / png_name

        img = load_image(src_path)
        img.save(dst_path, "PNG")

    print(f"Saved PNG images to: {png_dir}")


if __name__ == "__main__":
    main()

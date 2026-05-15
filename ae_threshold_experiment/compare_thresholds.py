import argparse
import numpy as np


def load_errors(path):
    return np.load(path)


def compute_metrics(known_errors, novel_errors, threshold):
    # error > threshold ise "novel" diyoruz
    known_pred_novel = known_errors > threshold
    novel_pred_novel = novel_errors > threshold

    false_positive = known_pred_novel.mean()
    false_negative = (~novel_pred_novel).mean()

    known_correct = (~known_pred_novel).mean()
    novel_correct = novel_pred_novel.mean()
    accuracy = (known_correct + novel_correct) / 2

    return {
        "threshold": threshold,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": accuracy,
        "known_correct": known_correct,
        "novel_correct": novel_correct,
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Threshold:       {metrics['threshold']:.6f}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"False Positive:  {metrics['false_positive']:.4f}")
    print(f"False Negative:  {metrics['false_negative']:.4f}")
    print(f"Known Correct:   {metrics['known_correct']:.4f}")
    print(f"Novel Correct:   {metrics['novel_correct']:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--known",
        type=str,
        default="ae_threshold_experiment/results/errors_empty.npy",
    )

    parser.add_argument(
        "--novel",
        type=str,
        nargs="+",
        default=[
            "ae_threshold_experiment/results/errors_empty_random6.npy",
            "ae_threshold_experiment/results/errors_empty16.npy",
            "ae_threshold_experiment/results/errors_fourrooms.npy",
            "ae_threshold_experiment/results/errors_doorkey.npy",
        ],
    )

    args = parser.parse_args()

    known_errors = load_errors(args.known)
    novel_errors = np.concatenate([load_errors(path) for path in args.novel])

    print("Loaded errors")
    print(f"Known samples: {len(known_errors)}")
    print(f"Novel samples: {len(novel_errors)}")

    print("\nError summary")
    print("-------------")
    print(f"Known mean: {known_errors.mean():.6f}")
    print(f"Known std:  {known_errors.std():.6f}")
    print(f"Known max:  {known_errors.max():.6f}")
    print(f"Novel mean: {novel_errors.mean():.6f}")
    print(f"Novel std:  {novel_errors.std():.6f}")
    print(f"Novel min:  {novel_errors.min():.6f}")

    # Method 1: fixed threshold examples
    fixed_thresholds = [0.002, 0.005, 0.01, 0.02]

    for t in fixed_thresholds:
        metrics = compute_metrics(known_errors, novel_errors, t)
        print_metrics(f"Fixed threshold = {t}", metrics)

    # Method 2: mean + k * std
    for k in [1, 2, 3]:
        threshold = known_errors.mean() + k * known_errors.std()
        metrics = compute_metrics(known_errors, novel_errors, threshold)
        print_metrics(f"Mean + {k} * Std", metrics)

    # Method 3: percentile threshold
    for p in [90, 95, 99]:
        threshold = np.percentile(known_errors, p)
        metrics = compute_metrics(known_errors, novel_errors, threshold)
        print_metrics(f"{p}th percentile", metrics)


if __name__ == "__main__":
    main()
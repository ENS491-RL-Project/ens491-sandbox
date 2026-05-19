import argparse
import numpy as np


def load_errors(path):
    return np.load(path)


def simulate_ae_only_column_opening(
    known_errors,
    task_error_files,
    threshold,
    min_consecutive_steps=1,
):
    current_column = 0
    opened_columns = 0
    results = []

    # 1) Known task içinde false positive ölç
    consecutive_novel = 0
    false_positive_openings = 0

    for error in known_errors:
        if error > threshold:
            consecutive_novel += 1
        else:
            consecutive_novel = 0

        if consecutive_novel >= min_consecutive_steps:
            false_positive_openings += 1
            consecutive_novel = 0

    # 2) Novel tasklerde yeni column açılıyor mu ölç
    for task_name, error_path in task_error_files:
        errors = load_errors(error_path)

        consecutive_novel = 0
        detected = False
        detection_step = None

        for step, error in enumerate(errors):
            if error > threshold:
                consecutive_novel += 1
            else:
                consecutive_novel = 0

            if consecutive_novel >= min_consecutive_steps:
                detected = True
                detection_step = step
                opened_columns += 1
                current_column += 1
                break

        results.append(
            {
                "task": task_name,
                "detected": detected,
                "detection_step": detection_step,
            }
        )

    return {
        "threshold": threshold,
        "min_consecutive_steps": min_consecutive_steps,
        "false_positive_openings": false_positive_openings,
        "opened_columns": opened_columns,
        "task_results": results,
    }


def print_results(results):
    print("\nAE-only Column Opening Simulation")
    print("---------------------------------")
    print(f"Threshold: {results['threshold']:.6f}")
    print(f"Min consecutive steps: {results['min_consecutive_steps']}")
    print(f"False positive openings on known task: {results['false_positive_openings']}")
    print(f"Opened columns on novel tasks: {results['opened_columns']}")

    print("\nPer-task detection")
    print("------------------")

    for item in results["task_results"]:
        if item["detected"]:
            print(
                f"{item['task']}: detected at step {item['detection_step']}"
            )
        else:
            print(f"{item['task']}: NOT detected")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--known",
        type=str,
        default="ae_threshold_experiment/results/errors_sparse_empty.npy",
    )

    parser.add_argument("--threshold", type=float, default=0.004384)

    parser.add_argument("--min-consecutive-steps", type=int, default=5)

    args = parser.parse_args()

    known_errors = load_errors(args.known)

    task_error_files = [
        (
            "Empty-Random-6x6",
            "ae_threshold_experiment/results/errors_sparse_empty_random6.npy",
        ),
        (
            "Empty-16x16",
            "ae_threshold_experiment/results/errors_sparse_empty16.npy",
        ),
        (
            "FourRooms",
            "ae_threshold_experiment/results/errors_sparse_fourrooms.npy",
        ),
        (
            "DoorKey",
            "ae_threshold_experiment/results/errors_sparse_doorkey.npy",
        ),
    ]

    results = simulate_ae_only_column_opening(
        known_errors=known_errors,
        task_error_files=task_error_files,
        threshold=args.threshold,
        min_consecutive_steps=args.min_consecutive_steps,
    )

    print_results(results)


if __name__ == "__main__":
    main()
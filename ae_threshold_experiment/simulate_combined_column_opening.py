import argparse
import numpy as np


# =========================================================
# AE SIGNAL
# =========================================================

def ae_detect_novelty(
    errors,
    threshold,
    min_consecutive_steps,
):
    consecutive = 0

    for step, error in enumerate(errors):
        if error > threshold:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= min_consecutive_steps:
            return True, step

    return False, None


# =========================================================
# REWARD PLATEAU SIGNAL
# =========================================================

def reward_plateau_detected(
    rewards,
    window_size,
    min_improvement,
    patience,
):
    best_avg = -np.inf
    bad_windows = 0

    for start in range(0, len(rewards) - window_size + 1, window_size):
        end = start + window_size

        window_avg = rewards[start:end].mean()
        improvement = window_avg - best_avg

        if improvement > min_improvement:
            best_avg = window_avg
            bad_windows = 0
        else:
            bad_windows += 1

        if bad_windows >= patience:
            return True, end

    return False, None


# =========================================================
# SYNTHETIC REWARDS
# =========================================================

def generate_rewards(mode, length=200):
    rng = np.random.default_rng(42)

    if mode == "improving":
        rewards = np.linspace(0.1, 1.0, length)
        rewards += rng.normal(0, 0.03, length)

    elif mode == "plateau":
        rewards = np.ones(length) * 0.2
        rewards += rng.normal(0, 0.02, length)

    elif mode == "delayed":
        first = np.ones(length // 2) * 0.2
        second = np.linspace(0.2, 0.9, length - length // 2)
        rewards = np.concatenate([first, second])
        rewards += rng.normal(0, 0.02, length)

    elif mode == "noisy":
        rewards = np.ones(length) * 0.5
        rewards += rng.normal(0, 0.08, length)

    else:
        raise ValueError(f"Unknown reward mode: {mode}")

    return np.clip(rewards, 0.0, 1.0)


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--logic",
        type=str,
        default="and",
        choices=["and", "or"],
    )

    parser.add_argument(
        "--errors",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--reward-mode",
        type=str,
        default="plateau",
        choices=["improving", "plateau", "delayed", "noisy"],
    )

    # Optimized AE parameters
    parser.add_argument("--threshold", type=float, default=0.004384)
    parser.add_argument("--min-consecutive-steps", type=int, default=5)

    # Optimized reward parameters
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    # -----------------------------------------------------

    errors = np.load(args.errors)

    ae_detected, ae_step = ae_detect_novelty(
        errors=errors,
        threshold=args.threshold,
        min_consecutive_steps=args.min_consecutive_steps,
    )

    rewards = generate_rewards(args.reward_mode)

    reward_detected, reward_step = reward_plateau_detected(
        rewards=rewards,
        window_size=args.window_size,
        min_improvement=args.min_improvement,
        patience=args.patience,
    )

    # -----------------------------------------------------

    if args.logic == "and":
        open_column = ae_detected and reward_detected

    elif args.logic == "or":
        open_column = ae_detected or reward_detected

    else:
        raise ValueError("Unknown logic")

    # -----------------------------------------------------

    print("\nCombined Column Opening Simulation")
    print("----------------------------------")

    print(f"Logic: {args.logic.upper()}")
    print(f"Reward mode: {args.reward_mode}")

    print("\nAE signal")
    print("---------")
    print(f"Detected: {ae_detected}")
    print(f"Detection step: {ae_step}")

    print("\nReward signal")
    print("-------------")
    print(f"Detected: {reward_detected}")
    print(f"Detection step: {reward_step}")

    print("\nFINAL DECISION")
    print("--------------")

    if open_column:
        print("OPEN NEW COLUMN")
    else:
        print("DO NOT OPEN NEW COLUMN")


if __name__ == "__main__":
    main()
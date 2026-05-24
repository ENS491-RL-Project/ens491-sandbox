import argparse
import numpy as np


def generate_synthetic_rewards(mode, length):
    """
    Synthetic reward curves for testing reward-plateau column opening.

    improving:
        Reward steadily increases.
    plateau:
        Reward stays almost flat.
    delayed:
        Reward is flat at first, then improves.
    noisy:
        Reward fluctuates around same level.
    """
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
        rewards += rng.normal(0, 0.025, length)

    elif mode == "noisy":
        rewards = np.ones(length) * 0.5
        rewards += rng.normal(0, 0.08, length)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return np.clip(rewards, 0.0, 1.0)


def detect_reward_plateau(
    rewards,
    window_size=20,
    min_improvement=0.02,
    patience=3,
):
    """
    Detect plateau using moving-window reward averages.

    Logic:
    - Split reward curve into windows.
    - Compare current window average to best previous average.
    - If improvement is smaller than min_improvement for 'patience'
      consecutive windows, declare plateau.
    """
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="plateau",
        choices=["improving", "plateau", "delayed", "noisy"],
    )

    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        modes = ["plateau", "improving", "delayed", "noisy"]
        window_sizes = [10, 20, 40]
        min_improvements = [0.01, 0.02, 0.05]
        patiences = [2, 3, 5]

        print("\nReward Plateau Parameter Sweep")
        print("-------------------------------")

        for window_size in window_sizes:
            for min_improvement in min_improvements:
                for patience in patiences:
                    print(
                        f"\nwindow={window_size}, "
                        f"min_improvement={min_improvement}, "
                        f"patience={patience}"
                    )

                    for mode in modes:
                        rewards = generate_synthetic_rewards(mode, args.length)

                        detected, step = detect_reward_plateau(
                            rewards=rewards,
                            window_size=window_size,
                            min_improvement=min_improvement,
                            patience=patience,
                        )

                        decision = "OPEN" if detected else "NO_OPEN"
                        print(f"  {mode:10s} -> {decision:7s} step={step}")

        return

    rewards = generate_synthetic_rewards(args.mode, args.length)

    plateau_detected, detection_step = detect_reward_plateau(
        rewards=rewards,
        window_size=args.window_size,
        min_improvement=args.min_improvement,
        patience=args.patience,
    )

    print("\nReward-Plateau Column Opening Simulation")
    print("-----------------------------------------")
    print(f"Reward mode: {args.mode}")
    print(f"Window size: {args.window_size}")
    print(f"Min improvement: {args.min_improvement}")
    print(f"Patience: {args.patience}")

    if plateau_detected:
        print(f"Decision: OPEN NEW COLUMN")
        print(f"Plateau detected at step: {detection_step}")
    else:
        print("Decision: DO NOT open new column")
        print("Plateau not detected")

    print("\nReward summary")
    print("--------------")
    print(f"Mean reward: {rewards.mean():.4f}")
    print(f"Final reward: {rewards[-1]:.4f}")
    print(f"Max reward: {rewards.max():.4f}")


if __name__ == "__main__":
    main()
"""
Pearls AQI Predictor — CLI Entry Point

Usage:
    python main.py --pipeline feature      # Run feature pipeline once
    python main.py --pipeline backfill     # Run historical backfill
    python main.py --pipeline train        # Run training pipeline
    python main.py --pipeline predict      # Run inference (next 3 days)
    python main.py --check-config          # Validate environment variables
"""

import argparse
import sys
import os

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Pearls AQI Predictor — End-to-End ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pipeline",
        choices=["feature", "backfill", "train", "predict"],
        help="Which pipeline to run",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate that all required environment variables are set",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Override the city from .env (e.g. --city lahore)",
    )

    args = parser.parse_args()

    # ── Config check ──────────────────────────────────────────────────────────
    if args.check_config or args.pipeline is None:
        from config import validate_config, CITY_CONFIG
        print("=== Pearls AQI Predictor ===")
        print(f"  Target city: {CITY_CONFIG['name']}, {CITY_CONFIG['country']}")
        try:
            validate_config()
            print("  ✅ Configuration looks good!")
        except ValueError as e:
            print(f"  ❌ {e}")
            sys.exit(1)
        if args.pipeline is None:
            parser.print_help()
        return

    # ── Pipeline dispatch ─────────────────────────────────────────────────────
    if args.pipeline == "feature":
        print("Running feature pipeline...")
        from feature_pipeline import run as run_feature
        run_feature()

    elif args.pipeline == "backfill":
        print("Running historical backfill...")
        from backfill import run as run_backfill
        run_backfill()

    elif args.pipeline == "train":
        print("Running training pipeline...")
        from training_pipeline import run as run_training
        run_training()

    elif args.pipeline == "predict":
        print("Running inference (next 3-day forecast)...")
        from inference import run as run_inference
        run_inference()


if __name__ == "__main__":
    main()
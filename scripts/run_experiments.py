"""
Experiment runner: trains both scenarios for each seed and records results.

Modes
-----
single  — One pair (seed 1001), full epochs.  For smoke-testing the pipeline.
pilot   — N pairs (default 3), reduced epochs.  For quick sanity-checking.
full    — All 30 pairs, full epochs.  The actual scientific campaign.

Each campaign is identified by a run_id (default: auto-generated timestamp).
Results are appended to results/metrics.csv; completed (run_id, scenario,
seed) triples are skipped automatically, enabling resumption.

A manifest.json is saved to results/campaigns/<run_id>/ with configuration,
seeds, data hash, device info, and TF/Python versions.

Usage
-----
# Smoke test (one pair):
python scripts/run_experiments.py --mode single

# Pilot (3 pairs, 5 epochs max):
python scripts/run_experiments.py --mode pilot

# Full campaign (30 pairs) — generates all 60 trainings:
python scripts/run_experiments.py --mode full

# Resume a named campaign:
python scripts/run_experiments.py --mode full --run-id campaign_20241115_143022

Note: the full campaign (60 trainings) is not run automatically during
      implementation.  Run the command above explicitly when ready.
"""

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_experiments")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_run_id() -> str:
    return "campaign_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_manifest(run_id: str, mode: str, config: dict, data_hash: str) -> dict:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "run_id":        run_id,
        "mode":          mode,
        "start_time":    datetime.now(timezone.utc).isoformat(),
        "end_time":      None,
        "config":        config,
        "data_hash":     data_hash,
        "tf_version":    tf.__version__,
        "python_version": platform.python_version(),
        "platform":      platform.platform(),
        "gpu_available": len(gpus) > 0,
        "gpu_devices":   [d.name for d in gpus],
        "determinism_enabled": None,  # updated after attempt
    }


def save_manifest(manifest: dict, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)


# ──────────────────────────────────────────────────────────────────────
# Data loading and preprocessing (once per campaign)
# ──────────────────────────────────────────────────────────────────────

def load_and_prepare_data(config: dict):
    """Load raw data, deduplicate, split, preprocess.

    Saves:
      - data/processed/split_ids.json   (partition indices)
      - data/processed/tokenizer.pkl    (fitted on train only)

    Returns:
      X_train, y_train, X_val, y_val, X_test, y_test, test_ids, data_hash
    """
    from src.data import (
        apply_split_ids,
        check_invalid_labels,
        check_missing_values,
        compute_file_hash,
        handle_duplicates,
        load_raw_data,
        load_split_ids,
        save_split_ids,
        split_data,
    )
    from src.preprocessing import prepare_splits, save_tokenizer

    raw_path = REPO_ROOT / "data" / "raw" / "SMSSpamCollection"
    processed_dir = REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    data_hash = "sha256:" + compute_file_hash(raw_path)
    logger.info("Data hash: %s", data_hash)

    df = load_raw_data(raw_path)

    # Integrity checks
    check_missing_values(df)
    invalid = check_invalid_labels(df)
    if not invalid.empty:
        raise ValueError(f"Found {len(invalid)} rows with invalid labels — aborting.")

    df_clean, dup_report = handle_duplicates(df)
    logger.info("Duplicate report: %s", dup_report)

    # Split (or reload if already saved)
    split_ids_path = processed_dir / "split_ids.json"
    if split_ids_path.exists():
        logger.info("Reloading existing split IDs from %s", split_ids_path)
        split_ids = load_split_ids(split_ids_path)
        df_train, df_val, df_test = apply_split_ids(df_clean, split_ids)
    else:
        df_train, df_val, df_test = split_data(
            df_clean,
            train_ratio=config["split"]["train_ratio"],
            val_ratio=config["split"]["val_ratio"],
            test_ratio=config["split"]["test_ratio"],
            seed=config["split"]["seed"],
        )
        save_split_ids(df_train, df_val, df_test, processed_dir)

    # Preprocess (tokeniser from train only)
    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer = prepare_splits(
        df_train, df_val, df_test,
        max_vocab_size=config["preprocessing"]["max_vocab_size"],
        max_length=config["preprocessing"]["max_length"],
    )

    tokenizer_path = processed_dir / "tokenizer.pkl"
    save_tokenizer(tokenizer, tokenizer_path)

    test_ids = df_test["id"].to_numpy()

    logger.info(
        "Data ready — train: %d, val: %d, test: %d", len(y_train), len(y_val), len(y_test)
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, test_ids, data_hash


# ──────────────────────────────────────────────────────────────────────
# Single pair training
# ──────────────────────────────────────────────────────────────────────

def run_pair(
    seed: int,
    run_id: str,
    config: dict,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    test_ids,
    max_epochs_override: int | None = None,
):
    """Train and evaluate one pair of models for the given seed.

    Skips individual (scenario, seed) combinations that already exist in the
    metrics CSV so the campaign can be safely resumed.
    """
    from src.evaluate import (
        append_metrics_row,
        is_run_completed,
        save_predictions,
        save_training_history,
    )
    from src.train import (
        create_paired_models,
        enable_determinism,
        preshuffle_training_data,
        train_model,
    )

    metrics_csv = REPO_ROOT / "results" / "metrics.csv"
    pred_dir    = REPO_ROOT / "results" / "predictions" / run_id
    hist_dir    = REPO_ROOT / "results" / "campaigns" / run_id / "histories"
    ckpt_dir    = REPO_ROOT / "checkpoints" / run_id

    # Apply epoch override (pilot mode)
    effective_config = config.copy()
    if max_epochs_override is not None:
        effective_config["training"] = dict(config["training"])
        effective_config["training"]["max_epochs"] = max_epochs_override

    # Pre-shuffle once: both models use the same batch ordering
    X_tr_sh, y_tr_sh = preshuffle_training_data(X_train, y_train, seed)

    # Build paired models with identical shared weights
    logger.info("Building paired models for seed %d …", seed)
    model_add, model_sdp = create_paired_models(effective_config, seed)

    scenarios = [
        ("additive",          model_add),
        ("scaled_dot_product", model_sdp),
    ]

    for scenario, model in scenarios:
        if is_run_completed(metrics_csv, run_id, scenario, seed):
            logger.info("Already completed: run=%s  scenario=%s  seed=%d — skipping.", run_id, scenario, seed)
            continue

        logger.info("Training  run=%s  scenario=%-20s  seed=%d", run_id, scenario, seed)

        ckpt_path = ckpt_dir / f"{scenario}_seed{seed}" / "weights.weights.h5"

        history, train_time, best_epoch, total_epochs = train_model(
            model,
            X_tr_sh, y_tr_sh,
            X_val,   y_val,
            effective_config,
            pair_seed=seed,
            checkpoint_path=ckpt_path,
            verbose=0,
        )

        # Evaluate on test set (final evaluation — no feedback loop)
        from src.evaluate import evaluate_model
        metrics = evaluate_model(
            model, X_test, y_test,
            threshold=effective_config["evaluation"]["threshold"],
            batch_size=effective_config["training"]["batch_size"],
        )

        # Persist results
        append_metrics_row(
            metrics_csv, run_id, scenario, seed,
            metrics, train_time, best_epoch, total_epochs,
        )
        save_predictions(
            pred_dir / f"{scenario}_seed{seed}.csv",
            message_ids=test_ids,
            y_true=y_test,
            probabilities=metrics["probabilities"],
            predictions=metrics["predictions"],
        )
        save_training_history(
            hist_dir / f"{scenario}_seed{seed}.json",
            history,
        )

    # Free memory
    import tensorflow as tf
    tf.keras.backend.clear_session()


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run spam-detection experiments.")
    parser.add_argument(
        "--mode",
        choices=["single", "pilot", "full"],
        default="single",
        help="single=one pair; pilot=few pairs/epochs; full=30 pairs (default: single).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Reuse an existing run ID to resume a campaign.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "experiment.yaml"),
        help="Path to experiment.yaml.",
    )
    parser.add_argument(
        "--pilot-pairs",
        type=int,
        default=None,
        help="Number of pairs in pilot mode (overrides config).",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))

    run_id = args.run_id or make_run_id()
    run_dir = REPO_ROOT / "results" / "campaigns" / run_id
    logger.info("Run ID: %s", run_id)

    # Determine seeds and epoch limit from mode
    all_seeds = config["training"]["seeds"]
    max_epochs_override = None

    if args.mode == "single":
        seeds = all_seeds[:1]
        logger.info("Mode: single (1 pair, seed %d)", seeds[0])
    elif args.mode == "pilot":
        n_pairs = args.pilot_pairs or config["pilot"]["n_pairs"]
        max_epochs_override = config["pilot"]["max_epochs"]
        seeds = all_seeds[:n_pairs]
        logger.info("Mode: pilot (%d pairs, max %d epochs)", n_pairs, max_epochs_override)
    else:
        seeds = all_seeds
        logger.info("Mode: full (%d pairs)", len(seeds))

    # Attempt to enable TF determinism
    from src.train import enable_determinism
    det_ok = enable_determinism()

    # Load and prepare data (once)
    X_train, y_train, X_val, y_val, X_test, y_test, test_ids, data_hash = (
        load_and_prepare_data(config)
    )

    # Build and save manifest
    manifest = build_manifest(run_id, args.mode, config, data_hash)
    manifest["determinism_enabled"] = det_ok
    manifest["seeds_this_run"] = seeds
    save_manifest(manifest, run_dir)

    # Run experiments
    logger.info("Starting %d pairs …", len(seeds))
    for i, seed in enumerate(seeds, 1):
        logger.info("--- Pair %d / %d  seed=%d ---", i, len(seeds), seed)
        run_pair(
            seed=seed,
            run_id=run_id,
            config=config,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
            test_ids=test_ids,
            max_epochs_override=max_epochs_override,
        )

    # Update manifest with end time
    manifest["end_time"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest, run_dir)

    logger.info("Campaign '%s' complete.  Results → results/metrics.csv", run_id)


if __name__ == "__main__":
    main()

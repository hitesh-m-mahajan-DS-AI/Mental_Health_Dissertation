"""Single user-facing entry point for independent motivation and empathy probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis_config import load_analysis_methodology
from src.probe import (
    AGGREGATIONS,
    DEPTH_KEYS,
    evaluate_all_layers,
    load_capture_records,
    load_residual_matrix,
    make_grouped_split,
    prepare_layer_designs,
    run_permutations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent grouped residual-stream probes for motivation and empathy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis_methodology.json"),
        help="Analysis methodology JSON (default: configs/analysis_methodology.json)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate scientific settings without reading activation tensors.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel permutation workers (default: 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methodology = load_analysis_methodology(args.config)
    methodology.require_resolved("linear_probing")

    if args.validate_only:
        print(
            "Linear-probing methodology is complete for activation run "
            f"{methodology.activation_run_id}."
        )
        return

    root = Path(__file__).resolve().parent
    probing = methodology.raw["linear_probing"]
    split_config = probing["grouped_split_algorithm"]
    unresolved = probing["unresolved_required"]
    c_grid = tuple(float(value) for value in unresolved["logistic_regression_C_or_selection_grid"])
    max_iter = int(unresolved["solver_and_max_iterations"]["max_iter"])
    split_seed = int(unresolved["grouped_split_seed"])
    permutations = int(unresolved["label_permutation_count"])
    run_id = methodology.activation_run_id

    output_root = root / "results" / "probes"
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset_index, dataset in enumerate(methodology.raw["tasks"]):
        records = load_capture_records(root, dataset, run_id)
        split = make_grouped_split(
            records.labels,
            records.groups,
            ratios=(0.70, 0.15, 0.15),
            seed=split_seed,
            candidate_assignments=int(split_config["candidate_assignments"]),
        )
        dataset_root = output_root / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / f"split_{run_id}.json").write_text(
            json.dumps(split.metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"{dataset}: split {split.metadata['splits']}", flush=True)

        for aggregation_index, aggregation in enumerate(AGGREGATIONS):
            print(f"  {aggregation}: loading 33 residual depths", flush=True)
            matrix = load_residual_matrix(records, aggregation)
            designs = prepare_layer_designs(matrix, split)
            del matrix
            observed = evaluate_all_layers(
                designs,
                records.labels,
                split,
                c_grid=c_grid,
                max_iter=max_iter,
                random_state=split_seed,
            )
            permutation_seed = split_seed + 1000 * dataset_index + 100 * aggregation_index
            checkpoint = dataset_root / f"{aggregation}_permutations_{run_id}.npz"
            null_results = run_permutations(
                designs,
                records.labels,
                split,
                count=permutations,
                seed=permutation_seed,
                c_grid=c_grid,
                max_iter=max_iter,
                workers=args.workers,
                checkpoint_path=checkpoint,
            )

            observed_macro = observed[:, 4]
            null_macro = null_results[:, :, 4]
            observed_delta = observed_macro - observed_macro[0]
            null_delta = null_macro - null_macro[:, [0]]
            p_shuffled = (1 + (null_macro >= observed_macro).sum(axis=0)) / (permutations + 1)
            p_layer0_delta = (1 + (null_delta >= observed_delta).sum(axis=0)) / (
                permutations + 1
            )
            p_max_stat = (1 + (null_macro.max(axis=1)[:, None] >= observed_macro).sum(axis=0)) / (
                permutations + 1
            )
            baseline_count = int(unresolved["shuffled_label_repetitions"])

            layers = []
            for depth, key in enumerate(DEPTH_KEYS):
                baseline = null_results[:baseline_count, depth]
                layers.append(
                    {
                        "depth": depth,
                        "activation_key": key,
                        "train_rank": designs[depth].train_rank,
                        "development_rank": designs[depth].development_rank,
                        "selected_C": float(observed[depth, 0]),
                        "validation": {
                            "accuracy": float(observed[depth, 1]),
                            "macro_f1": float(observed[depth, 2]),
                        },
                        "test": {
                            "accuracy": float(observed[depth, 3]),
                            "macro_f1": float(observed[depth, 4]),
                        },
                        "shuffled_label_baseline_100": {
                            "accuracy_mean": float(baseline[:, 3].mean()),
                            "macro_f1_mean": float(baseline[:, 4].mean()),
                            "macro_f1_2_5_percentile": float(
                                np.quantile(baseline[:, 4], 0.025)
                            ),
                            "macro_f1_97_5_percentile": float(
                                np.quantile(baseline[:, 4], 0.975)
                            ),
                        },
                        "permutation_p_values": {
                            "shuffled_score": float(p_shuffled[depth]),
                            "improvement_over_layer_0": float(p_layer0_delta[depth]),
                            "max_statistic_across_depths": float(p_max_stat[depth]),
                        },
                    }
                )

            payload = {
                "activation_run_id": run_id,
                "dataset": dataset,
                "aggregation": aggregation,
                "examples": len(records.example_ids),
                "truncated_examples": int(records.truncated.sum()),
                "split": split.metadata,
                "methodology": probing,
                "permutation_seed": permutation_seed,
                "permutations": permutations,
                "layers": layers,
            }
            result_path = dataset_root / f"{aggregation}_{run_id}.json"
            result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()

"""Independent grouped linear probing over captured residual-stream activations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from joblib import Parallel, delayed
from safetensors import safe_open
from scipy.linalg import qr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


DEPTH_KEYS = ("layer_00.residual_pre",) + tuple(
    f"layer_{layer:02d}.residual_post" for layer in range(32)
)
AGGREGATIONS = ("final_valid_response_token", "mean_response_token_activation")


@dataclass(frozen=True)
class CaptureRecords:
    dataset: str
    example_ids: tuple[str, ...]
    labels: np.ndarray
    groups: np.ndarray
    residual_paths: tuple[Path, ...]
    truncated: np.ndarray


@dataclass(frozen=True)
class GroupedSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LayerDesign:
    train: np.ndarray
    validation: np.ndarray
    development: np.ndarray
    test: np.ndarray
    train_rank: int
    development_rank: int


def load_capture_records(project_root: Path, dataset: str, run_id: str) -> CaptureRecords:
    metadata_path = (
        project_root
        / "results"
        / "activations"
        / dataset
        / "metadata"
        / f"{run_id}_examples.jsonl"
    )
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 200:
        raise ValueError(f"Expected 200 {dataset} capture records, found {len(records)}")

    paths: list[Path] = []
    for record in records:
        residual = [
            item for item in record["output_files"] if item["activation_group"] == "residual_stream"
        ]
        if len(residual) != 1:
            raise ValueError(f"Expected one residual file for {record['example_id']}")
        path = Path(residual[0]["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)

    return CaptureRecords(
        dataset=dataset,
        example_ids=tuple(str(record["example_id"]) for record in records),
        labels=np.asarray([int(record["label"]) for record in records], dtype=np.int8),
        groups=np.asarray([str(record["conversation_id"]) for record in records], dtype=object),
        residual_paths=tuple(paths),
        truncated=np.asarray([bool(record["was_truncated"]) for record in records]),
    )


def load_residual_matrix(records: CaptureRecords, aggregation: str) -> np.ndarray:
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    matrix = np.empty((len(records.example_ids), len(DEPTH_KEYS), 4096), dtype=np.float32)
    for example_index, path in enumerate(records.residual_paths):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            available = set(tensors.keys())
            missing = set(DEPTH_KEYS) - available
            if missing:
                raise ValueError(f"{path} is missing residual tensors: {sorted(missing)}")
            for depth, key in enumerate(DEPTH_KEYS):
                activation = tensors.get_tensor(key).float()
                if activation.ndim != 2 or activation.shape[1] != 4096:
                    raise ValueError(f"Unexpected shape for {key} in {path}: {tuple(activation.shape)}")
                if aggregation == "final_valid_response_token":
                    vector = activation[-1]
                else:
                    response_tokens = activation[1:] if activation.shape[0] > 1 else activation
                    vector = response_tokens.mean(dim=0)
                matrix[example_index, depth] = vector.numpy()
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values found in {records.dataset} {aggregation} matrix")
    return matrix


def make_grouped_split(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    ratios: tuple[float, float, float],
    seed: int,
    candidate_assignments: int,
) -> GroupedSplit:
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_sizes = np.bincount(inverse)
    group_positives = np.bincount(inverse, weights=labels).astype(np.int64)
    group_negatives = group_sizes - group_positives
    targets = np.asarray(ratios, dtype=np.float64)
    target_sizes = targets * len(labels)
    target_positives = targets * int(labels.sum())
    target_negatives = targets * int((labels == 0).sum())

    rng = np.random.default_rng(seed)
    best_score = np.inf
    best_assignment: np.ndarray | None = None
    generated = 0
    batch_size = 5000
    while generated < candidate_assignments:
        current = min(batch_size, candidate_assignments - generated)
        assignments = rng.choice(3, size=(current, len(unique_groups)), p=targets)
        sizes = np.stack(
            [((assignments == split) * group_sizes).sum(axis=1) for split in range(3)], axis=1
        )
        positives = np.stack(
            [
                ((assignments == split) * group_positives).sum(axis=1)
                for split in range(3)
            ],
            axis=1,
        )
        negatives = np.stack(
            [
                ((assignments == split) * group_negatives).sum(axis=1)
                for split in range(3)
            ],
            axis=1,
        )
        valid = (positives > 0).all(axis=1) & (negatives > 0).all(axis=1)
        if valid.any():
            score = (
                ((sizes - target_sizes) ** 2 / (target_sizes + 1)).sum(axis=1)
                + ((positives - target_positives) ** 2 / (target_positives + 1)).sum(axis=1)
                + ((negatives - target_negatives) ** 2 / (target_negatives + 1)).sum(axis=1)
            )
            score[~valid] = np.inf
            index = int(np.argmin(score))
            if score[index] < best_score:
                best_score = float(score[index])
                best_assignment = assignments[index].copy()
        generated += current

    if best_assignment is None:
        raise ValueError("Could not create a grouped split containing both classes in every split")

    sample_assignment = best_assignment[inverse]
    indices = tuple(np.flatnonzero(sample_assignment == split) for split in range(3))
    names = ("train", "validation", "test")
    metadata: dict[str, Any] = {
        "algorithm": "seeded_group_assignment_search",
        "seed": seed,
        "candidate_assignments": candidate_assignments,
        "objective_score": best_score,
        "grouping_verified_disjoint": True,
        "splits": {},
    }
    seen: set[str] = set()
    for name, split_indices in zip(names, indices, strict=True):
        split_groups = set(str(value) for value in groups[split_indices])
        if seen & split_groups:
            raise AssertionError("Conversation group appears in multiple splits")
        seen |= split_groups
        split_labels = labels[split_indices]
        metadata["splits"][name] = {
            "examples": int(len(split_indices)),
            "groups": len(split_groups),
            "label_counts": {
                "0": int((split_labels == 0).sum()),
                "1": int((split_labels == 1).sum()),
            },
        }
    return GroupedSplit(train=indices[0], validation=indices[1], test=indices[2], metadata=metadata)


def _row_space_basis(matrix: np.ndarray) -> np.ndarray:
    q, r, _ = qr(matrix.T, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0:
        raise ValueError("Training activation matrix has zero rank")
    tolerance = np.finfo(matrix.dtype).eps * max(matrix.shape) * diagonal[0]
    rank = int((diagonal > tolerance).sum())
    return np.asarray(q[:, :rank], dtype=np.float32)


def prepare_layer_designs(matrix: np.ndarray, split: GroupedSplit) -> tuple[LayerDesign, ...]:
    development_indices = np.concatenate([split.train, split.validation])
    designs: list[LayerDesign] = []
    for depth in range(matrix.shape[1]):
        scaler = StandardScaler(copy=True)
        scaler.fit(matrix[split.train, depth])
        scaled = scaler.transform(matrix[:, depth]).astype(np.float32, copy=False)

        train_basis = _row_space_basis(scaled[split.train])
        development_basis = _row_space_basis(scaled[development_indices])
        designs.append(
            LayerDesign(
                train=scaled[split.train] @ train_basis,
                validation=scaled[split.validation] @ train_basis,
                development=scaled[development_indices] @ development_basis,
                test=scaled[split.test] @ development_basis,
                train_rank=train_basis.shape[1],
                development_rank=development_basis.shape[1],
            )
        )
    return tuple(designs)


def _score_predictions(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    return (
        float(accuracy_score(labels, predictions)),
        float(f1_score(labels, predictions, average="macro", zero_division=0)),
    )


def evaluate_layer(
    design: LayerDesign,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_development: np.ndarray,
    y_test: np.ndarray,
    *,
    c_grid: tuple[float, ...],
    max_iter: int,
    random_state: int,
) -> tuple[float, float, float, float, float]:
    best_c = c_grid[0]
    best_validation = (-np.inf, -np.inf)
    for c_value in c_grid:
        classifier = LogisticRegression(
            C=c_value,
            l1_ratio=0,
            solver="lbfgs",
            max_iter=max_iter,
            class_weight=None,
            random_state=random_state,
        )
        classifier.fit(design.train, y_train)
        validation = _score_predictions(y_validation, classifier.predict(design.validation))
        selection_key = (validation[1], validation[0])
        if selection_key > best_validation:
            best_validation = selection_key
            best_c = c_value

    final_classifier = LogisticRegression(
        C=best_c,
        l1_ratio=0,
        solver="lbfgs",
        max_iter=max_iter,
        class_weight=None,
        random_state=random_state,
    )
    final_classifier.fit(design.development, y_development)
    test_accuracy, test_macro_f1 = _score_predictions(
        y_test, final_classifier.predict(design.test)
    )
    return best_c, best_validation[1], best_validation[0], test_accuracy, test_macro_f1


def evaluate_all_layers(
    designs: tuple[LayerDesign, ...],
    labels: np.ndarray,
    split: GroupedSplit,
    *,
    c_grid: tuple[float, ...],
    max_iter: int,
    random_state: int,
) -> np.ndarray:
    y_train = labels[split.train]
    y_validation = labels[split.validation]
    development_indices = np.concatenate([split.train, split.validation])
    y_development = labels[development_indices]
    y_test = labels[split.test]
    return np.asarray(
        [
            evaluate_layer(
                design,
                y_train,
                y_validation,
                y_development,
                y_test,
                c_grid=c_grid,
                max_iter=max_iter,
                random_state=random_state,
            )
            for design in designs
        ],
        dtype=np.float64,
    )


def _permutation_result(
    permutation_seed: int,
    designs: tuple[LayerDesign, ...],
    development_labels: np.ndarray,
    test_labels: np.ndarray,
    train_size: int,
    c_grid: tuple[float, ...],
    max_iter: int,
) -> np.ndarray:
    rng = np.random.default_rng(permutation_seed)
    shuffled = rng.permutation(development_labels)
    y_train = shuffled[:train_size]
    y_validation = shuffled[train_size:]
    rows = []
    with threadpool_limits(limits=1):
        for design in designs:
            rows.append(
                evaluate_layer(
                    design,
                    y_train,
                    y_validation,
                    shuffled,
                    test_labels,
                    c_grid=c_grid,
                    max_iter=max_iter,
                    random_state=permutation_seed,
                )
            )
    return np.asarray(rows, dtype=np.float32)


def run_permutations(
    designs: tuple[LayerDesign, ...],
    labels: np.ndarray,
    split: GroupedSplit,
    *,
    count: int,
    seed: int,
    c_grid: tuple[float, ...],
    max_iter: int,
    workers: int,
    checkpoint_path: Path,
    chunk_size: int = 20,
) -> np.ndarray:
    development_indices = np.concatenate([split.train, split.validation])
    development_labels = labels[development_indices]
    test_labels = labels[split.test]
    seed_sequence = np.random.SeedSequence(seed)
    permutation_seeds = np.asarray(
        [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(count)],
        dtype=np.uint32,
    )

    completed = np.empty((0, len(designs), 5), dtype=np.float32)
    if checkpoint_path.is_file():
        with np.load(checkpoint_path) as checkpoint:
            stored_seeds = checkpoint["permutation_seeds"]
            stored_results = checkpoint["results"]
        if not np.array_equal(stored_seeds, permutation_seeds[: len(stored_seeds)]):
            raise ValueError(f"Permutation checkpoint seed mismatch: {checkpoint_path}")
        completed = stored_results

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(len(completed), count, chunk_size):
        stop = min(start + chunk_size, count)
        batch = Parallel(n_jobs=workers, prefer="threads")(
            delayed(_permutation_result)(
                int(permutation_seeds[index]),
                designs,
                development_labels,
                test_labels,
                len(split.train),
                c_grid,
                max_iter,
            )
            for index in range(start, stop)
        )
        completed = np.concatenate([completed, np.stack(batch)], axis=0)
        temporary = checkpoint_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            permutation_seeds=permutation_seeds[: len(completed)],
            results=completed,
        )
        os.replace(temporary, checkpoint_path)
        print(f"    permutations {len(completed)}/{count}", flush=True)
    return completed


def label_counts(labels: Iterable[int]) -> dict[str, int]:
    values = np.asarray(tuple(labels))
    return {"0": int((values == 0).sum()), "1": int((values == 1).sum())}

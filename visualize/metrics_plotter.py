"""Plot metric-vs-step curves from saved singular value JSON records."""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .plotter import extract_layer_id, get_record_label, sanitize_path_component


_WEIGHT_TYPES = ["wq", "wk", "wv", "wo", "w1", "w2", "w3", "c_fc", "c_proj"]
_WEIGHT_TYPE_ORDER = {weight_type: idx for idx, weight_type in enumerate(_WEIGHT_TYPES)}
_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def extract_weight_type(layer_name: str) -> Optional[str]:
    """Extract the weight/module type from an exact layer name."""
    last_part = layer_name.split(".")[-1]
    if last_part in _WEIGHT_TYPE_ORDER:
        return last_part

    for weight_type in _WEIGHT_TYPES:
        if f".{weight_type}" in layer_name or layer_name.endswith(weight_type):
            return weight_type
    return None


def select_singular_values(layer_data: Dict[str, Any], record_metadata: Dict[str, Any]) -> List[float]:
    """Select the singular values to use for a layer according to PC rules."""
    if record_metadata.get("pc_enabled") and "singular_values_pc" in layer_data:
        return layer_data["singular_values_pc"]
    return layer_data.get("singular_values", [])


def compute_modified_condition_number(layer_data: Dict[str, Any], record_metadata: Dict[str, Any]) -> float:
    """Compute top-10%-mean divided by bottom-10%-mean from singular values."""
    singular_values = select_singular_values(layer_data, record_metadata)
    values = np.asarray(singular_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    values = np.sort(values)[::-1]
    k = max(1, math.ceil(values.size * 0.1))
    numerator = float(np.mean(values[:k]))
    denominator = float(np.mean(values[-k:]))

    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")

    result = numerator / denominator
    return result if np.isfinite(result) else float("nan")


def compute_quantile_condition_number(layer_data: Dict[str, Any],
                                      record_metadata: Dict[str, Any],
                                      q_hi: float = 0.9,
                                      q_lo: float = 0.1) -> float:
    """Compute q_hi-quantile divided by q_lo-quantile from singular values."""
    singular_values = select_singular_values(layer_data, record_metadata)
    values = np.asarray(singular_values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return float("nan")

    numerator = float(np.quantile(values, q_hi))
    denominator = float(np.quantile(values, q_lo))

    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")

    result = numerator / denominator
    return result if np.isfinite(result) else float("nan")


def compute_svd_entropy(layer_data: Dict[str, Any], record_metadata: Dict[str, Any]) -> float:
    """Compute the normalized entropy of the squared-singular-value distribution.

    Treats p_i = sigma_i^2 / sum(sigma_j^2) as a probability distribution and
    returns H_norm = -sum(p_i * log(p_i)) / log(n), scaled to [0, 1].
    A flat spectrum gives 1.0; a rank-1-dominated spectrum gives ~0.
    """
    singular_values = select_singular_values(layer_data, record_metadata)
    values = np.asarray(singular_values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 2:
        return float("nan")

    sq = values ** 2
    total = sq.sum()
    if total <= 0.0:
        return float("nan")

    probs = sq / total
    entropy = float(-np.sum(probs * np.log(probs)))
    normalized = entropy / math.log(values.size)
    return normalized if np.isfinite(normalized) else float("nan")


def aggregate_mean(values: Sequence[float]) -> float:
    """Aggregate with arithmetic mean after filtering invalid values."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def aggregate_geometric_mean(values: Sequence[float]) -> float:
    """Aggregate with geometric mean over positive finite values only."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array > 0.0)]
    if array.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(array))))


def extract_weight_norm(layer_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    """Read the saved weight norm directly from layer data."""
    value = layer_data.get("weight_norm")
    if value is None:
        return float("nan")
    result = float(value)
    return result if np.isfinite(result) else float("nan")


def extract_max_singular_value(layer_data: Dict[str, Any], record_metadata: Dict[str, Any]) -> float:
    """Read the largest singular value from the selected spectrum."""
    singular_values = select_singular_values(layer_data, record_metadata)
    if not singular_values:
        return float("nan")
    values = np.asarray(singular_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def extract_original_max_singular_value(layer_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    """Read the largest singular value from the original weight spectrum."""
    singular_values = layer_data.get("singular_values", [])
    if not singular_values:
        return float("nan")
    values = np.asarray(singular_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


AGGREGATOR_REGISTRY: Dict[str, Callable[[Sequence[float]], float]] = {
    "mean": aggregate_mean,
    "geometric_mean": aggregate_geometric_mean,
}

METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "modified_condition_number": {
        "function": compute_modified_condition_number,
        "display_name": "Modified Condition Number",
        "y_label": "Modified Condition Number",
        "default_global_aggregator": "geometric_mean",
    },
    "quantile_condition_number": {
        "function": compute_quantile_condition_number,
        "display_name": "Quantile Condition Number (q90/q10)",
        "y_label": "Quantile Condition Number (q90/q10)",
        "default_global_aggregator": "geometric_mean",
    },
    "svd_entropy": {
        "function": compute_svd_entropy,
        "display_name": "Normalized SVD Entropy",
        "y_label": "Normalized SVD Entropy",
        "default_global_aggregator": "mean",
    },
    "weight_norm": {
        "function": extract_weight_norm,
        "display_name": "Weight Norm",
        "y_label": "Weight Norm",
        "default_global_aggregator": "mean",
        "paired_metric": "original_max_singular_value",
        "paired_display_name": "Original Max Singular Value",
        "paired_y_label": "Original Max Singular Value",
        "plot_levels": ["per_layer"],
    },
    "max_singular_value": {
        "function": extract_max_singular_value,
        "display_name": "Max Singular Value",
        "y_label": "Max Singular Value",
        "default_global_aggregator": "mean",
    },
    "original_max_singular_value": {
        "function": extract_original_max_singular_value,
        "display_name": "Original Max Singular Value",
        "y_label": "Original Max Singular Value",
        "default_global_aggregator": "mean",
    },
}


def _extract_step_from_filename(filepath: Path) -> Optional[int]:
    match = re.search(r"singular_values_step_(\d+)\.json$", filepath.name)
    if match:
        return int(match.group(1))
    return None


def _get_record_step(record: Dict[str, Any], filepath: Path) -> int:
    metadata = record.get("metadata", {})
    metadata_step = metadata.get("checkpoint_step")
    if metadata_step is not None:
        return int(metadata_step)

    filename_step = _extract_step_from_filename(filepath)
    if filename_step is not None:
        return filename_step

    raise ValueError(f"Could not determine checkpoint step for {filepath}")


def _load_experiment_records(experiment_dir: str) -> Dict[str, Any]:
    exp_path = Path(experiment_dir).resolve()
    if not exp_path.is_dir():
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")

    json_paths = sorted(exp_path.glob("singular_values_step_*.json"))
    if not json_paths:
        raise ValueError(f"No singular_values_step_*.json found in {experiment_dir}")

    records: List[Dict[str, Any]] = []
    for json_path in json_paths:
        with open(json_path, "r") as f:
            record = json.load(f)
        record["_source_path"] = str(json_path)
        record["_step"] = _get_record_step(record, json_path)
        records.append(record)

    records.sort(key=lambda record: (record["_step"], record["_source_path"]))
    label = get_record_label(records[0], 0) if records else exp_path.name

    return {
        "path": exp_path,
        "label": label,
        "records": records,
    }


def _resolve_output_root(experiment_paths: Sequence[Path], output_dir: Optional[str]) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()

    if len(experiment_paths) == 1:
        return experiment_paths[0] / "metrics_vs_step"

    parent_paths = [str(path.parent) for path in experiment_paths]
    common_parent = Path(os.path.commonpath(parent_paths))
    combined_name = "__".join(sanitize_path_component(path.name) for path in experiment_paths)
    return common_parent / combined_name / "metrics_vs_step"


def _layer_sort_key(layer_name: str) -> Tuple[int, int, str]:
    weight_type = extract_weight_type(layer_name)
    return (
        extract_layer_id(layer_name),
        _WEIGHT_TYPE_ORDER.get(weight_type or "", len(_WEIGHT_TYPE_ORDER)),
        layer_name,
    )


def _make_unique_labels(experiments: Sequence[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    seen_counts: Dict[str, int] = defaultdict(int)

    for experiment in experiments:
        base_label = experiment["label"]
        fallback = sanitize_path_component(experiment["path"].name)
        count = seen_counts[base_label]
        seen_counts[base_label] += 1
        if count == 0:
            labels.append(base_label)
        else:
            labels.append(f"{base_label} ({fallback})")

    return labels


def _extract_metric_series(experiment: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    metric_spec = METRIC_REGISTRY[metric_name]
    metric_function = metric_spec["function"]
    block_aggregator = AGGREGATOR_REGISTRY["mean"]
    global_aggregator = AGGREGATOR_REGISTRY[metric_spec["default_global_aggregator"]]

    per_layer: Dict[str, Dict[int, float]] = defaultdict(dict)
    block_values_by_step: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    global_values_by_step: Dict[int, List[float]] = defaultdict(list)

    for record in experiment["records"]:
        step = record["_step"]
        metadata = record.get("metadata", {})
        for layer_name, layer_data in record.get("layers", {}).items():
            metric_value = metric_function(layer_data, metadata)
            per_layer[layer_name][step] = metric_value
            global_values_by_step[step].append(metric_value)

            weight_type = extract_weight_type(layer_name)
            if weight_type is not None:
                block_values_by_step[weight_type][step].append(metric_value)

    per_block = {
        weight_type: {
            step: block_aggregator(values)
            for step, values in sorted(step_map.items())
        }
        for weight_type, step_map in block_values_by_step.items()
    }
    global_series = {
        step: global_aggregator(values)
        for step, values in sorted(global_values_by_step.items())
    }

    return {
        "label": experiment["label"],
        "per_layer": dict(per_layer),
        "per_block": per_block,
        "global": global_series,
    }


def _plot_series(series_by_label: Dict[str, Dict[int, float]],
                 title: str,
                 y_label: str,
                 output_path: Path,
                 paired_series_by_label: Optional[Dict[str, Dict[int, float]]] = None,
                 paired_y_label: Optional[str] = None,
                 paired_display_name: Optional[str] = None) -> bool:
    if not series_by_label:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    paired_plotted = False

    for idx, (label, step_map) in enumerate(series_by_label.items()):
        if not step_map:
            continue

        steps = sorted(step_map)
        values = [step_map[step] for step in steps]
        if not np.isfinite(np.asarray(values, dtype=float)).any():
            continue

        color = _COLORS[idx % len(_COLORS)]
        ax.plot(steps, values, marker="o", linewidth=2, markersize=4, color=color, label=label)
        plotted = True

    if paired_series_by_label:
        for idx, (label, step_map) in enumerate(paired_series_by_label.items()):
            if not step_map:
                continue

            steps = sorted(step_map)
            values = [step_map[step] for step in steps]
            if not np.isfinite(np.asarray(values, dtype=float)).any():
                continue

            color = _COLORS[idx % len(_COLORS)]
            paired_label = label if paired_display_name is None else f"{label} ({paired_display_name})"
            ax.plot(
                steps,
                values,
                marker="x",
                linewidth=2,
                markersize=4,
                linestyle="--",
                color=color,
                label=paired_label,
            )
            paired_plotted = True

    if not plotted and not paired_plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel(y_label if not paired_plotted or not paired_y_label else f"{y_label} / {paired_y_label}", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return True


def plot_metrics_vs_step(experiment_dirs: Sequence[str],
                         metric_names: Optional[Sequence[str]] = None,
                         output_dir: Optional[str] = None) -> List[str]:
    """Load experiment directories and plot metric-vs-step curves."""
    if not experiment_dirs:
        raise ValueError("At least one experiment directory is required")

    if metric_names is None:
        metric_names = list(METRIC_REGISTRY)

    unknown_metrics = [metric_name for metric_name in metric_names if metric_name not in METRIC_REGISTRY]
    if unknown_metrics:
        raise ValueError(
            f"Unknown metrics: {unknown_metrics}. Available metrics: {sorted(METRIC_REGISTRY)}"
        )

    experiments = [_load_experiment_records(experiment_dir) for experiment_dir in experiment_dirs]
    output_root = _resolve_output_root([experiment["path"] for experiment in experiments], output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []

    unique_labels = _make_unique_labels(experiments)

    for metric_name in metric_names:
        metric_spec = METRIC_REGISTRY[metric_name]
        metric_dir = output_root / metric_name
        per_layer_dir = metric_dir / "per_layer"
        per_block_dir = metric_dir / "per_block"

        experiment_series = {
            label: _extract_metric_series(experiment, metric_name)
            for label, experiment in zip(unique_labels, experiments)
        }

        paired_metric_name = metric_spec.get("paired_metric")
        paired_metric_spec = METRIC_REGISTRY[paired_metric_name] if paired_metric_name is not None else None
        paired_experiment_series = None
        if paired_metric_spec is not None:
            paired_experiment_series = {
                label: _extract_metric_series(experiment, paired_metric_name)
                for label, experiment in zip(unique_labels, experiments)
            }

        plot_levels = set(metric_spec.get("plot_levels", ["per_layer", "per_block", "global"]))

        all_layers = sorted(
            {
                layer_name
                for series in experiment_series.values()
                for layer_name in series["per_layer"].keys()
            },
            key=_layer_sort_key,
        )
        all_blocks = sorted(
            {
                block_name
                for series in experiment_series.values()
                for block_name in series["per_block"].keys()
            },
            key=lambda block_name: (_WEIGHT_TYPE_ORDER.get(block_name, len(_WEIGHT_TYPE_ORDER)), block_name),
        )

        for layer_name in all_layers:
            output_path = per_layer_dir / f"{sanitize_path_component(layer_name)}.png"
            series_by_label = {
                label: series["per_layer"][layer_name]
                for label, series in experiment_series.items()
                if layer_name in series["per_layer"]
            }
            paired_series_by_label = None
            if paired_experiment_series is not None:
                paired_series_by_label = {
                    label: series["per_layer"][layer_name]
                    for label, series in paired_experiment_series.items()
                    if layer_name in series["per_layer"]
                }
            if _plot_series(
                series_by_label,
                title=f"{metric_spec['display_name']} vs Step - {layer_name}",
                y_label=metric_spec["y_label"],
                output_path=output_path,
                paired_series_by_label=paired_series_by_label,
                paired_y_label=metric_spec.get("paired_y_label"),
                paired_display_name=metric_spec.get("paired_display_name"),
            ):
                saved_paths.append(str(output_path))

        if "per_block" in plot_levels:
            for block_name in all_blocks:
                output_path = per_block_dir / f"{sanitize_path_component(block_name)}.png"
                series_by_label = {
                    label: series["per_block"][block_name]
                    for label, series in experiment_series.items()
                    if block_name in series["per_block"]
                }
                paired_series_by_label = None
                if paired_experiment_series is not None:
                    paired_series_by_label = {
                        label: series["per_block"][block_name]
                        for label, series in paired_experiment_series.items()
                        if block_name in series["per_block"]
                    }
                if _plot_series(
                    series_by_label,
                    title=f"{metric_spec['display_name']} vs Step - {block_name.upper()}",
                    y_label=metric_spec["y_label"],
                    output_path=output_path,
                    paired_series_by_label=paired_series_by_label,
                    paired_y_label=metric_spec.get("paired_y_label"),
                    paired_display_name=metric_spec.get("paired_display_name"),
                ):
                    saved_paths.append(str(output_path))

        if "global" in plot_levels:
            global_output_path = metric_dir / "global.png"
            global_series_by_label = {
                label: series["global"]
                for label, series in experiment_series.items()
                if series["global"]
            }
            paired_global_series_by_label = None
            if paired_experiment_series is not None:
                paired_global_series_by_label = {
                    label: series["global"]
                    for label, series in paired_experiment_series.items()
                    if series["global"]
                }
            if _plot_series(
                global_series_by_label,
                title=f"{metric_spec['display_name']} vs Step - Global",
                y_label=metric_spec["y_label"],
                output_path=global_output_path,
                paired_series_by_label=paired_global_series_by_label,
                paired_y_label=metric_spec.get("paired_y_label"),
                paired_display_name=metric_spec.get("paired_display_name"),
            ):
                saved_paths.append(str(global_output_path))

    return saved_paths


def main() -> None:
    """CLI entry point for plotting metric-vs-step curves."""
    parser = argparse.ArgumentParser(
        description="Plot metric-vs-step curves from experiment directories containing singular_values_step_*.json"
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="One or more experiment directories under visualization_output",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=f"Metrics to plot. Available: {', '.join(METRIC_REGISTRY)}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <exp_dir>/metrics_vs_step or a combined sibling directory.",
    )
    args = parser.parse_args()

    plot_metrics_vs_step(
        experiment_dirs=args.experiment_dirs,
        metric_names=args.metrics,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

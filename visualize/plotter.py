"""Plotting module for singular value histograms."""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

PC_LEVEL_CUTOFFS = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.3}


def load_json_records(filepaths: List[str]) -> List[Dict[str, Any]]:
    """Load multiple JSON files containing singular values.

    Args:
        filepaths: List of paths to JSON files

    Returns:
        List of loaded records
    """
    records = []
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            records.append(json.load(f))
    return records


def sanitize_path_component(value: str) -> str:
    """Convert an arbitrary string into a filesystem-safe path component."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "item"


def extract_layer_id(layer_name: str) -> int:
    """Extract layer/block ID from layer name.

    Args:
        layer_name: e.g., "layers.0.attention.wq" or "transformer.h.5.attn.wq"

    Returns:
        Layer ID (integer), or -1 if not found
    """
    match = re.search(r'(?:layers|h)\.(\d+)', layer_name)
    if match:
        return int(match.group(1))
    return -1


def get_record_label(record: Dict[str, Any], idx: int) -> str:
    """Get label for a record (use wandb_comment if available)."""
    meta = record.get("metadata", {})
    comment = meta.get("wandb_comment", "")
    if comment:
        return comment
    return f"Record {idx+1}"


def plot_singular_values(records: List[Dict[str, Any]],
                         output_dir: str = None,
                         per_step: bool = False):
    """Plot histograms of singular values.

    Creates folders by weight type (wq, wk, wv, wo, w1, w2, w3),
    each containing layer{N}.png histograms for all layers.

    Args:
        records: List of loaded JSON records
        output_dir: Directory to save plots (defaults to sibling dir with combined wandb_comments)
        per_step: If True, create separate subfolders for each step
    """
    if not records:
        print("No records to plot")
        return

    # Determine output directory
    if output_dir is None:
        json_dirs = [Path(r.get("_source_path", ".")).parent for r in records]
        if len(records) > 1:
            # Multiple JSONs: common parent / combined sanitized dir names
            common_parent = Path(os.path.commonpath([str(d) for d in json_dirs]))
            dir_name = "__".join(sanitize_path_component(d.name) for d in json_dirs)
            output_dir = common_parent / dir_name
        else:
            # Single JSON: use same dir as before
            output_dir = json_dirs[0]
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all unique layer names
    layer_names = set()
    for record in records:
        layer_names.update(record["layers"].keys())

    # Define weight types
    weight_types = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3', 'c_fc', 'c_proj']

    # Group layers by (layer_id, weight_type)
    layer_weight_map = {}  # (layer_id, weight_type) -> [layer_names]

    for layer_name in layer_names:
        layer_id = extract_layer_id(layer_name)
        for wt in weight_types:
            if wt in layer_name:
                key = (layer_id, wt)
                if key not in layer_weight_map:
                    layer_weight_map[key] = []
                layer_weight_map[key].append(layer_name)
                break

    # Color scheme: different colors for each combination
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Create plots: one folder per weight type, one plot per layer
    for (layer_id, weight_type), layer_names_list in sorted(layer_weight_map.items()):
        if not layer_names_list:
            continue

        # Create weight type folder
        wt_dir = output_dir / weight_type
        wt_dir.mkdir(parents=True, exist_ok=True)

        _, ax = plt.subplots(figsize=(10, 6))

        multi_json_mode = len(records) > 1

        # Pre-collect all data to compute shared bin edges (avoids per-record
        # independent binning, which causes bars at different positions/widths
        # and makes one histogram visually swallow the other).
        per_record_svs = []
        per_record_svs_pc = []
        for record in records:
            svs, svs_pc = [], []
            for layer_name in layer_names_list:
                if layer_name in record["layers"]:
                    layer_data = record["layers"][layer_name]
                    svs.extend(layer_data["singular_values"])
                    if "singular_values_pc" in layer_data:
                        svs_pc.extend(layer_data["singular_values_pc"])
            per_record_svs.append(svs)
            per_record_svs_pc.append(svs_pc)

        # Build shared bin edges from the union of whichever series will be plotted
        if multi_json_mode:
            plot_data_for_bins = [svs_pc if svs_pc else svs
                                  for svs, svs_pc in zip(per_record_svs, per_record_svs_pc)]
        else:
            # Single record: may plot both svs and svs_pc
            plot_data_for_bins = per_record_svs + per_record_svs_pc
        combined = [v for series in plot_data_for_bins for v in series]
        if combined:
            n_bins = max(len(combined) // (2 * max(len(records), 1)), 20)
            shared_bins = np.linspace(min(combined), max(combined), n_bins + 1)
        else:
            shared_bins = 20

        color_idx = 0
        for record_idx, (all_svs, all_svs_pc) in enumerate(zip(per_record_svs, per_record_svs_pc)):
            base_label = get_record_label(records[record_idx], record_idx)

            if all_svs:
                color = COLORS[color_idx % len(COLORS)]

                if multi_json_mode and all_svs_pc:
                    # Multiple JSONs: only plot PC version for PC models
                    ax.hist(all_svs_pc, bins=shared_bins, alpha=0.5, label=f"{base_label} (PC)", color=color)
                elif all_svs_pc:
                    # Single JSON: plot both original and PC
                    ax.hist(all_svs, bins=shared_bins, alpha=0.5, label=base_label, color=color)
                    color_idx += 1
                    pc_color = COLORS[color_idx % len(COLORS)]
                    ax.hist(all_svs_pc, bins=shared_bins, alpha=0.5, label=f"{base_label} (PC)", color=pc_color)
                else:
                    # No PC version: plot original
                    ax.hist(all_svs, bins=shared_bins, alpha=0.5, label=base_label, color=color)

                color_idx += 1

        ax.set_xlabel("Singular Value", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Layer {layer_id} - {weight_type.upper()}", fontsize=14)
        ax.legend(fontsize=10)

        plt.tight_layout()

        # Save to weight type folder
        output_path = wt_dir / f"layer{layer_id}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


def plot_normalized_weight_histograms(records: List[Dict[str, Any]],
                                      output_dir: str = None):
    """Plot S/||W|| singular value histograms for PC layers.

    Uses singular_values and weight_norm stored in JSON. weight_norm is
    consistent with pc_norm_type (Frobenius or operator norm) as computed
    during the SVD analysis pass. Draws a vertical dashed cutoff line
    determined by pc_level.

    Output: {output_dir}/normalized_weights/{weight_type}/layer{N}.png

    Args:
        records: List of loaded JSON records (with _source_path set)
        output_dir: Directory to save plots (defaults to same dir as JSON)
    """
    if not records:
        print("No records to plot")
        return

    # Determine output directory (same logic as plot_singular_values)
    if output_dir is None:
        json_dirs = [Path(r.get("_source_path", ".")).parent for r in records]
        if len(records) > 1:
            common_parent = Path(os.path.commonpath([str(d) for d in json_dirs]))
            dir_name = "__".join(sanitize_path_component(d.name) for d in json_dirs)
            output_dir = common_parent / dir_name
        else:
            output_dir = json_dirs[0]
    else:
        output_dir = Path(output_dir)

    # Collect layer names that have weight_norm (PC-enabled layers)
    layer_names = set()
    for record in records:
        for layer_name, layer_data in record.get("layers", {}).items():
            if "weight_norm" in layer_data:
                layer_names.add(layer_name)

    if not layer_names:
        print("No 'weight_norm' found in records; skipping normalized SV histograms.")
        return

    norm_dir = output_dir / "normalized_weights"
    norm_dir.mkdir(parents=True, exist_ok=True)

    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    weight_types = ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3', 'c_fc', 'c_proj']

    # Group layers by (layer_id, weight_type)
    layer_weight_map = {}
    for layer_name in layer_names:
        layer_id = extract_layer_id(layer_name)
        for wt in weight_types:
            if wt in layer_name:
                layer_weight_map.setdefault((layer_id, wt), []).append(layer_name)
                break

    for (layer_id, weight_type), lnames in sorted(layer_weight_map.items()):
        wt_dir = norm_dir / weight_type
        wt_dir.mkdir(parents=True, exist_ok=True)

        _, ax = plt.subplots(figsize=(10, 6))

        for record_idx, record in enumerate(records):
            label = get_record_label(record, record_idx)
            color = COLORS[record_idx % len(COLORS)]

            # Collect S/||W|| across all layers in this group
            all_sv_normalized = []
            for lname in lnames:
                layer_data = record.get("layers", {}).get(lname, {})
                svs = layer_data.get("singular_values")
                w_norm = layer_data.get("weight_norm")
                if not svs or not w_norm or w_norm == 0:
                    continue
                S = np.array(svs, dtype=np.float32)
                all_sv_normalized.extend((S / w_norm).tolist())

            if not all_sv_normalized:
                continue

            n_bins = max(len(all_sv_normalized) // 2, 20)
            eps = 0.05
            shared_bins = np.linspace(0.0 - eps, 1.0 + eps, n_bins + 1)
            ax.hist(all_sv_normalized, bins=shared_bins, alpha=0.5, color=color, label=label)

            # Draw vertical dashed line at x = cutoff based on pc_level
            pc_config = record.get("metadata", {}).get("pc_config", {})
            pc_level = pc_config.get("pc_level")
            if pc_level in PC_LEVEL_CUTOFFS:
                cutoff = PC_LEVEL_CUTOFFS[pc_level]
                ax.axvline(x=cutoff, linestyle='--', color=color, alpha=0.85,
                           label=f"cutoff={cutoff} (pc_level={pc_level})")

        ax.set_xlabel("S / \u2016W\u2016  singular value", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Layer {layer_id} - {weight_type.upper()}  (S/\u2016W\u2016 histogram)", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        plt.tight_layout()

        output_path = wt_dir / f"layer{layer_id}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


def main():
    """CLI entry point for standalone plotting."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot singular value histograms or W/||W|| histograms from JSON files."
    )
    parser.add_argument("json_files", nargs="+", help="Path(s) to singular_values JSON file(s)")
    parser.add_argument("--normalized", action="store_true",
                        help="Plot W/||W|| element histograms instead of singular value histograms")
    args = parser.parse_args()

    # Load records and store source path
    records = []
    for filepath in args.json_files:
        with open(filepath, 'r') as f:
            record = json.load(f)
            record["_source_path"] = filepath
            records.append(record)

    if args.normalized:
        plot_normalized_weight_histograms(records)
    else:
        plot_singular_values(records)


if __name__ == "__main__":
    main()

"""Visualization module for SVD analysis of model weights."""

__all__ = ["run_visualize", "plot_singular_values", "plot_normalized_weight_histograms", "plot_metrics_vs_step"]


def run_visualize(*args, **kwargs):
    from .svd_analyzer import run_visualize as _run_visualize

    return _run_visualize(*args, **kwargs)


def plot_singular_values(*args, **kwargs):
    from .plotter import plot_singular_values as _plot_singular_values

    return _plot_singular_values(*args, **kwargs)


def plot_normalized_weight_histograms(*args, **kwargs):
    from .plotter import plot_normalized_weight_histograms as _fn
    return _fn(*args, **kwargs)


def plot_metrics_vs_step(*args, **kwargs):
    from .metrics_plotter import plot_metrics_vs_step as _plot_metrics_vs_step

    return _plot_metrics_vs_step(*args, **kwargs)

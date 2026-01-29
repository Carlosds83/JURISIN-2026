# Palette loading utilities for the Viewer.
"""Read theme/palette.yml and expose helpers to style charts consistently."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_PALETTE: Dict[str, Dict[str, Any]] = {
    "base": {
        "background": "#f6f7fb",
        "paper": "#ffffff",
        "plot": "#f0f2f5",
        "text": "#1e1f29",
        "muted": "#5f6673",
        "accent": "#274060",
    },
    "series": {"true": "#1b9aaa", "pred": "#ef476f"},
    "triggers": {
        "interpretability": "#5b8ff9",
        "relevance": "#5ad8a6",
        "completeness": "#9270ca",
        "differential_regime": "#f6bd16",
        "discretionality": "#e86452",
    },
    "risk": {"low": "#4caf50", "high": "#e53935"},
    "divergent": {"start": "#e3f2fd", "end": "#1a73e8"},
    "categorical": [
        "#5b8ff9",
        "#5ad8a6",
        "#9270ca",
        "#f6bd16",
        "#e86452",
        "#4caf50",
        "#e53935",
    ],
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_palette(path: Path = Path("theme/palette.yml")) -> Dict[str, Any]:
    """Load palette YAML, falling back to defaults when unavailable."""

    palette: Dict[str, Any] = DEFAULT_PALETTE
    if path.exists() and yaml is not None:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                palette = _deep_merge(DEFAULT_PALETTE, loaded)
        except Exception:
            palette = DEFAULT_PALETTE
    return palette


def apply_plot_theme(fig, palette: Dict[str, Any]):
    """Apply background and font colors to a Plotly figure."""

    base = palette.get("base", {})
    fig.update_layout(
        plot_bgcolor=base.get("plot", "#ffffff"),
        paper_bgcolor=base.get("paper", "#ffffff"),
        font={"color": base.get("text", "#000000")},
    )
    return fig


def get_trigger_color(palette: Dict[str, Any], trigger_key: str) -> str:
    """Map trigger names to palette colors."""

    return palette.get("triggers", {}).get(trigger_key, palette["base"].get("accent", "#274060"))


def get_series_color(palette: Dict[str, Any], series_key: str) -> str:
    """Retrieve colors for series such as ground truth / predicted."""

    return palette.get("series", {}).get(series_key, palette["base"].get("accent", "#274060"))


def get_risk_color(palette: Dict[str, Any], level: str) -> str:
    """Return the high/low risk color."""

    return palette.get("risk", {}).get(level.lower(), palette["base"].get("accent", "#274060"))


def get_divergent_scale(palette: Dict[str, Any]) -> List[str]:
    """Return a two-stop scale for heatmaps."""

    divergent = palette.get("divergent", {})
    return [
        divergent.get("start", palette["base"].get("background", "#f0f0f0")),
        divergent.get("end", palette["base"].get("accent", "#274060")),
    ]


def get_categorical_sequence(palette: Dict[str, Any]) -> List[str]:
    """Expose a categorical color list for multi-series charts."""

    sequence: Iterable[str] = palette.get("categorical") or []
    fallback = [palette["series"]["true"], palette["series"]["pred"], palette["base"]["accent"]]
    result = [color for color in sequence if color]
    if not result:
        result = fallback
    return result

# Shared UI helpers for the Streamlit viewer.
"""Sidebar, tooltip, and layout utilities reused across the app."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import streamlit as st

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def load_tooltips(path: Path) -> Dict[str, str]:
    """Read copy/tooltips_en.yml while tolerating missing YAML support."""

    if not path.exists():
        return {}

    if yaml is not None:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if v is not None}
        except Exception:
            pass

    fallback: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fallback[key.strip()] = value.strip().strip('"')
    return fallback


def render_tooltip(key: str, tooltips: Dict[str, str]) -> None:
    """Centralized place for rendering tooltip copy."""

    text = tooltips.get(key)
    if text:
        st.caption(f"? {text}")


def render_sidebar(
    run_names: List[str],
    default_selection: str,
    load_manifest_fn: Callable[[str], Optional[Dict[str, str]]],
    tooltips: Dict[str, str],
) -> tuple[str, Optional[Dict[str, str]]]:
    """Render the sidebar controls (run selector + metadata expander)."""

    if not run_names:
        return default_selection, None

    with st.sidebar:
        st.header("Run controls")
        default_index = 0
        if default_selection in run_names:
            default_index = run_names.index(default_selection)
        selected = st.selectbox(
            "Select a run",
            run_names,
            index=default_index,
            key="sidebar_run_selector",
        )
        render_tooltip("run_selector", tooltips)
        manifest = load_manifest_fn(selected)
        with st.expander("Run metadata", expanded=False):
            render_tooltip("manifest_panel", tooltips)
            if not manifest:
                st.info("No manifest.json found for this run.")
            else:
                for field in ("timestamp", "input_dataset", "row_count", "python_version"):
                    if field in manifest:
                        st.write(f"**{field}:** {manifest[field]}")
    return selected, manifest

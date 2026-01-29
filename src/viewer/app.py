# Viewer entry point for rendering run outputs via Streamlit.
"""Streamlit UI composed of sidebar + tabs, powered entirely by saved Runner outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import (
    load_palette,
    apply_plot_theme,
    get_trigger_color,
    get_series_color,
    get_risk_color,
    get_divergent_scale,
    get_categorical_sequence,
)
from ui_components import load_tooltips, render_sidebar, render_tooltip

CONCEPTS: Dict[str, Dict[str, object]] = {
    "interpretability": {
        "label": "Interpretability",
        "aliases": ["interpretability", "interpretabilidad"],
        "required": True,
        "type": "binary",
    },
    "relevance": {
        "label": "Relevance",
        "aliases": ["relevance", "relevancia"],
        "required": True,
        "type": "binary",
    },
    "completeness": {
        "label": "Completeness",
        "aliases": ["completeness", "completitud"],
        "required": True,
        "type": "binary",
    },
    "differential_regime": {
        "label": "Differential Regime",
        "aliases": ["differential_regime", "differential regime", "regimen diferencial"],
        "required": True,
        "type": "binary",
    },
    "discretionality": {
        "label": "Discretionality",
        "aliases": ["discretionality", "discrecionalidad"],
        "required": True,
        "type": "binary",
    },
    "complexity_score": {
        "label": "Complexity score",
        "aliases": ["complexity_score", "complexity", "complexityscore"],
        "required": True,
        "type": "numeric",
    },
    "risk_score": {
        "label": "Risk score",
        "aliases": ["risk_score", "risk", "riskscore"],
        "required": True,
        "type": "numeric",
    },
    "law": {
        "label": "Law",
        "aliases": ["law", "ley", "statute"],
        "required": True,
        "type": "categorical",
    },
    "identifier": {
        "label": "Article identifier",
        "aliases": ["article_id", "article", "id", "identifier"],
        "required": False,
        "type": "categorical",
    },
}

BINARY_KEYS = [
    "interpretability",
    "relevance",
    "completeness",
    "differential_regime",
    "discretionality",
]
SCATTER_COLOR_OPTIONS = [
    "None",
    "Law",
    "High/Low risk_score",
    "High/Low complexity_score",
]
PARALLEL_COLOR_OPTIONS = ["Law", "High/Low risk_score", "High/Low complexity_score"]
NOT_MAPPED_OPTION = "-- Not mapped --"
EMBEDDING_PLOT_TYPES = ["Scatter", "Centroids overlay", "KDE"]


def normalize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def discover_run_directories(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        return []
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    return sorted(run_dirs, key=lambda path: path.name, reverse=True)


def load_manifest(run_dir: Path) -> Optional[Dict[str, str]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_base_dataset(run_dir: Path) -> Optional[pd.DataFrame]:
    dataset_path = run_dir / "data" / "base_modificada.parquet"
    if not dataset_path.exists():
        return None
    try:
        return pd.read_parquet(dataset_path)
    except (ImportError, ValueError):
        st.error("Unable to read Parquet. Install pyarrow or fastparquet in this environment.")
        return None


def load_embeddings_manifest(run_dir: Path) -> Optional[Dict[str, object]]:
    manifest_path = run_dir / "embeddings" / "embeddings_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def load_embedding_rules(run_dir: Path) -> List[Dict[str, object]]:
    rules_path = run_dir / "embeddings" / "groups" / "rules.json"
    if not rules_path.exists():
        return []
    try:
        data = json.loads(rules_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [rule for rule in data if isinstance(rule, dict)]
    except json.JSONDecodeError:
        return []
    return []


def load_rule_labels(run_dir: Path, relative_path: str) -> Optional[pd.DataFrame]:
    labels_path = (run_dir / "embeddings" / relative_path).resolve()
    if not labels_path.exists():
        return None
    try:
        return pd.read_parquet(labels_path)
    except (ImportError, ValueError):
        st.error("Unable to read embeddings labels. Install pyarrow or fastparquet.")
        return None


def load_projection_dataframe(run_dir: Path, relative_path: str) -> Optional[pd.DataFrame]:
    projection_path = (run_dir / "embeddings" / relative_path).resolve()
    if not projection_path.exists():
        return None
    try:
        return pd.read_parquet(projection_path)
    except (ImportError, ValueError):
        st.error("Unable to read projection coordinates. Install pyarrow or fastparquet.")
        return None


def load_embedding_stats(run_dir: Path, embedding_id: str, projection: str, rule_id: str) -> Optional[Dict[str, object]]:
    stats_path = run_dir / "embeddings" / "stats" / f"stats_{embedding_id}_{projection}_{rule_id}.json"
    if not stats_path.exists():
        return None
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def build_control_key(run_name: str, suffix: str) -> str:
    safe_run = run_name.replace(" ", "_")
    return f"{safe_run}_{suffix}"


def get_series(dataset: pd.DataFrame, mapping: Dict[str, Optional[str]], key: str):
    column = mapping.get(key)
    if not column or column not in dataset.columns:
        return None
    return dataset[column]


def get_numeric_series(
    dataset: pd.DataFrame, mapping: Dict[str, Optional[str]], key: str
) -> Optional[pd.Series]:
    series = get_series(dataset, mapping, key)
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce")


def get_categorical_series(
    dataset: pd.DataFrame, mapping: Dict[str, Optional[str]], key: str
) -> Optional[pd.Series]:
    series = get_series(dataset, mapping, key)
    if series is None:
        return None
    return series.astype(str)


def infer_initial_mapping(dataset: pd.DataFrame) -> Dict[str, Optional[str]]:
    normalized_columns = {normalize_token(col): col for col in dataset.columns}
    mapping: Dict[str, Optional[str]] = {}
    for key, config in CONCEPTS.items():
        mapping[key] = None
        for alias in config["aliases"]:
            alias_key = normalize_token(alias)
            if alias_key in normalized_columns:
                mapping[key] = normalized_columns[alias_key]
                break
    return mapping


def prepare_column_mapping(
    run_key: str, dataset: pd.DataFrame, tooltips: Dict[str, str]
) -> Dict[str, Optional[str]]:
    session_key = build_control_key(run_key, "column_mapping")
    if session_key not in st.session_state:
        st.session_state[session_key] = infer_initial_mapping(dataset)

    current_mapping: Dict[str, Optional[str]] = st.session_state[session_key]
    st.subheader("Column mapping")
    render_tooltip("column_mapping_info", tooltips)
    columns = dataset.columns.tolist()
    updated_mapping: Dict[str, Optional[str]] = {}

    for key, config in CONCEPTS.items():
        label = f"{config['label']} column"
        options = [NOT_MAPPED_OPTION] + columns
        current_value = current_mapping.get(key)
        default_index = options.index(current_value) if current_value in options else 0
        selection = st.selectbox(
            label,
            options,
            index=default_index,
            key=f"{session_key}_{key}",
        )
        updated_mapping[key] = None if selection == NOT_MAPPED_OPTION else selection

    st.session_state[session_key] = updated_mapping
    missing_required = [
        config["label"]
        for key, config in CONCEPTS.items()
        if config.get("required") and not updated_mapping.get(key)
    ]
    if missing_required:
        st.warning(
            "Map the following columns to unlock every chart: "
            + ", ".join(missing_required)
        )
    return updated_mapping


def render_dataset_table(dataset: Optional[pd.DataFrame], tooltips: Dict[str, str]):
    st.subheader("Modified dataset")
    render_tooltip("dataset_table", tooltips)
    if dataset is None:
        st.warning("base_modificada.parquet not found for this run.")
        return
    st.dataframe(dataset)


def render_binary_distribution_section(
    run_key: str,
    dataset: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    with st.expander("Binary distribution", expanded=False):
        render_tooltip("binary_distribution_section", tooltips)
        categories = [key for key in BINARY_KEYS if mapping.get(key)]
        if not categories:
            st.info("Map at least one binary column to display this chart.")
            return

        labels = [CONCEPTS[key]["label"] for key in categories]
        selected_label = st.selectbox(
            "Binary category",
            labels,
            key=build_control_key(run_key, "binary_category"),
        )
        render_tooltip("binary_distribution_selector", tooltips)
        selected_key = categories[labels.index(selected_label)]

        mode = st.radio(
            "Display mode",
            ["Counts", "Proportions"],
            key=build_control_key(run_key, "binary_mode"),
            horizontal=True,
        )
        render_tooltip("binary_distribution_mode", tooltips)

        series = get_numeric_series(dataset, mapping, selected_key)
        if series is None:
            st.info("Selected column is not numeric.")
            return

        cleaned = series.dropna()
        counts = cleaned.value_counts().reindex([0, 1], fill_value=0)
        total = counts.sum()
        if total == 0:
            st.info("No records available for the selected category.")
            return

        y_values = counts.values if mode == "Counts" else (counts / total).values
        text = [f"{int(val)}" for val in counts.values] if mode == "Counts" else [f"{val:.1%}" for val in (counts / total).values]
        y_label = "Count" if mode == "Counts" else "Proportion"
        bar_color = get_trigger_color(palette, selected_key)

        fig = px.bar(
            x=["0", "1"],
            y=y_values,
            labels={"x": "Value", "y": y_label},
            text=text,
            color_discrete_sequence=[bar_color],
        )
        fig.update_layout(showlegend=False)
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)


def render_cooccurrence_section(
    run_key: str,
    dataset: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    with st.expander("Co-occurrence matrix", expanded=False):
        render_tooltip("cooccurrence_section", tooltips)
        categories = [key for key in BINARY_KEYS if mapping.get(key)]
        if len(categories) < 2:
            st.info("Map at least two binary columns to study co-occurrence.")
            return

        label_lookup = [CONCEPTS[key]["label"] for key in categories]
        default_selection = label_lookup[:2]
        selected_labels = st.multiselect(
            "Binary categories",
            label_lookup,
            default=default_selection,
            key=build_control_key(run_key, "cooccurrence_selection"),
        )
        render_tooltip("cooccurrence_selector", tooltips)
        if len(selected_labels) < 2:
            st.info("Select at least two categories.")
            return

        selected_keys = [categories[label_lookup.index(label)] for label in selected_labels]
        binary_cols = {key: mapping[key] for key in selected_keys}
        binary_df = dataset[list(binary_cols.values())].apply(pd.to_numeric, errors="coerce")
        binary_df = binary_df.dropna()
        if binary_df.empty:
            st.info("No overlapping rows found for the selected categories.")
            return

        mode = st.radio(
            "Display",
            ["Counts", "Proportions"],
            key=build_control_key(run_key, "cooccurrence_mode"),
            horizontal=True,
        )

        if len(selected_keys) == 2:
            renamed = binary_df.rename(
                columns={
                    binary_cols[selected_keys[0]]: selected_labels[0],
                    binary_cols[selected_keys[1]]: selected_labels[1],
                }
            )
            table = pd.crosstab(renamed[selected_labels[0]], renamed[selected_labels[1]])
            table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            if mode == "Proportions" and table.to_numpy().sum() > 0:
                table = table / table.to_numpy().sum()
            fig = px.imshow(
                table,
                text_auto=True,
                color_continuous_scale=get_divergent_scale(palette),
                labels={"x": selected_labels[1], "y": selected_labels[0]},
            )
        else:
            renamed = {
                binary_cols[key]: CONCEPTS[key]["label"] for key in selected_keys
            }
            matrix_df = binary_df.rename(columns=renamed)
            labels = list(renamed.values())
            matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
            total_rows = len(matrix_df)
            for row_label in labels:
                for col_label in labels:
                    both = (matrix_df[row_label] == 1) & (matrix_df[col_label] == 1)
                    value = both.sum()
                    if mode == "Proportions" and total_rows > 0:
                        value = value / total_rows
                    matrix.loc[row_label, col_label] = value
            fig = px.imshow(
                matrix,
                text_auto=True,
                color_continuous_scale=get_divergent_scale(palette),
                labels={"x": "Category", "y": "Category"},
            )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)


def render_boxplot_interpretability_section(
    run_key: str,
    dataset: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    with st.expander("Interpretability distribution", expanded=False):
        render_tooltip("boxplot_interpretability", tooltips)
        target = get_numeric_series(dataset, mapping, "interpretability")
        if target is None:
            st.info("Interpretability column is not mapped.")
            return
        target = target.dropna()
        if target.empty:
            st.info("Interpretability values are missing.")
            return

        grouping_candidates = [
            key for key in BINARY_KEYS if key != "interpretability" and mapping.get(key)
        ]
        if not grouping_candidates:
            st.info("Map another binary column to group interpretability values.")
            return

        labels = [CONCEPTS[key]["label"] for key in grouping_candidates]
        selected_label = st.selectbox(
            "Grouping category",
            labels,
            key=build_control_key(run_key, "interpretability_group"),
        )
        selected_key = grouping_candidates[labels.index(selected_label)]
        group_series = get_numeric_series(dataset, mapping, selected_key)
        if group_series is None:
            st.info("Selected grouping column is unavailable.")
            return

        df_plot = pd.DataFrame(
            {
                "Interpretability": target,
                "Grouping": dataset.loc[target.index, mapping[selected_key]].astype(str),
            }
        ).dropna()
        if df_plot.empty:
            st.info("No overlapping rows for this grouping.")
            return
        fig = px.box(
            df_plot,
            x="Grouping",
            y="Interpretability",
            color="Grouping",
            points="all",
            color_discrete_sequence=get_categorical_sequence(palette),
        )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)


def render_score_boxplot_section(
    run_key: str,
    dataset: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    score_key: str,
    section_title: str,
    tooltip_key: str,
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    with st.expander(section_title, expanded=False):
        render_tooltip(tooltip_key, tooltips)
        score_series = get_numeric_series(dataset, mapping, score_key)
        if score_series is None:
            st.info(f"{CONCEPTS[score_key]['label']} is not mapped.")
            return
        score_series = score_series.dropna()
        if score_series.empty:
            st.info("No numeric values available for this score.")
            return

        grouping_options: List[Tuple[str, Optional[str]]] = [("None", None)]
        grouping_options += [
            (CONCEPTS[key]["label"], key)
            for key in BINARY_KEYS
            if mapping.get(key)
        ]
        if mapping.get("law"):
            grouping_options.append((CONCEPTS["law"]["label"], "law"))

        option_labels = [label for label, _ in grouping_options]
        selected_label = st.selectbox(
            "Grouping",
            option_labels,
            key=build_control_key(run_key, f"{score_key}_grouping"),
        )
        selected_key = dict(grouping_options)[selected_label]

        df_plot = pd.DataFrame({CONCEPTS[score_key]["label"]: score_series}).dropna()
        if selected_key:
            group_series = get_series(dataset, mapping, selected_key)
            if group_series is None:
                st.info("Selected grouping column is unavailable.")
                return
            df_plot["Grouping"] = group_series.loc[df_plot.index].astype(str)
            fig = px.box(
                df_plot,
                x="Grouping",
                y=CONCEPTS[score_key]["label"],
                color="Grouping",
                points="all",
                color_discrete_sequence=get_categorical_sequence(palette),
            )
        else:
            fig = px.box(
                df_plot,
                y=CONCEPTS[score_key]["label"],
                points="all",
                color_discrete_sequence=[get_series_color(palette, "true")],
            )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)


def compute_high_low_labels(
    values: pd.Series,
    method: str,
    custom_threshold: float,
) -> Tuple[pd.Series, float]:
    threshold = values.median() if method == "Median" else custom_threshold
    labels = values.apply(lambda val: "High" if val >= threshold else "Low")
    return labels, threshold


def render_scatter_section(
    run_key: str,
    dataset: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    with st.expander("Complexity vs risk scatter", expanded=False):
        render_tooltip("scatter_section", tooltips)
        complexity = get_numeric_series(dataset, mapping, "complexity_score")
        risk = get_numeric_series(dataset, mapping, "risk_score")
        if complexity is None or risk is None:
            st.info("Both complexity_score and risk_score must be mapped.")
            return

        combined = pd.DataFrame({"Complexity": complexity, "Risk": risk}).dropna()
        if combined.empty:
            st.info("No overlapping rows for complexity and risk.")
            return

        color_choice = st.selectbox(
            "Color by",
            SCATTER_COLOR_OPTIONS,
            key=build_control_key(run_key, "scatter_color"),
        )

        color_kwargs: Dict[str, object] = {}
        if color_choice == "Law":
            law_series = get_categorical_series(dataset, mapping, "law")
            if law_series is not None:
                combined["Color"] = law_series.loc[combined.index]
                color_kwargs["color"] = "Color"
                color_kwargs["color_discrete_sequence"] = get_categorical_sequence(palette)
        elif color_choice in {"High/Low risk_score", "High/Low complexity_score"}:
            base_key = "risk_score" if "risk" in color_choice else "complexity_score"
            values = (risk if base_key == "risk_score" else complexity).loc[combined.index]
            method = st.radio(
                "Threshold method",
                ["Median", "Custom"],
                horizontal=True,
                key=build_control_key(run_key, f"{color_choice}_method"),
            )
            render_tooltip("scatter_threshold", tooltips)
            if method == "Custom":
                default_value = float(values.median()) if not values.empty else 0.0
                threshold_value = st.number_input(
                    "Custom threshold",
                    value=default_value,
                    key=build_control_key(run_key, f"{color_choice}_threshold"),
                )
            else:
                threshold_value = float(values.median()) if not values.empty else 0.0
            labels, _ = compute_high_low_labels(values, method, threshold_value)
            combined["Color"] = labels
            color_kwargs["color"] = "Color"
            color_kwargs["color_discrete_map"] = {
                "High": get_risk_color(palette, "high"),
                "Low": get_risk_color(palette, "low"),
            }
        else:
            color_kwargs["color_discrete_sequence"] = [get_series_color(palette, "true")]

        fig = px.scatter(
            combined,
            x="Complexity",
            y="Risk",
            **color_kwargs,
        )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)


def render_parallel_coordinates_section(
    run_key: str,
    dataset: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    with st.expander("Parallel coordinates", expanded=False):
        render_tooltip("parallel_coordinates_section", tooltips)
        required_missing = [key for key in BINARY_KEYS if not mapping.get(key)]
        if required_missing:
            st.info("Map every binary trigger to enable this plot.")
            return

        renamed = {
            mapping[key]: CONCEPTS[key]["label"] for key in BINARY_KEYS if mapping.get(key)
        }
        data = dataset[list(renamed.keys())].rename(columns=renamed)
        data = data.apply(pd.to_numeric, errors="coerce").dropna()
        if data.empty:
            st.info("Binary columns contain no numeric values.")
            return

        max_rows = len(data)
        if max_rows > 1000:
            sample_size = st.slider(
                "Rows to plot",
                min_value=200,
                max_value=max_rows,
                value=min(1000, max_rows),
                step=50,
                key=build_control_key(run_key, "parallel_sample"),
            )
            render_tooltip("parallel_sampling", tooltips)
            data = data.sample(sample_size, random_state=42)

        color_choice = st.selectbox(
            "Color by",
            PARALLEL_COLOR_OPTIONS,
            key=build_control_key(run_key, "parallel_color"),
        )
        render_tooltip("parallel_color_selector", tooltips)

        color_values: Optional[pd.Series] = None
        if color_choice == "Law":
            law_series = get_categorical_series(dataset, mapping, "law")
            if law_series is not None:
                color_values = law_series.loc[data.index]
        else:
            base_key = "risk_score" if "risk" in color_choice else "complexity_score"
            base_series = get_numeric_series(dataset, mapping, base_key)
            if base_series is not None:
                base_series = base_series.loc[data.index]
                method = st.radio(
                    "Threshold method",
                    ["Median", "Custom"],
                    horizontal=True,
                    key=build_control_key(run_key, f"{color_choice}_method"),
                )
                if method == "Custom":
                    default_value = float(base_series.median()) if not base_series.empty else 0.0
                    threshold_value = st.number_input(
                        "Custom threshold",
                        value=default_value,
                        key=build_control_key(run_key, f"{color_choice}_threshold"),
                    )
                else:
                    threshold_value = float(base_series.median()) if not base_series.empty else 0.0
                labels, _ = compute_high_low_labels(base_series, method, threshold_value)
                color_values = labels

        if color_values is not None:
            categories = color_values.astype("category")
            codes = categories.cat.codes
            legend_map = dict(enumerate(categories.cat.categories))
            st.caption(
                "Color legend codes: "
                + ", ".join(f"{code}->{label}" for code, label in legend_map.items())
            )
            color_scale = get_divergent_scale(palette)
            fig = px.parallel_coordinates(
                data,
                color=codes,
                color_continuous_scale=color_scale,
                labels={col: col for col in data.columns},
            )
        else:
            fig = px.parallel_coordinates(
                data,
                labels={col: col for col in data.columns},
            )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)


def render_law_summary_section(
    run_key: str,
    dataset: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    tooltips: Dict[str, str],
):
    with st.expander("Law summary table", expanded=False):
        render_tooltip("law_summary_section", tooltips)
        law_series = get_categorical_series(dataset, mapping, "law")
        interpretability = get_numeric_series(dataset, mapping, "interpretability")
        complexity = get_numeric_series(dataset, mapping, "complexity_score")
        risk = get_numeric_series(dataset, mapping, "risk_score")
        if law_series is None or interpretability is None or complexity is None or risk is None:
            st.info("Map law, interpretability, complexity_score, and risk_score to inspect this summary.")
            return

        available_categories = [key for key in BINARY_KEYS if mapping.get(key)]
        option_labels = [CONCEPTS[key]["label"] for key in available_categories]
        selected_labels = st.multiselect(
            "Category columns",
            option_labels,
            default=option_labels,
            key=build_control_key(run_key, "law_summary_categories"),
        )
        render_tooltip("law_summary_selector", tooltips)
        selected_keys = [
            available_categories[option_labels.index(label)] for label in selected_labels
        ]

        data = pd.DataFrame(
            {
                "Law": law_series,
                "Interpretability": interpretability,
                "Complexity": complexity,
                "Risk": risk,
            }
        )
        for key in selected_keys:
            data[CONCEPTS[key]["label"]] = get_numeric_series(dataset, mapping, key)
        data = data.dropna(subset=["Law"])
        if data.empty:
            st.info("No rows available after filtering by Law.")
            return

        group = data.groupby("Law")
        summary = pd.DataFrame(
            {
                "mean_interpretability": group["Interpretability"].mean(),
                "mean_complexity_score": group["Complexity"].mean(),
                "mean_risk_score": group["Risk"].mean(),
            }
        )
        for key in selected_keys:
            label = CONCEPTS[key]["label"]
            summary[f"prop_{label}"] = group[label].mean()
        st.dataframe(summary.reset_index())


def render_extreme_articles_section(
    run_key: str,
    dataset: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    tooltips: Dict[str, str],
):
    with st.expander("Extreme articles", expanded=False):
        render_tooltip("extremes_section", tooltips)
        law_series = get_categorical_series(dataset, mapping, "law")
        if law_series is None:
            st.info("Map the Law column to list articles.")
            return

        score_options = [
            (CONCEPTS["complexity_score"]["label"], "complexity_score"),
            (CONCEPTS["risk_score"]["label"], "risk_score"),
        ]
        option_labels = [label for label, _ in score_options]
        selected_label = st.selectbox(
            "Score",
            option_labels,
            key=build_control_key(run_key, "extreme_score"),
        )
        render_tooltip("extremes_selector", tooltips)
        score_key = dict(score_options)[selected_label]
        score_series = get_numeric_series(dataset, mapping, score_key)
        if score_series is None:
            st.info("Selected score column is not mapped.")
            return

        direction = st.selectbox(
            "Direction",
            ["Top 5", "Bottom 5"],
            key=build_control_key(run_key, "extreme_direction"),
        )

        identifier_series = get_categorical_series(dataset, mapping, "identifier")
        identifier = (
            identifier_series
            if identifier_series is not None
            else pd.Series(dataset.index.astype(str), index=dataset.index, name="Row index")
        )
        table = pd.DataFrame(
            {
                "Identifier": identifier,
                "Law": law_series,
                selected_label: score_series,
            }
        ).dropna(subset=[selected_label])
        if table.empty:
            st.info("No rows available for ranking.")
            return

        ascending = direction == "Bottom 5"
        result = table.sort_values(by=selected_label, ascending=ascending).head(5)
        st.dataframe(result)


def render_data_explorer(
    run_key: str,
    dataset: Optional[pd.DataFrame],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    if dataset is None or dataset.empty:
        st.info("Dataset is empty; data explorer is disabled.")
        return

    with st.expander("Column mapping", expanded=False):
        mapping = prepare_column_mapping(run_key, dataset, tooltips)

    render_binary_distribution_section(run_key, dataset, mapping, tooltips, palette)
    render_cooccurrence_section(run_key, dataset, mapping, tooltips, palette)
    render_boxplot_interpretability_section(run_key, dataset, mapping, tooltips, palette)

    render_score_boxplot_section(
        run_key,
        dataset,
        mapping,
        score_key="complexity_score",
        section_title="Complexity score boxplot",
        tooltip_key="boxplot_complexity",
        tooltips=tooltips,
        palette=palette,
    )
    render_score_boxplot_section(
        run_key,
        dataset,
        mapping,
        score_key="risk_score",
        section_title="Risk score boxplot",
        tooltip_key="boxplot_risk",
        tooltips=tooltips,
        palette=palette,
    )
    render_scatter_section(run_key, dataset, mapping, tooltips, palette)
    render_parallel_coordinates_section(run_key, dataset, mapping, tooltips, palette)
    render_law_summary_section(run_key, dataset, mapping, tooltips)
    render_extreme_articles_section(run_key, dataset, mapping, tooltips)


def load_classification_results_table(classification_dir: Path) -> Optional[pd.DataFrame]:
    table_path = classification_dir / "results_table.parquet"
    if not table_path.exists():
        return None
    try:
        return pd.read_parquet(table_path)
    except Exception as error:
        st.warning(f"Unable to read {table_path.name}: {error}")
        return None


def build_experiment_metadata(
    classification_dir: Path, results_table: Optional[pd.DataFrame]
) -> Dict[str, Dict[str, Optional[str]]]:
    metadata: Dict[str, Dict[str, Optional[str]]] = {}
    experiments_dir = classification_dir / "experiments"

    if results_table is not None and not results_table.empty:
        id_col = next(
            (
                column
                for column in ("experiment_id", "experiment", "id", "name")
                if column in results_table.columns
            ),
            None,
        )
        if id_col:
            for _, row in results_table.iterrows():
                experiment_id = str(row[id_col])
                entry = metadata.setdefault(experiment_id, {})
                for field in ("target_name", "embedding", "token_level", "model_name"):
                    if field in row and pd.notna(row[field]) and row[field] != "":
                        entry[field] = str(row[field])

    if experiments_dir.exists():
        for experiment_path in experiments_dir.iterdir():
            if experiment_path.is_dir():
                metadata.setdefault(experiment_path.name, {})

    return metadata


def render_classification_selectors(
    run_key: str,
    metadata: Dict[str, Dict[str, Optional[str]]],
    tooltips: Dict[str, str],
) -> tuple[Optional[str], Optional[str]]:
    if not metadata:
        st.info("No classification experiments found in this run.")
        return None, None

    records = []
    for experiment_id, meta in metadata.items():
        entry = {"experiment_id": experiment_id}
        entry.update(meta)
        records.append(entry)
    meta_df = pd.DataFrame(records)

    if "target_name" not in meta_df.columns:
        meta_df["target_name"] = "Unknown target"

    target_options = (
        sorted(meta_df["target_name"].dropna().unique().tolist()) or ["Unknown target"]
    )
    target = st.selectbox(
        "Target",
        target_options,
        key=build_control_key(run_key, "classification_target"),
    )

    filtered = meta_df[meta_df["target_name"] == target]
    for field in ("embedding", "token_level", "model_name"):
        if field in filtered.columns and filtered[field].dropna().nunique() > 1:
            options = ["All"] + sorted(
                filtered[field].dropna().astype(str).unique().tolist()
            )
            selection = st.selectbox(
                field.replace("_", " ").title(),
                options,
                key=build_control_key(run_key, f"classification_{field}"),
            )
            if selection != "All":
                filtered = filtered[filtered[field].astype(str) == selection]

    if filtered.empty:
        st.info("No experiment matches the selected filters.")
        return target, None

    experiment_ids = filtered["experiment_id"].astype(str).unique().tolist()
    experiment_id = st.selectbox(
        "Experiment",
        experiment_ids,
        key=build_control_key(run_key, "classification_experiment"),
    )

    return target, experiment_id


def load_experiment_artifacts(
    classification_dir: Path, experiment_id: str
) -> Dict[str, object]:
    artifacts: Dict[str, object] = {}
    experiment_dir = classification_dir / "experiments" / experiment_id
    if not experiment_dir.exists():
        st.warning(f"Experiment folder not found: {experiment_id}")
        return artifacts

    predictions_path = experiment_dir / "predictions.parquet"
    if predictions_path.exists():
        try:
            artifacts["predictions"] = pd.read_parquet(predictions_path)
        except Exception as error:
            st.error(f"Unable to read predictions.parquet: {error}")
    else:
        st.info("predictions.parquet not found in the selected experiment.")

    metrics_path = experiment_dir / "metrics.json"
    if metrics_path.exists():
        try:
            artifacts["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            st.warning(f"metrics.json is not valid JSON: {error}")

    class_labels_path = experiment_dir / "class_labels.json"
    if class_labels_path.exists():
        try:
            artifacts["label_map"] = json.loads(
                class_labels_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            st.warning("class_labels.json is not valid JSON.")

    confusion_path = None
    for filename in ("confusion_matrix.csv", "confusion_matrix.json", "confusion_matrix.npy"):
        candidate = experiment_dir / filename
        if candidate.exists():
            confusion_path = candidate
            break

    if confusion_path is not None:
        try:
            if confusion_path.suffix == ".csv":
                artifacts["confusion_matrix"] = pd.read_csv(confusion_path, index_col=0)
            elif confusion_path.suffix == ".json":
                matrix = json.loads(confusion_path.read_text(encoding="utf-8"))
                artifacts["confusion_matrix"] = pd.DataFrame(matrix)
            elif confusion_path.suffix == ".npy":
                matrix = np.load(confusion_path)
                artifacts["confusion_matrix"] = pd.DataFrame(matrix)
        except Exception as error:
            st.warning(f"Unable to load {confusion_path.name}: {error}")

    return artifacts


def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.fillna("N/A").astype(str)


def get_ordered_classes(y_true: pd.Series, y_pred: pd.Series) -> List[str]:
    combined = pd.concat([y_true, y_pred], ignore_index=True)
    classes: List[str] = []
    for value in combined:
        if value not in classes:
            classes.append(value)
    return classes


def map_label_for_display(label: str, label_map: Optional[Dict[str, str]]) -> str:
    if not label_map:
        return str(label)
    return str(label_map.get(str(label), label_map.get(label, label)))


def render_class_distribution_section(
    run_key: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    classes: List[str],
    display_labels: Dict[str, str],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    with st.expander("Class distribution", expanded=False):
        render_tooltip("classification.class_distribution", tooltips)
        true_counts = y_true.value_counts().reindex(classes, fill_value=0)
        pred_counts = y_pred.value_counts().reindex(classes, fill_value=0)
        chart_df = pd.DataFrame(
            {
                "Class": [display_labels[label] for label in classes]
                + [display_labels[label] for label in classes],
                "Count": true_counts.tolist() + pred_counts.tolist(),
                "Type": ["Ground truth"] * len(classes) + ["Predicted"] * len(classes),
            }
        )
        fig = px.bar(
            chart_df,
            x="Class",
            y="Count",
            color="Type",
            barmode="group",
            color_discrete_map={
                "Ground truth": get_series_color(palette, "true"),
                "Predicted": get_series_color(palette, "pred"),
            },
        )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)


def compute_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, classes: List[str]
) -> pd.DataFrame:
    matrix = pd.crosstab(y_true, y_pred, dropna=False)
    matrix = matrix.reindex(index=classes, columns=classes, fill_value=0)
    return matrix


def render_confusion_matrix_section(
    run_key: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    classes: List[str],
    display_labels: Dict[str, str],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    with st.expander("Confusion matrix", expanded=False):
        render_tooltip("classification.confusion_matrix", tooltips)
        matrix = compute_confusion_matrix(y_true, y_pred, classes)
        mode = st.radio(
            "Values",
            ["Counts", "Row-normalized"],
            horizontal=True,
            key=build_control_key(run_key, "confusion_mode"),
        )
        if mode == "Row-normalized":
            normalized = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
            heatmap = normalized
            text_auto = ".2f"
        else:
            heatmap = matrix
            text_auto = True

        display_index = [display_labels[label] for label in matrix.index]
        display_columns = [display_labels[label] for label in matrix.columns]

        fig = px.imshow(
            heatmap,
            x=display_columns,
            y=display_index,
            text_auto=text_auto,
            color_continuous_scale=get_divergent_scale(palette),
        )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)


def calculate_classification_metrics(
    y_true: pd.Series, y_pred: pd.Series, classes: List[str]
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    rows = []
    total_tp = total_fp = total_fn = 0
    for label in classes:
        true_mask = y_true == label
        pred_mask = y_pred == label
        tp = (true_mask & pred_mask).sum()
        fp = (~true_mask & pred_mask).sum()
        fn = (true_mask & ~pred_mask).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = true_mask.sum()
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    metrics_df = pd.DataFrame(rows)
    total_support = metrics_df["support"].sum()
    macro = {
        "precision": metrics_df["precision"].mean(),
        "recall": metrics_df["recall"].mean(),
        "f1": metrics_df["f1"].mean(),
    }
    weighted = {
        "precision": (metrics_df["precision"] * metrics_df["support"]).sum() / total_support
        if total_support
        else 0.0,
        "recall": (metrics_df["recall"] * metrics_df["support"]).sum() / total_support
        if total_support
        else 0.0,
        "f1": (metrics_df["f1"] * metrics_df["support"]).sum() / total_support
        if total_support
        else 0.0,
    }
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        ),
    }

    summaries = {"macro": macro, "weighted": weighted, "micro": micro}
    return metrics_df, summaries


def render_per_class_metrics_section(
    run_key: str,
    metrics_df: pd.DataFrame,
    summaries: Dict[str, Dict[str, float]],
    display_labels: Dict[str, str],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    with st.expander("Per-class metrics", expanded=False):
        render_tooltip("classification.per_class_metrics", tooltips)
        if metrics_df.empty:
            st.info("No metrics available for the selected experiment.")
            return

        metrics_df = metrics_df.copy()
        metrics_df["Class"] = metrics_df["label"].map(display_labels)

        metric_choice = st.selectbox(
            "Metric",
            ["f1", "precision", "recall"],
            index=0,
            key=build_control_key(run_key, "classification_metric_choice"),
        )
        fig = px.bar(
            metrics_df,
            x="Class",
            y=metric_choice,
            color_discrete_sequence=[get_series_color(palette, "true")],
        )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)

        summary_rows = []
        for name, values in summaries.items():
            summary_rows.append(
                {
                    "summary": name,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                }
            )
        st.dataframe(pd.DataFrame(summary_rows))


def extract_score_series(predictions: pd.DataFrame) -> Dict[str, pd.Series]:
    scores: Dict[str, pd.Series] = {}
    score_columns = [
        column for column in predictions.columns if column.startswith("score_")
    ]
    if score_columns:
        for column in score_columns:
            label = column.split("score_", 1)[1]
            scores[label] = pd.to_numeric(predictions[column], errors="coerce")
        return scores

    if "y_score" in predictions.columns:
        scores["__binary__"] = pd.to_numeric(predictions["y_score"], errors="coerce")
        return scores

    for column in predictions.columns:
        if predictions[column].dtype != object:
            continue
        series = predictions[column]
        sample = series.dropna().head(1)
        if sample.empty:
            continue
        value = sample.iloc[0]
        if isinstance(value, dict):
            for key in value.keys():
                scores[key] = pd.to_numeric(
                    series.apply(
                        lambda entry: entry.get(key) if isinstance(entry, dict) else None
                    ),
                    errors="coerce",
                )
            return scores
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                continue
            if isinstance(parsed, dict):
                parsed_series = series.apply(
                    lambda entry: json.loads(entry) if isinstance(entry, str) else None
                )
                for key in parsed.keys():
                    scores[key] = pd.to_numeric(
                        parsed_series.apply(
                            lambda entry: entry.get(key) if isinstance(entry, dict) else None
                        ),
                        errors="coerce",
                    )
                return scores

    return scores


def compute_curve_points(
    y_true_binary: pd.Series, y_score: pd.Series, curve_type: str
) -> Optional[pd.DataFrame]:
    data = pd.DataFrame({"label": y_true_binary, "score": y_score}).dropna()
    if data.empty:
        return None

    total_pos = data["label"].sum()
    total_neg = len(data) - total_pos
    if total_pos == 0 or total_neg == 0:
        return None

    data = data.sort_values(by="score", ascending=False)
    tp = fp = 0
    x_points: List[float] = []
    y_points: List[float] = []

    if curve_type == "Precision-Recall":
        x_points.append(0.0)
        y_points.append(1.0)
        for value in data["label"]:
            if value == 1:
                tp += 1
            else:
                fp += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / total_pos if total_pos else 0.0
            x_points.append(recall)
            y_points.append(precision)
        return pd.DataFrame({"x": x_points, "y": y_points})

    x_points.append(0.0)
    y_points.append(0.0)
    for value in data["label"]:
        if value == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos if total_pos else 0.0
        fpr = fp / total_neg if total_neg else 0.0
        x_points.append(fpr)
        y_points.append(tpr)
    x_points.append(1.0)
    y_points.append(1.0)
    return pd.DataFrame({"x": x_points, "y": y_points})


def render_pr_roc_section(
    run_key: str,
    y_true: pd.Series,
    classes: List[str],
    score_data: Dict[str, pd.Series],
    is_multiclass: bool,
    display_labels: Dict[str, str],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    with st.expander("PR / ROC curves", expanded=False):
        render_tooltip("classification.pr_roc", tooltips)
        if not score_data:
            st.info("Score columns not found. Provide y_score or per-class score columns.")
            return

        curve_type = st.radio(
            "Curve type",
            ["Precision-Recall", "ROC"],
            horizontal=True,
            key=build_control_key(run_key, "classification_curve_type"),
        )

        y_true_binary: pd.Series
        y_score = None
        legend_label = ""

        if is_multiclass:
            class_options = [display_labels[label] for label in classes]
            macro_available = all(label in score_data for label in classes)
            if macro_available:
                class_options.append("Macro-average")
            selection = st.selectbox(
                "Class to plot",
                class_options,
                key=build_control_key(run_key, "classification_curve_class"),
            )
            if selection == "Macro-average":
                stacked_true = []
                stacked_scores = []
                for label in classes:
                    if label not in score_data:
                        continue
                    stacked_true.append((y_true == label).astype(int))
                    stacked_scores.append(score_data[label])
                if not stacked_true:
                    st.info("Scores missing for macro-average computation.")
                    return
                y_true_binary = pd.concat(stacked_true, ignore_index=True)
                y_score = pd.concat(stacked_scores, ignore_index=True)
                legend_label = "Macro-average"
            else:
                selected_label = next(
                    key for key, value in display_labels.items() if value == selection
                )
                if selected_label not in score_data:
                    st.info("Scores missing for the selected class.")
                    return
                y_true_binary = (y_true == selected_label).astype(int)
                y_score = score_data[selected_label]
                legend_label = selection
        else:
            display_options = [display_labels[label] for label in classes]
            default_index = 1 if len(display_options) > 1 else 0
            selection = st.selectbox(
                "Positive class",
                display_options,
                index=default_index,
                key=build_control_key(run_key, "classification_positive_class"),
            )
            positive_label = next(
                key for key, value in display_labels.items() if value == selection
            )
            if positive_label in score_data:
                y_score = score_data[positive_label]
            elif "__binary__" in score_data:
                y_score = score_data["__binary__"]
            else:
                st.info("Probability column not found. Provide y_score or score_<class>.")
                return
            y_true_binary = (y_true == positive_label).astype(int)
            legend_label = selection

        if y_score is None:
            st.info("Scores missing for the selected option.")
            return

        curve_df = compute_curve_points(y_true_binary, y_score, curve_type)
        if curve_df is None:
            st.info("Not enough positives/negatives to compute the curve.")
            return

        axis_labels = (
            ("Recall", "Precision") if curve_type == "Precision-Recall" else ("FPR", "TPR")
        )
        fig = px.area(
            curve_df,
            x="x",
            y="y",
            labels={"x": axis_labels[0], "y": axis_labels[1]},
        )
        fig.update_traces(line_color=get_series_color(palette, "true"))
        fig.update_layout(title=legend_label)
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)


def render_classification_results(
    run_key: str,
    experiment_id: str,
    target: Optional[str],
    artifacts: Dict[str, object],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    predictions = artifacts.get("predictions")
    if predictions is None:
        st.info("Predictions not available for this experiment.")
        return

    if not {"y_true", "y_pred"}.issubset(predictions.columns):
        st.warning("predictions.parquet must include y_true and y_pred columns.")
        return

    y_true = normalize_label_series(predictions["y_true"])
    y_pred = normalize_label_series(predictions["y_pred"])
    classes = get_ordered_classes(y_true, y_pred)
    if not classes:
        st.info("No class labels found in predictions.")
        return

    label_map = artifacts.get("label_map")
    display_labels = {label: map_label_for_display(label, label_map) for label in classes}
    is_multiclass = len(classes) > 2

    render_tooltip("classification.results_tab", tooltips)
    render_class_distribution_section(
        run_key, y_true, y_pred, classes, display_labels, tooltips, palette
    )
    render_confusion_matrix_section(
        run_key, y_true, y_pred, classes, display_labels, tooltips, palette
    )
    metrics_df, summaries = calculate_classification_metrics(y_true, y_pred, classes)
    render_per_class_metrics_section(
        run_key, metrics_df, summaries, display_labels, tooltips, palette
    )
    score_data = extract_score_series(predictions)
    render_pr_roc_section(
        run_key, y_true, classes, score_data, is_multiclass, display_labels, tooltips, palette
    )

    if is_multiclass:
        render_tooltip("classification.multiclass_note", tooltips)
        st.info(
            f"Multiclass target detected ({len(classes)} classes). "
            "Curves are computed one-vs-rest per class."
        )


def render_data_tab(
    run_key: str,
    dataset: Optional[pd.DataFrame],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    st.header("Data")
    data_tabs = st.tabs(["Dataset", "Data Explorer"])
    with data_tabs[0]:
        render_dataset_table(dataset, tooltips)
    with data_tabs[1]:
        render_data_explorer(run_key, dataset, tooltips, palette)


def render_classification_tab(
    run_key: str,
    run_dir: Path,
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    st.header("Classification")
    render_tooltip("classification.screen", tooltips)
    classification_dir = run_dir / "classification"
    if not classification_dir.exists():
        st.info("No classification outputs found for this run.")
        return

    results_table = load_classification_results_table(classification_dir)
    class_tabs = st.tabs(["Data", "Results"])
    with class_tabs[0]:
        if results_table is None or results_table.empty:
            st.info("results_table.parquet not found yet.")
        else:
            st.dataframe(results_table)
    with class_tabs[1]:
        metadata = build_experiment_metadata(classification_dir, results_table)
        target, experiment_id = render_classification_selectors(run_key, metadata, tooltips)
        if not experiment_id:
            return
        artifacts = load_experiment_artifacts(classification_dir, experiment_id)
        render_classification_results(run_key, experiment_id, target, artifacts, tooltips, palette)


def _build_group_color_map(rule_meta: Dict[str, object], palette: Dict[str, Dict[str, object]]) -> Dict[str, str]:
    positive = str(rule_meta.get("positive_label", "Group A"))
    negative = str(rule_meta.get("negative_label", "Group B"))
    color_map = {
        positive: get_series_color(palette, "true"),
        negative: get_series_color(palette, "pred"),
    }
    return color_map


def _locate_row(dataset: Optional[pd.DataFrame], row_id: str, row_id_column: Optional[str]) -> Optional[pd.Series]:
    if dataset is None or row_id_column is None:
        return None
    if row_id_column == "__index__":
        mask = dataset.index.astype(str) == row_id
        if mask.any():
            return dataset.loc[mask].iloc[0]
        return None
    if row_id_column not in dataset.columns:
        return None
    mask = dataset[row_id_column].astype(str) == row_id
    if mask.any():
        return dataset.loc[mask].iloc[0]
    return None


def _render_embeddings_chart(
    plot_df: pd.DataFrame,
    plot_type: str,
    color_map: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
) -> None:
    if plot_type == "Scatter":
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="group_label",
            hover_data=["row_id"],
            color_discrete_map=color_map,
        )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)
        return

    if plot_type == "Centroids overlay":
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="group_label",
            hover_data=["row_id"],
            opacity=0.4,
            color_discrete_map=color_map,
        )
        centroids = plot_df.groupby("group_label")[["x", "y"]].mean().reset_index()
        for _, row in centroids.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["x"]],
                    y=[row["y"]],
                    mode="markers+text",
                    text=[row["group_label"]],
                    textposition="top center",
                    marker_symbol="x",
                    marker_size=18,
                    marker_color=color_map.get(row["group_label"], get_series_color(palette, "true")),
                    name=f"{row['group_label']} centroid",
                )
            )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)
        return

    if plot_type == "KDE":
        fig = px.density_contour(
            plot_df,
            x="x",
            y="y",
            color="group_label",
            color_discrete_map=color_map,
            hover_data=["row_id"],
            contours_coloring="fill",
        )
        apply_plot_theme(fig, palette)
        st.plotly_chart(fig, use_container_width=True)
        return


def render_text_inspector(
    run_key: str,
    dataset: Optional[pd.DataFrame],
    manifest: Dict[str, object],
    row_ids: List[str],
    tooltips: Dict[str, str],
):
    with st.expander("Text inspector", expanded=False):
        render_tooltip("embeddings.text_inspector", tooltips)
        if not row_ids:
            st.info("Select a projection above to load row ids.")
            return
        selector = st.selectbox(
            "Row identifier",
            row_ids,
            key=build_control_key(run_key, "embeddings_row_selector"),
        )
        render_tooltip("embeddings.row_selector", tooltips)
        row_id_column = manifest.get("row_id_column")
        snapshot = _locate_row(dataset, selector, row_id_column if isinstance(row_id_column, str) else None)
        if snapshot is None:
            st.info("Row not found in base_modificada. Verify the manifest mappings.")
            return
        text_column = manifest.get("text_column")
        if isinstance(text_column, str) and text_column in snapshot.index:
            st.text_area("Document text", str(snapshot[text_column]), height=200)
        else:
            st.write("Text column not available in dataset.")
        mapping = infer_initial_mapping(dataset) if dataset is not None else {}
        law_column = mapping.get("law")
        if law_column and law_column in snapshot.index:
            st.write(f"**Law:** {snapshot[law_column]}")
        trigger_values = []
        for key in BINARY_KEYS:
            column = mapping.get(key)
            if column and column in snapshot.index:
                trigger_values.append(
                    {
                        "Trigger": CONCEPTS[key]["label"],
                        "Value": snapshot[column],
                    }
                )
        if trigger_values:
            st.table(pd.DataFrame(trigger_values))


def render_embeddings_tab(
    run_key: str,
    run_dir: Path,
    dataset: Optional[pd.DataFrame],
    tooltips: Dict[str, str],
    palette: Dict[str, Dict[str, object]],
):
    st.header("Embeddings")
    render_tooltip("embeddings.screen", tooltips)
    manifest = load_embeddings_manifest(run_dir)
    if not manifest:
        st.info("No embeddings outputs found for this run.")
        return
    models = manifest.get("models") or []
    if not models:
        st.info("Embeddings manifest does not list any models.")
        return

    model_labels = [f"{entry.get('model_name', 'Unknown')} ({entry.get('embedding_id')})" for entry in models]
    selected_model_label = st.selectbox(
        "Embedding model",
        model_labels,
        key=build_control_key(run_key, "embeddings_model_select"),
    )
    render_tooltip("embeddings.selector_model", tooltips)
    selected_model = models[model_labels.index(selected_model_label)]

    projections = selected_model.get("projections") or {}
    if not projections:
        st.info("No projections registered for the selected model.")
        return
    projection_names = sorted(projections.keys())
    selected_projection = st.selectbox(
        "Projection type",
        projection_names,
        key=build_control_key(run_key, "embeddings_projection_select"),
    )
    render_tooltip("embeddings.selector_projection", tooltips)

    rules = load_embedding_rules(run_dir)
    if not rules:
        st.info("No comparison rules found. Ensure rules.json exists in embeddings/groups.")
        return
    rule_labels = [f"{rule.get('description', rule.get('rule_id'))}" for rule in rules]
    selected_rule_label = st.selectbox(
        "Comparison rule",
        rule_labels,
        key=build_control_key(run_key, "embeddings_rule_select"),
    )
    render_tooltip("embeddings.selector_rule", tooltips)
    selected_rule = rules[rule_labels.index(selected_rule_label)]

    plot_type = st.selectbox(
        "Plot type",
        EMBEDDING_PLOT_TYPES,
        key=build_control_key(run_key, "embeddings_plot_type"),
    )
    render_tooltip("embeddings.selector_plot", tooltips)

    projection_rel_path = projections.get(selected_projection)
    if not projection_rel_path:
        st.info("Projection file not listed in manifest.")
        return
    projection_df = load_projection_dataframe(run_dir, projection_rel_path)
    if projection_df is None or projection_df.empty:
        st.info("Projection coordinates could not be loaded.")
        return

    labels_rel_path = selected_rule.get("labels_path")
    if not isinstance(labels_rel_path, str):
        st.info("Rule labels path missing in rules.json.")
        return
    labels_df = load_rule_labels(run_dir, labels_rel_path)
    if labels_df is None or labels_df.empty:
        st.info("Labels file is empty or missing required columns.")
        return

    merged = projection_df.merge(labels_df, on="row_id", how="inner")
    merged = merged.dropna(subset=["x", "y", "group_label"])
    if merged.empty:
        st.info("No overlapping rows found between projection and labels.")
        return

    color_map = _build_group_color_map(selected_rule, palette)
    with st.expander("Projection explorer", expanded=True):
        render_tooltip("embeddings.plot_section", tooltips)
        _render_embeddings_chart(merged, plot_type, color_map, palette)
        stats = load_embedding_stats(
            run_dir,
            str(selected_model.get("embedding_id")),
            selected_projection,
            str(selected_rule.get("rule_id")),
        )
        if stats:
            render_tooltip("embeddings.stats_note", tooltips)
            distance = stats.get("centroid_distance")
            sizes = stats.get("group_sizes", {})
            st.caption(
                f"Centroid distance: {distance:.4f} | Group sizes: {sizes}"
                if isinstance(distance, (int, float))
                else f"Group sizes: {sizes}"
            )
    unique_rows = merged["row_id"].dropna().astype(str).unique().tolist()
    render_text_inspector(run_key, dataset, manifest, unique_rows, tooltips)


def render_placeholder_tab(title: str):
    st.header(title)
    st.info("This section will be implemented in a future milestone.")


def main() -> None:
    st.set_page_config(page_title="Gray Areas Viewer", layout="wide")
    palette = load_palette(Path("theme/palette.yml"))
    tooltips = load_tooltips(Path("copy/tooltips_en.yml"))

    runs_root = Path("runs")
    run_directories = discover_run_directories(runs_root)
    if not run_directories:
        st.warning("No runs found. Execute the Runner before opening the Viewer.")
        return

    run_names = [run.name for run in run_directories]
    manifest_loader = lambda name: load_manifest(runs_root / name)
    default_run = run_names[0]
    selected_name, manifest = render_sidebar(run_names, default_run, manifest_loader, tooltips)

    selected_run = runs_root / selected_name
    dataset = load_base_dataset(selected_run)

    tabs = st.tabs(["Data", "Classification", "Embeddings", "Sensitivity", "Zero-shot"])
    with tabs[0]:
        render_data_tab(selected_name, dataset, tooltips, palette)
    with tabs[1]:
        render_classification_tab(selected_name, selected_run, tooltips, palette)
    with tabs[2]:
        render_embeddings_tab(selected_name, selected_run, dataset, tooltips, palette)
    with tabs[3]:
        render_placeholder_tab("Sensitivity")
    with tabs[4]:
        render_placeholder_tab("Zero-shot")


if __name__ == "__main__":
    main()

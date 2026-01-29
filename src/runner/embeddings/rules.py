# Labeling rules for embeddings comparisons.
"""Derive group assignments (A vs B) for each trigger so projections can be compared."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

TRIGGER_ALIASES: Dict[str, List[str]] = {
    "relevance": ["relevance", "relevancia"],
    "completeness": ["completeness", "completitud"],
    "differential_regime": ["differential_regime", "differential regime", "regimen diferencial"],
    "discretionality": ["discretionality", "discrecionalidad"],
}
INTERPRETABILITY_ALIASES = ["interpretability", "interpretabilidad"]


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _find_column(dataframe: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {_normalize(column): column for column in dataframe.columns}
    for candidate in candidates:
        token = _normalize(candidate)
        if token in normalized:
            return normalized[token]
    return None


@dataclass
class RuleDefinition:
    """Metadata describing how a rule groups rows."""

    rule_id: str
    description: str
    column: Optional[str]
    positive_label: str
    negative_label: str


@dataclass
class RuleResult:
    """Actual labels per row for a specific rule."""

    definition: RuleDefinition
    labels: pd.DataFrame  # columns: row_id, group_label, group_value


def _build_binary_rule(
    dataframe: pd.DataFrame,
    column: str,
    rule_id: str,
    positive_label: str,
    negative_label: str,
    row_ids: pd.Series,
) -> RuleResult:
    series = pd.to_numeric(dataframe[column], errors="coerce")
    group_values = (series.fillna(0) != 0).astype(int)
    labels = pd.DataFrame(
        {
            "row_id": row_ids.astype(str),
            "group_value": group_values,
            "group_label": group_values.map({1: positive_label, 0: negative_label}),
        }
    )
    definition = RuleDefinition(
        rule_id=rule_id,
        description=f"{column} != 0 vs == 0",
        column=column,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    return RuleResult(definition=definition, labels=labels)


def _build_interpretability_rule(
    dataframe: pd.DataFrame,
    column: str,
    row_ids: pd.Series,
) -> RuleResult:
    series = pd.to_numeric(dataframe[column], errors="coerce")
    group_values = series.apply(lambda value: 1 if value == 1 else 0)
    group_values = group_values.fillna(0).astype(int)
    labels = pd.DataFrame(
        {
            "row_id": row_ids.astype(str),
            "group_value": group_values,
            "group_label": group_values.map(
                {1: "Interpretability = 1", 0: "Interpretability >= 2"}
            ),
        }
    )
    definition = RuleDefinition(
        rule_id="interpretability_split",
        description="Interpretability == 1 vs Interpretability in {2, 3}",
        column=column,
        positive_label="Interpretability = 1",
        negative_label="Interpretability >= 2",
    )
    return RuleResult(definition=definition, labels=labels)


def generate_rules(dataframe: pd.DataFrame, row_ids: pd.Series) -> List[RuleResult]:
    """Build every rule definition alongside the corresponding labels.

    Args:
        dataframe (pandas.DataFrame): Dataset used for the current run.
        row_ids (pandas.Series): Identifier per row to preserve alignment.

    Returns:
        list[RuleResult]: Rules available for embeddings comparisons.

    Viewer:
        The Embeddings tab loads the saved labels to color and filter projections.
    """

    rules: List[RuleResult] = []
    for trigger_key, aliases in TRIGGER_ALIASES.items():
        column = _find_column(dataframe, aliases)
        if not column:
            continue
        rule_id = f"{trigger_key}_binary"
        positive = f"{trigger_key.replace('_', ' ').title()} != 0"
        negative = f"{trigger_key.replace('_', ' ').title()} = 0"
        rules.append(
            _build_binary_rule(
                dataframe=dataframe,
                column=column,
                rule_id=rule_id,
                positive_label=positive,
                negative_label=negative,
                row_ids=row_ids,
            )
        )

    interpretability_column = _find_column(dataframe, INTERPRETABILITY_ALIASES)
    if interpretability_column:
        rules.append(
            _build_interpretability_rule(
                dataframe=dataframe,
                column=interpretability_column,
                row_ids=row_ids,
            )
        )

    return rules

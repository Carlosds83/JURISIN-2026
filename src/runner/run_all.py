# Runner entry point for orchestrating dataset ingestion and per-run scaffolding.
"""Utility script to bootstrap a minimal runner execution."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    from embeddings.run_embeddings import run_embeddings_pipeline
except Exception as error:  # pragma: no cover - optional dependency warning
    run_embeddings_pipeline = None  # type: ignore[assignment]
    EMBEDDINGS_IMPORT_ERROR = error
else:  # pragma: no cover - track error-free import
    EMBEDDINGS_IMPORT_ERROR = None

try:
    from classification.run_classification import run_classification_pipeline
except Exception as error:  # pragma: no cover - optional dependency warning
    run_classification_pipeline = None  # type: ignore[assignment]
    CLASSIFICATION_IMPORT_ERROR = error
else:  # pragma: no cover - track error-free import
    CLASSIFICATION_IMPORT_ERROR = None


def parse_arguments(raw_args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments required to execute the runner.

    Args:
        raw_args (list[str] | None): Optional override for CLI arguments (used in tests).

    Returns:
        argparse.Namespace: Parsed namespace with dataset_path y sheet_name.

    Viewer:
        Not applicable (Runner only).
    """

    parser = argparse.ArgumentParser(
        description="Minimal runner que prepara corridas a partir de CSV, Parquet o Excel."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Ruta al dataset local en formato CSV, Parquet o Excel (.xlsx/.xls).",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default=None,
        help="Nombre de hoja a cargar si el archivo es Excel (por defecto la primera).",
    )
    return parser.parse_args(raw_args)


def load_dataset(dataset_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV, Parquet, or Excel dataset from disk using pandas.

    Args:
        dataset_path (Path): Path to the dataset file provided by the user.
        sheet_name (str | None): Optional Excel sheet override (None loads first sheet).

    Returns:
        pandas.DataFrame: Loaded dataset ready for validation.

    Viewer:
        Not applicable (Runner only).
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        print(f"[Runner] Cargando dataset CSV desde {dataset_path} ...")
        # Muchos datasets fiscales en español incluyen caracteres acentuados, así que
        # hacemos fallback a Latin-1 cuando UTF-8 falla para evitar bloqueos.
        try:
            return pd.read_csv(dataset_path, encoding="utf-8")
        except UnicodeDecodeError:
            print(
                "[Runner] Advertencia: no se pudo leer como UTF-8, "
                "reintentando con Latin-1."
            )
            try:
                return pd.read_csv(dataset_path, encoding="latin-1")
            except UnicodeDecodeError as latin_error:
                raise ValueError(
                    "No se pudo leer el CSV con codificaciones UTF-8 ni Latin-1. "
                    "Verifica la codificación del archivo."
                ) from latin_error
    if suffix in {".xlsx", ".xls"}:
        print(f"[Runner] Cargando dataset Excel desde {dataset_path} ...")
        # La base maestra proviene de Excel entregado por la investigadora, así que
        # soportamos directamente hojas (.xlsx/.xls) respetando el nombre si se indica.
        excel_kwargs = {"sheet_name": sheet_name or 0}
        if suffix == ".xlsx":
            excel_kwargs["engine"] = "openpyxl"
        return pd.read_excel(dataset_path, **excel_kwargs)
    if suffix in {".parquet", ".pq"}:
        print(f"[Runner] Cargando dataset Parquet desde {dataset_path} ...")
        return pd.read_parquet(dataset_path)

    supported = ".csv, .parquet, .xlsx o .xls"
    raise ValueError(f"Formato no soportado: {suffix}. Solo se aceptan {supported}.")


def ensure_non_empty(dataset: pd.DataFrame) -> None:
    """Validate that the dataset contains at least one row.

    Args:
        dataset (pandas.DataFrame): Dataset that must be validated.

    Viewer:
        Not applicable (Runner only).
    """

    if dataset.empty:
        raise ValueError("El dataset esta vacio. Usa un archivo con al menos una fila.")


def create_run_directory(base_dir: Path = Path("runs")) -> tuple[Path, str]:
    """Create a unique run directory following the required naming convention.

    Args:
        base_dir (Path): Base path where run folders should be stored.

    Returns:
        tuple[Path, str]: Path to the run directory and the ISO timestamp used.

    Viewer:
        Not applicable (Runner only).
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_id = uuid.uuid4().hex[:6]
    run_dir = base_dir / f"run_{timestamp}_{short_id}"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=False)
    print(f"[Runner] Carpeta de corrida creada en {run_dir}")
    return run_dir, timestamp


def augment_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create a copy of the dataset including the placeholder columns.

    Args:
        dataset (pandas.DataFrame): Original dataset provided by the user.

    Returns:
        pandas.DataFrame: Dataset copy including the required placeholder columns.

    Viewer:
        Base modificada en la pantalla Data (tabla principal).
    """

    augmented = dataset.copy()
    for placeholder in ("complexity_index", "risk_index"):
        augmented[placeholder] = pd.NA
    return augmented


def persist_dataset(dataset: pd.DataFrame, run_dir: Path) -> Path:
    """Save the augmented dataset under the data/ folder as Parquet.

    Args:
        dataset (pandas.DataFrame): Dataset to store.
        run_dir (Path): Base directory for the current run.

    Returns:
        Path: Output file path for the stored dataset.

    Viewer:
        Data screen reads this Parquet as base_modificada.
    """

    output_path = run_dir / "data" / "base_modificada.parquet"
    try:
        dataset.to_parquet(output_path, index=False)
    except (ImportError, ValueError) as parquet_error:
        raise RuntimeError(
            "No fue posible guardar el Parquet. Instala pyarrow o fastparquet."
        ) from parquet_error

    print(f"[Runner] Dataset guardado en {output_path}")
    return output_path


def write_manifest(
    run_dir: Path,
    timestamp: str,
    dataset_path: Path,
    row_count: int,
    script_name: str,
) -> Path:
    """Persist the manifest.json file containing metadata for the run.

    Args:
        run_dir (Path): Directory where the manifest should live.
        timestamp (str): Timestamp associated with the run name.
        dataset_path (Path): Path to the input dataset.
        row_count (int): Number of rows in the dataset.
        script_name (str): Name of the script executed.

    Returns:
        Path: Full path to the manifest file.

    Viewer:
        Settings screen usa este manifest para listar corridas.
    """

    manifest = {
        "timestamp": timestamp,
        "input_dataset": str(dataset_path),
        "row_count": row_count,
        "python_version": platform.python_version(),
        "script_name": script_name,
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Runner] manifest.json creado en {manifest_path}")
    return manifest_path


def main(raw_args: Optional[List[str]] = None) -> None:
    """Entrypoint that wires argument parsing, IO, and manifest creation.

    Args:
        raw_args (list[str] | None): Optional override for CLI arguments.

    Viewer:
        Not applicable (Runner only).
    """

    args = parse_arguments(raw_args)
    dataset_path = args.dataset_path.expanduser().resolve()

    dataset = load_dataset(dataset_path, sheet_name=args.sheet_name)
    ensure_non_empty(dataset)

    run_dir, timestamp = create_run_directory()
    augmented_dataset = augment_dataset(dataset)
    persist_dataset(augmented_dataset, run_dir)

    embeddings_help = "Install required packages via: pip install sentence-transformers torch umap-learn scikit-learn pyarrow"
    if run_embeddings_pipeline is None:
        print(
            "[Runner] Embeddings pipeline not available. "
            f"{EMBEDDINGS_IMPORT_ERROR} | {embeddings_help}"
        )
    else:
        print("[Runner] Embeddings started...")
        try:
            run_embeddings_pipeline(dataset=augmented_dataset, run_dir=run_dir)
        except RuntimeError as embeddings_error:
            print(f"[Runner] Embeddings skipped: {embeddings_error} | {embeddings_help}")
        except Exception as embeddings_error:
            print(f"[Runner] Embeddings failed: {embeddings_error}")
            raise
        else:
            print(f"[Runner] Embeddings finished. Outputs saved to {run_dir / 'embeddings'}")

    write_manifest(
        run_dir=run_dir,
        timestamp=timestamp,
        dataset_path=dataset_path,
        row_count=len(augmented_dataset),
        script_name=Path(__file__).name,
    )

    if run_classification_pipeline is None:
        print(
            "[Runner] Classification pipeline not available. "
            f"{CLASSIFICATION_IMPORT_ERROR}"
        )
    else:
        try:
            run_classification_pipeline(dataset=dataset, run_dir=run_dir)
        except RuntimeError as classification_error:
            print(f"[Runner] Classification skipped: {classification_error}")
        except Exception as classification_error:
            print(f"[Runner] Classification failed: {classification_error}")
            raise

    print("[Runner] Ejecucion finalizada correctamente.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # pragma: no cover - basic CLI reporting
        print(f"[Runner] Error: {error}", file=sys.stderr)
        sys.exit(1)

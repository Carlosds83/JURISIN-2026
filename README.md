# Gray Areas – NLP Exploration Tool

This project implements an exploratory NLP system to analyze and visualize
“gray area triggers” in Colombian Free Trade Zone (Zona Franca) tax regulation.

The system is intentionally split into two independent components:

- **Runner**: executes all computational pipelines (embeddings, classification,
  projections, sensitivity analysis, zero-shot experiments) and stores results
  on disk.
- **Viewer**: a user-friendly interface that only loads and visualizes previously
  generated results. The Viewer never executes heavy computations.

## Philosophy
This software is designed for:
- transparency and traceability
- reproducible experimentation
- exploratory analysis rather than automated legal decision-making

## Repository structure
- `src/runner/` – computation and pipelines
- `src/viewer/` – visualization-only interface
- `runs/` – versioned outputs from each run
- `PROMPT_MAESTRO_GRAY_AREAS.md` – architectural and design rules for development

## Status
The project is under active development. Components are implemented
incrementally and validated step by step.

## Runner dependencies
The Runner expects the following Python packages to be installed in the active
environment:

- `pandas`, `pyarrow`, `openpyxl`
- `scikit-learn`
- `torch`
- `transformers`

Install them with `pip install pandas pyarrow openpyxl scikit-learn torch transformers`
before executing `python src/runner/run_all.py`.

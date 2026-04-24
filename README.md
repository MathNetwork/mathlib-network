# mathlib-network

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19746468.svg)](https://doi.org/10.5281/zenodo.19746468) [![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-MathNetwork%2FMathlibGraph-yellow)](https://huggingface.co/datasets/MathNetwork/MathlibGraph)

Python pipeline for extracting and analyzing the dependency graph of Lean 4's [Mathlib](https://github.com/leanprover-community/mathlib4) as a multi-layer network (declaration, module, namespace).

The pipeline:

1. **Extracts** the dependency graph from a local Mathlib build using `lean4export` and `lean-training-data` (see `src/scripts/extract.sh`), producing node and edge CSVs.
2. **Parses** raw output into a canonical graph (`src/parser/`).
3. **Analyzes** the graph at three layers — declaration, module, namespace — computing degree distributions, PageRank/HITS, betweenness, communities, cascades, and robustness (`src/analysis/`).
4. **Generates figures** and summary tables (`src/plots/`, `src/scripts/`).

## Data

The extracted graph (308,129 declarations, 8.4M edges, 7,563 modules) is published on HuggingFace: [MathNetwork/MathlibGraph](https://huggingface.co/datasets/MathNetwork/MathlibGraph). You don't need to re-run extraction to use the analysis scripts — they can load the graph directly from HuggingFace.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full analysis pipeline on the published dataset:

```bash
python -m src.main
```

Individual analyses live under `src/scripts/` and can be run directly, e.g.:

```bash
python -m src.scripts.analyze_module_depth
python -m src.scripts.run_pagerank_community
```

To re-extract the graph from a local Mathlib build:

```bash
bash src/scripts/extract.sh --local --output ./output
```

## Package Layout

- `src/parser/` — extract graph from Lean 4 sources
- `src/analysis/structure/` — degree distribution, PageRank/HITS, betweenness, community detection
- `src/analysis/dynamics/` — cascade and robustness analysis
- `src/plots/` — figure generation
- `src/scripts/` — standalone analyses and pipeline entry points
- `src/tests/` — pytest suite

## License

MIT

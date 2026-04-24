# mathlib-network

[![DOI](https://zenodo.org/badge/DOI/PENDING.svg)](https://doi.org/PENDING) [![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-MathNetwork%2FMathlibGraph-yellow)](https://huggingface.co/datasets/MathNetwork/MathlibGraph)

Python pipeline for extracting and analyzing the dependency graph of Lean 4's Mathlib as a multi-layer network (declaration, module, namespace).

## Paper

This repository contains the code accompanying:

**The Network Structure of Mathlib: Software Engineering vs. Mathematical Dependencies**
Xinze Li, Nanyun Peng, Simone Severini, Patrick Shafto (2026)

arXiv: [link TBD]

## Data

The extracted graph dataset (308,129 declarations, 8.4M edges, 7,563 modules) is available on HuggingFace:

[MathNetwork/MathlibGraph](https://huggingface.co/datasets/MathNetwork/MathlibGraph)

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

See scripts in `src/` for extraction, analysis, and figure generation.

## Citation

If you use this code, please cite:

```bibtex
@article{mathlib_network_2026,
  author  = {Li, Xinze and Peng, Nanyun and Severini, Simone and Shafto, Patrick},
  title   = {The Network Structure of Mathlib: Software Engineering vs. Mathematical Dependencies},
  year    = {2026}
}

@software{mathlib_network_code_2026,
  author    = {Li, Xinze and Peng, Nanyun and Severini, Simone and Shafto, Patrick},
  title     = {mathlib-network: Code for "The Network Structure of Mathlib"},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {PENDING},
  url       = {https://doi.org/PENDING}
}
```

DOI will be filled in after Zenodo archives v1.0.0 release.

## License

MIT

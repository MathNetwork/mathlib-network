"""TDD tests for the cache-based replot system.

Design:
- src/plots/cache/ stores intermediate data as CSV/JSON files
- Each plot module exposes a pure rendering function (data in, figure out)
- replot_all.py reads cache and calls rendering functions
- Changing font sizes only requires replot_all.py (seconds, no recomputation)
"""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent  # src/
CACHE_DIR = ROOT / "plots" / "cache"
FIGURES_DIR = ROOT.parent / "paper" / "figures"


# ── Test 1: Cache directory and manifest ────────────────────────────
class TestCacheStructure:
    def test_cache_dir_exists(self):
        """Cache directory must exist."""
        assert CACHE_DIR.is_dir(), f"Missing cache directory: {CACHE_DIR}"

    def test_manifest_exists(self):
        """A manifest file lists all cached datasets and their target figures."""
        manifest_path = CACHE_DIR / "manifest.json"
        assert manifest_path.exists(), f"Missing {manifest_path}"
        manifest = json.loads(manifest_path.read_text())
        assert isinstance(manifest, dict)
        # Each entry maps a cache file to a list of figure filenames it produces
        assert len(manifest) > 0, "Manifest is empty"


# ── Test 2: replot_all.py exists and works from cache ───────────────
class TestReplotAll:
    def test_replot_module_exists(self):
        """replot_all.py must exist in src/plots/."""
        replot_path = ROOT / "plots" / "replot_all.py"
        assert replot_path.exists(), f"Missing {replot_path}"

    def test_replot_has_main(self):
        """replot_all.py must have a main() function."""
        replot_path = ROOT / "plots" / "replot_all.py"
        source = replot_path.read_text()
        assert "def main(" in source, "replot_all.py missing main() function"

    def test_replot_does_not_import_heavy_libs(self):
        """replot_all.py must NOT import datasets, networkx, or community
        (those are computation libs, not needed for pure replotting)."""
        replot_path = ROOT / "plots" / "replot_all.py"
        source = replot_path.read_text()
        for lib in ["from datasets import", "import networkx", "import community"]:
            assert lib not in source, (
                f"replot_all.py imports '{lib}' — it should only do rendering, "
                f"not computation"
            )


# ── Test 3: Cache files cover all 24 figures ────────────────────────
# All figures that should be reproducible from cache
EXPECTED_FIGURES = sorted([
    # "containment_curve.pdf" — hardcoded in replot_all.py, no cache needed
    "dag_structure.pdf",
    "degree_distribution.pdf",
    "namespace_heatmap_raw.pdf",
    "namespace_heatmap_tr.pdf",
    "module_centrality_indeg_pr.pdf",
    "module_centrality_indeg_betw.pdf",
    "module_centrality_betw_pr.pdf",
    "module_robustness_curve.pdf",
    "ns_degree_distribution.pdf",
    "ns_dag_structure.pdf",
    "ns_centrality_indeg_pr.pdf",
    "ns_centrality_indeg_betw.pdf",
    "ns_centrality_betw_pr.pdf",
    "ns_robustness_curve.pdf",
    "community_module_heatmap.pdf",
    "community_ns_heatmap.pdf",
    "community_decl_heatmap.pdf",
    "thm_degree_distribution.pdf",
    "thm_dag_structure.pdf",
    "thm_centrality_indeg_pr.pdf",
    "thm_centrality_indeg_betw.pdf",
    "thm_centrality_betw_pr.pdf",
    "thm_robustness_curve.pdf",
])


class TestCacheCoverage:
    def test_manifest_covers_all_figures(self):
        """The manifest must list cache entries that collectively produce
        every expected figure."""
        manifest_path = CACHE_DIR / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("manifest.json not yet created")
        manifest = json.loads(manifest_path.read_text())

        # Collect all figures mentioned in manifest
        covered = set()
        for cache_file, info in manifest.items():
            for fig in info.get("figures", []):
                covered.add(fig)

        missing = set(EXPECTED_FIGURES) - covered
        assert not missing, f"Figures not covered by cache: {sorted(missing)}"

    def test_all_cache_files_exist(self):
        """Every cache file listed in the manifest must actually exist."""
        manifest_path = CACHE_DIR / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("manifest.json not yet created")
        manifest = json.loads(manifest_path.read_text())

        missing = []
        for cache_file in manifest:
            if not (CACHE_DIR / cache_file).exists():
                missing.append(cache_file)
        assert not missing, f"Cache files listed but missing: {missing}"

"""TDD tests for unified plot style.

These tests enforce that:
1. plot_style.py exports canonical font-size constants.
2. All plotting scripts use these constants instead of hardcoded fontsize values.
3. The duplicate plot_style.py in src/scripts/ is eliminated (single source of truth).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # src/

# ── Canonical font-size spec ────────────────────────────────────────
EXPECTED_FONT_SIZES = {
    "TITLE_FS": 18,
    "LABEL_FS": 16,
    "TICK_FS": 14,
    "LEGEND_FS": 14,
    "ANNOT_FS": 11,
}

# All plotting scripts that generate paper figures
PLOT_SCRIPTS = sorted(
    list((ROOT / "plots").glob("*.py"))
    + list((ROOT / "scripts").glob("*.py"))
    + list((ROOT / "analysis").rglob("*.py")),
)
# Exclude __init__.py, plot_style.py itself, and non-plotting utilities
PLOT_SCRIPTS = [
    p
    for p in PLOT_SCRIPTS
    if p.name not in ("__init__.py", "plot_style.py", "conftest.py")
    and not p.name.startswith("test_")
]


# ── Test 1: plot_style.py exports the canonical constants ───────────
class TestPlotStyleExports:
    def test_constants_exist_and_match(self):
        """plot_style.py must export TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, ANNOT_FS."""
        import importlib
        import sys

        style_path = ROOT / "plots" / "plot_style.py"
        assert style_path.exists(), f"Missing {style_path}"

        spec = importlib.util.spec_from_file_location("plot_style", style_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plot_style"] = mod
        spec.loader.exec_module(mod)

        for name, expected in EXPECTED_FONT_SIZES.items():
            actual = getattr(mod, name, None)
            assert actual is not None, f"plot_style.py missing constant {name}"
            assert actual == expected, (
                f"plot_style.{name} = {actual}, expected {expected}"
            )

    def test_rcparams_use_constants(self):
        """setup_style() must set rcParams consistent with the exported constants."""
        import importlib
        import sys

        style_path = ROOT / "plots" / "plot_style.py"
        spec = importlib.util.spec_from_file_location("plot_style", style_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plot_style"] = mod
        spec.loader.exec_module(mod)

        import matplotlib.pyplot as plt

        mod.setup_style()

        assert plt.rcParams["axes.titlesize"] == EXPECTED_FONT_SIZES["TITLE_FS"]
        assert plt.rcParams["axes.labelsize"] == EXPECTED_FONT_SIZES["LABEL_FS"]
        assert plt.rcParams["xtick.labelsize"] == EXPECTED_FONT_SIZES["TICK_FS"]
        assert plt.rcParams["ytick.labelsize"] == EXPECTED_FONT_SIZES["TICK_FS"]
        assert plt.rcParams["legend.fontsize"] == EXPECTED_FONT_SIZES["LEGEND_FS"]


# ── Test 2: No duplicate plot_style.py ──────────────────────────────
class TestSingleSourceOfTruth:
    def test_no_duplicate_plot_style(self):
        """There should be only ONE plot_style.py (in src/plots/)."""
        copies = list(ROOT.rglob("plot_style.py"))
        assert len(copies) == 1, (
            f"Expected 1 plot_style.py, found {len(copies)}: {copies}"
        )
        assert copies[0] == ROOT / "plots" / "plot_style.py"


# ── Test 3: No hardcoded fontsize in plotting scripts ───────────────
# Pattern: fontsize=<number> where number is a bare int literal
_HARDCODED_FONTSIZE_RE = re.compile(r"fontsize\s*=\s*\d+")
# Pattern: local variable definitions like  title_fs, label_fs, ... = 14, 12, ...
_LOCAL_FS_VAR_RE = re.compile(
    r"(title_fs|label_fs|tick_fs|legend_fs|annot_fs|dag_title_fs|dag_label_fs|dag_tick_fs|cent_title_fs|cent_label_fs|cent_tick_fs)"
    r"\s*[,=]"
)


def _has_matplotlib_usage(source: str) -> bool:
    """Check if a file actually uses matplotlib for plotting."""
    return "matplotlib" in source or "plt." in source or "fontsize" in source


class TestNoHardcodedFontsize:
    def test_no_hardcoded_fontsize_in_plot_scripts(self):
        """No plotting script should contain fontsize=<bare integer>."""
        violations = []
        for script in PLOT_SCRIPTS:
            source = script.read_text()
            if not _has_matplotlib_usage(source):
                continue
            for i, line in enumerate(source.splitlines(), 1):
                # Skip comments
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if _HARDCODED_FONTSIZE_RE.search(line):
                    violations.append(f"  {script.relative_to(ROOT)}:{i}  {line.strip()}")
        assert not violations, (
            "Hardcoded fontsize=<int> found (use constants from plot_style):\n"
            + "\n".join(violations)
        )

    def test_no_local_fontsize_variables(self):
        """No plotting script should define local title_fs/label_fs/tick_fs variables."""
        violations = []
        for script in PLOT_SCRIPTS:
            source = script.read_text()
            if not _has_matplotlib_usage(source):
                continue
            for i, line in enumerate(source.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if _LOCAL_FS_VAR_RE.search(line):
                    violations.append(f"  {script.relative_to(ROOT)}:{i}  {line.strip()}")
        assert not violations, (
            "Local fontsize variables found (use constants from plot_style):\n"
            + "\n".join(violations)
        )

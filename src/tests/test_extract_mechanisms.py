#!/usr/bin/env python3
"""Tests for Lean mechanism extraction pipeline.

Tests the NDJSON parser and validates expected properties of the
extracted language mechanisms from a Mathlib snapshot.
"""

import pytest
import json
import io
from pathlib import Path


# --- Sample NDJSON output (what we expect the Lean script to produce) ---

SAMPLE_OUTPUT = """\
{"type":"meta","key":"lean_version","value":"4.28.0-rc1"}
{"type":"meta","key":"total_constants","value":450000}
{"type":"meta","key":"mathlib_constants","value":308129}
{"type":"kind","name":"theorem","count":150000}
{"type":"kind","name":"definition","count":50000}
{"type":"kind","name":"inductive","count":10000}
{"type":"kind","name":"constructor","count":25000}
{"type":"kind","name":"recursor","count":12000}
{"type":"kind","name":"instance","count":30000}
{"type":"kind","name":"opaque","count":500}
{"type":"kind","name":"axiom","count":100}
{"type":"attribute","name":"simp","count":15000}
{"type":"attribute","name":"ext","count":2000}
{"type":"attribute","name":"instance","count":30000}
{"type":"attribute","name":"to_additive","count":5000}
{"type":"attribute","name":"reducible","count":3000}
{"type":"attribute","name":"norm_num","count":200}
{"type":"attribute","name":"aesop","count":150}
{"type":"class","name":"Add"}
{"type":"class","name":"Mul"}
{"type":"class","name":"Monoid"}
{"type":"class","name":"CommMonoid"}
{"type":"instance","name":"instAddNat","class":"Add"}
{"type":"instance","name":"instMulNat","class":"Mul"}
{"type":"structure_parent","child":"CommMonoid","parent":"Monoid"}
{"type":"structure_parent","child":"Monoid","parent":"MulOneClass"}
{"type":"to_additive_pair","source":"mul_comm","target":"add_comm"}
{"type":"to_additive_pair","source":"mul_one","target":"add_zero"}
{"type":"coercion","name":"Int.ofNat","coe_type":"coe"}
{"type":"coercion","name":"Subtype.val","coe_type":"coe"}
{"type":"coercion","name":"OrderDual.toDual","coe_type":"coe"}
{"type":"deriving_handler","name":"BEq"}
{"type":"deriving_handler","name":"Repr"}
{"type":"deriving_handler","name":"DecidableEq"}
{"type":"deriving_handler","name":"Hashable"}
{"type":"stmt_proof_stats","S_only":1200000,"P_only":3500000,"SP":3700000,"total_decls":308129}
{"type":"module_import","module":"Mathlib.Algebra.Group.Defs","imported":"Mathlib.Algebra.Group.Basic","is_exported":true}
{"type":"module_import","module":"Mathlib.Algebra.Group.Defs","imported":"Mathlib.Data.Int.Defs","is_exported":false}
{"type":"module_import","module":"Mathlib.Analysis.Normed.Group.Basic","imported":"Mathlib.Topology.MetricSpace.Basic","is_exported":true}
{"type":"decl_module","name":"instAddNat","module":"Mathlib.Algebra.Group.Defs"}
{"type":"decl_module","name":"instMulNat","module":"Mathlib.Algebra.Group.Defs"}
{"type":"decl_module","name":"mul_comm","module":"Mathlib.Algebra.Group.Basic"}
{"type":"def_height","name":"List.map","height":5,"reducibility":"regular"}
{"type":"def_height","name":"Nat.add","height":2,"reducibility":"regular"}
{"type":"def_height","name":"id","height":null,"reducibility":"abbrev"}
{"type":"def_height","name":"Classical.choice","height":null,"reducibility":"opaque"}
"""


class TestParseNDJSON:
    """Test NDJSON parsing."""

    def test_parse_valid_ndjson(self):
        """Each line should parse as valid JSON."""
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        assert result is not None

    def test_parse_empty_input(self):
        """Empty input should return empty result."""
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(""))
        assert result["meta"] == {}
        assert result["kinds"] == {}
        assert result["attributes"] == []
        assert result["classes"] == []

    def test_parse_skips_blank_lines(self):
        """Blank lines should be silently skipped."""
        from parser.from_mechanisms import parse_mechanisms

        input_text = '{"type":"meta","key":"total_constants","value":100}\n\n\n'
        result = parse_mechanisms(io.StringIO(input_text))
        assert result["meta"]["total_constants"] == 100


class TestMetaFields:
    """Test metadata extraction."""

    def test_meta_fields_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        assert result["meta"]["lean_version"] == "4.28.0-rc1"
        assert result["meta"]["total_constants"] == 450000
        assert result["meta"]["mathlib_constants"] == 308129


class TestKindCounts:
    """Test declaration kind counts."""

    def test_all_kinds_present(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        kinds = result["kinds"]
        assert "theorem" in kinds
        assert "definition" in kinds
        assert "inductive" in kinds

    def test_kind_counts_are_positive(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        for kind, count in result["kinds"].items():
            assert count > 0, f"Kind {kind} has count {count}"


class TestAttributes:
    """Test attribute extraction."""

    def test_known_attributes_present(self):
        """Standard attributes should appear in output."""
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        attr_names = {a["name"] for a in result["attributes"]}
        assert "simp" in attr_names
        assert "ext" in attr_names
        assert "instance" in attr_names
        assert "to_additive" in attr_names

    def test_attribute_counts_are_positive(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        for attr in result["attributes"]:
            assert attr["count"] > 0, f"Attribute {attr['name']} has count {attr['count']}"

    def test_attributes_sorted_by_count(self):
        """get_attributes_sorted should return descending order."""
        from parser.from_mechanisms import parse_mechanisms, get_attributes_sorted

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        sorted_attrs = get_attributes_sorted(result)
        counts = [a["count"] for a in sorted_attrs]
        assert counts == sorted(counts, reverse=True)


class TestClasses:
    """Test typeclass extraction."""

    def test_classes_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        class_names = {c["name"] for c in result["classes"]}
        assert "Add" in class_names
        assert "Monoid" in class_names

    def test_class_count(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        assert len(result["classes"]) == 4


class TestInstances:
    """Test instance extraction."""

    def test_instances_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        assert len(result["instances"]) == 2

    def test_instance_has_class(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        for inst in result["instances"]:
            assert "class" in inst
            assert inst["class"] != ""


class TestStructureParents:
    """Test structure inheritance extraction."""

    def test_parents_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        parents = result["structure_parents"]
        assert len(parents) == 2
        assert {"child": "CommMonoid", "parent": "Monoid"} in parents

    def test_parent_chain(self):
        """Should be able to reconstruct inheritance chains."""
        from parser.from_mechanisms import parse_mechanisms, get_parent_chain

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        chain = get_parent_chain(result, "CommMonoid")
        assert chain == ["CommMonoid", "Monoid", "MulOneClass"]


class TestToAdditive:
    """Test to_additive pair extraction."""

    def test_pairs_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        pairs = result["to_additive_pairs"]
        assert len(pairs) == 2
        assert {"source": "mul_comm", "target": "add_comm"} in pairs


class TestCoercions:
    """Test coercion extraction."""

    def test_coercions_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        assert len(result["coercions"]) == 3

    def test_coercion_fields(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        names = {c["name"] for c in result["coercions"]}
        assert "Int.ofNat" in names
        assert "Subtype.val" in names

    def test_coercion_has_type(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        for coe in result["coercions"]:
            assert "coe_type" in coe
            assert coe["coe_type"] in ("coe", "coeFun", "coeSort")


class TestDerivingHandlers:
    """Test deriving handler extraction."""

    def test_handlers_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        assert len(result["deriving_handlers"]) == 4

    def test_known_handlers_present(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        names = {h["name"] for h in result["deriving_handlers"]}
        assert "BEq" in names
        assert "Repr" in names
        assert "DecidableEq" in names


class TestStmtProofStats:
    """Test statement/proof edge classification."""

    def test_stats_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        stats = result["stmt_proof_stats"]
        assert stats is not None
        assert stats["S_only"] == 1200000
        assert stats["P_only"] == 3500000
        assert stats["SP"] == 3700000
        assert stats["total_decls"] == 308129

    def test_stats_partition_consistency(self):
        """S + P + SP should account for all edges."""
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        stats = result["stmt_proof_stats"]
        total_edges = stats["S_only"] + stats["P_only"] + stats["SP"]
        assert total_edges == 8400000


class TestModuleImports:
    """Test module import graph extraction."""

    def test_imports_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        assert len(result["module_imports"]) == 3

    def test_import_fields(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        for imp in result["module_imports"]:
            assert "module" in imp
            assert "imported" in imp
            assert "is_exported" in imp

    def test_public_private_split(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        public = [i for i in result["module_imports"] if i["is_exported"]]
        private = [i for i in result["module_imports"] if not i["is_exported"]]
        assert len(public) == 2
        assert len(private) == 1


class TestDeclModules:
    """Test declaration → file module mapping."""

    def test_decl_modules_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        assert len(result["decl_modules"]) == 3

    def test_decl_module_mapping(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        assert result["decl_modules"]["instAddNat"] == "Mathlib.Algebra.Group.Defs"
        assert result["decl_modules"]["mul_comm"] == "Mathlib.Algebra.Group.Basic"

    def test_empty_decl_modules(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(""))
        assert result["decl_modules"] == {}


class TestDefHeight:
    """Test definitional height extraction."""

    def test_def_heights_extracted(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        assert len(result["def_heights"]) == 4

    def test_regular_height(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        h = result["def_heights"]["List.map"]
        assert h["height"] == 5
        assert h["reducibility"] == "regular"

    def test_abbrev_height(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        h = result["def_heights"]["id"]
        assert h["height"] is None
        assert h["reducibility"] == "abbrev"

    def test_opaque_height(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        h = result["def_heights"]["Classical.choice"]
        assert h["height"] is None
        assert h["reducibility"] == "opaque"

    def test_empty_def_heights(self):
        from parser.from_mechanisms import parse_mechanisms

        result = parse_mechanisms(io.StringIO(""))
        assert result["def_heights"] == {}


class TestSummaryTable:
    """Test summary table generation for paper."""

    def test_generate_summary(self):
        """Should produce a summary dict suitable for the paper table."""
        from parser.from_mechanisms import parse_mechanisms, generate_summary

        result = parse_mechanisms(io.StringIO(SAMPLE_OUTPUT))
        summary = generate_summary(result)

        assert "total_constants" in summary
        assert "total_classes" in summary
        assert "total_instances" in summary
        assert "total_attributes" in summary
        assert "top_attributes" in summary
        assert len(summary["top_attributes"]) > 0


# --- Integration test with real data (skipped if file not present) ---

REAL_OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "mechanisms.ndjson"


@pytest.mark.skipif(
    not REAL_OUTPUT_PATH.exists(),
    reason="Real extraction output not available"
)
class TestRealData:
    """Integration tests with actual Lean extraction output."""

    def test_real_simp_present(self):
        """simp should be among the registered attributes."""
        from parser.from_mechanisms import parse_mechanisms

        with open(REAL_OUTPUT_PATH) as f:
            result = parse_mechanisms(f)
        attr_names = {a["name"] for a in result["attributes"]}
        assert "simp" in attr_names

    def test_real_class_count(self):
        """Mathlib should have hundreds of typeclasses."""
        from parser.from_mechanisms import parse_mechanisms

        with open(REAL_OUTPUT_PATH) as f:
            result = parse_mechanisms(f)
        assert len(result["classes"]) > 500

    def test_real_instance_count(self):
        """Mathlib should have tens of thousands of instances."""
        from parser.from_mechanisms import parse_mechanisms

        with open(REAL_OUTPUT_PATH) as f:
            result = parse_mechanisms(f)
        assert len(result["instances"]) > 10000

    def test_real_to_additive_count(self):
        """to_additive should produce thousands of pairs."""
        from parser.from_mechanisms import parse_mechanisms

        with open(REAL_OUTPUT_PATH) as f:
            result = parse_mechanisms(f)
        assert len(result["to_additive_pairs"]) > 2000

    def test_real_coercion_count(self):
        """Mathlib should have hundreds of registered coercions."""
        from parser.from_mechanisms import parse_mechanisms

        with open(REAL_OUTPUT_PATH) as f:
            result = parse_mechanisms(f)
        assert len(result["coercions"]) > 100

    def test_real_deriving_handlers(self):
        """There should be at least 10 registered deriving handlers."""
        from parser.from_mechanisms import parse_mechanisms

        with open(REAL_OUTPUT_PATH) as f:
            result = parse_mechanisms(f)
        assert len(result["deriving_handlers"]) > 10

    def test_real_stmt_proof_stats(self):
        """Statement/proof stats should be populated."""
        from parser.from_mechanisms import parse_mechanisms

        with open(REAL_OUTPUT_PATH) as f:
            result = parse_mechanisms(f)
        stats = result["stmt_proof_stats"]
        assert stats is not None
        assert stats["S_only"] > 0
        assert stats["P_only"] > 0
        assert stats["SP"] > 0

    def test_real_def_heights(self):
        """Definitional heights should be extracted for many definitions."""
        from parser.from_mechanisms import parse_mechanisms

        with open(REAL_OUTPUT_PATH) as f:
            result = parse_mechanisms(f)
        heights = result["def_heights"]
        if not heights:
            pytest.skip("def_height records not present in mechanisms.ndjson (re-run extraction)")
        # Most definitions have regular reducibility with a height
        regular = [h for h in heights.values() if h["reducibility"] == "regular"]
        assert len(regular) > 10000
        # Heights should be non-negative integers
        for h in regular:
            assert isinstance(h["height"], int)
            assert h["height"] >= 0

    def test_real_module_imports(self):
        """Module import graph should have thousands of edges."""
        from parser.from_mechanisms import parse_mechanisms

        with open(REAL_OUTPUT_PATH) as f:
            result = parse_mechanisms(f)
        assert len(result["module_imports"]) > 5000

    def test_real_import_utilization(self):
        """Import utilization should be computable from CSV + mechanisms."""
        from parser.from_mechanisms import compute_import_utilization

        hf_dir = Path.home() / ".cache/huggingface/hub/datasets--MathNetwork--MathlibGraph/snapshots/bc4173ec3beda64713ae81f602ce224491c61703"
        edges_path = hf_dir / "mathlib_edges.csv"
        if not edges_path.exists():
            pytest.skip("HuggingFace dataset not available")

        with open(REAL_OUTPUT_PATH) as f:
            from parser.from_mechanisms import parse_mechanisms
            result = parse_mechanisms(f)

        if not result["decl_modules"]:
            pytest.skip("decl_module records not present in mechanisms.ndjson (re-run extraction)")

        stats = compute_import_utilization(edges_path, result["module_imports"], result["decl_modules"])
        assert stats["total_import_edges"] > 5000
        assert 0 < stats["mean_util"] < 1
        assert stats["zero_util_edges"] >= 0

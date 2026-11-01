"""Checks for lib/cftr.py against the curated map (run scripts/cftr_variant_map.py first)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.cftr import classify_genotype as cg  # noqa: E402

NAN = float("nan")


def test_f508del_spellings_and_c_names():
    r = cg("deltaf508", NAN, "other", "c.1520_1522delTCT")       # c.1520_1522del == F508del on NM_000492.4
    assert r["f508del_genotype"] == "F508del homozygous" and r["n_eti_alleles"] == 2
    r = cg("[delta]F508", NAN, "other", "c.1542_1543delAT")      # == 1677delTA
    assert r["allele2"] == "1677delTA" and r["functional_genotype"] == "two MF alleles"


def test_polymorphism_and_missing_make_incomplete():
    r = cg("F508del", NAN, "M470V", NAN)
    assert r["f508del_genotype"] == "incomplete" and r["any_non_cf_polymorphism"]
    assert cg("other", NAN, "F508del", NAN)["f508del_genotype"] == "incomplete"     # 'other' without OtherMutation
    assert cg("not done", NAN, "unknown", NAN)["functional_genotype"] == "incomplete"


def test_rf_gating_and_eti():
    r = cg("2789+ 5G>A", NAN, "unknown", NAN)
    assert r["functional_genotype"] == "≥1 RF allele" and r["n_eti_alleles"] == 1
    r = cg("G178R", NAN, "G542X", NAN)
    assert r["functional_genotype"] == "other or unresolved" and r["functional_genotype_detail"].startswith("≥1 gating")
    r = cg("2183AA- >G", NAN, "2183AA->G", NAN)                  # NOT the label's 2183A>G (K684R)
    assert r["n_eti_alleles"] == 0 and r["functional_genotype"] == "two MF alleles"
    r = cg("L997F", NAN, "deltaF508", NAN)                       # label-listed but non-CF-causing
    assert r["n_eti_alleles"] == 2 and r["n_eti_alleles_cf"] == 1


def test_split_cis_complex():
    r = cg("S466X", NAN, "R1070Q", NAN)
    assert r["possible_cis_pair"] and r["functional_genotype"] == "incomplete" and r["n_eti_alleles"] == 0


def test_class_genotype():
    assert cg("F508del", NAN, "G542X", NAN)["class_genotype"] == "severe"             # II + I
    assert cg("E92K", NAN, "E92K", NAN)["class_genotype"] == "severe"                 # II
    assert cg("F508del", NAN, "R334W", NAN)["class_genotype"] == "mild"               # IV dominates
    assert cg("2789+ 5G>A", NAN, "unknown", NAN)["class_genotype"] == "mild"          # V, other slot incomplete
    assert cg("G542X", NAN, "unknown", NAN)["class_genotype"] == "incomplete"
    assert cg("N1303K", NAN, "G178R", NAN)["class_genotype"] == "severe"             # II + III
    assert cg("S1455X", NAN, "F508del", NAN)["class_genotype"] == "mild"              # best-available class VI


def test_clinician_class_assignments():
    # 1 Nov 2026: E1044G II, G314R II, L571S I, M1101R I, H1054D II (all severe)
    for v, cls in [("E1044G", "II"), ("G314R", "II"), ("L571S", "I"), ("M1101R", "I"), ("H1054D", "II")]:
        r = cg(v, NAN, "F508del", NAN)
        assert r["class1"].split(" ")[0] == cls and r["class_genotype"] == "severe", v
    assert cg("E1044G", NAN, "R334W", NAN)["class_genotype"] == "mild"

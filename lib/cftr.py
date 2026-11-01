"""Per-patient CFTR genotype categories from the curated variant map.

The allele-level decisions (normalisation, CFTR2 status, function group, ETI label status) are made once, in
scripts/cftr_variant_map.py, and stored in data/reference/cftr_variant_map.csv (one row per distinct raw string).
This module only looks raw strings up in that map and combines the two allele slots.

    from lib.cftr import classify_genotype
    classify_genotype("deltaF508", None, "other", "c.1542_1543delAT")

Conventions (see data/reference/CFTR_README.md for sources):
  * a slot is *complete* when its variant is F508del, CF-causing or of varying clinical consequence (VCC);
    'non-CF-causing' (polymorphism), 'uncertain' (VUS / unidentifiable string) and 'missing' slots are incomplete.
  * (a) F508del genotype: homozygous / heterozygous / no F508del require both slots complete; otherwise 'incomplete'.
  * (b) functional genotype (modulator-trial convention): '≥1 RF allele' if any complete slot is residual function
    (RF dominates phenotype, so this holds even when the other slot is incomplete); 'two MF alleles' if both slots are
    F508del or minimal function; 'incomplete' if any slot is incomplete; otherwise 'other or unresolved' (gating,
    corrector-responsive processing variants, other CF-causing variants without an MF/RF designation).
  * (c) ETI: number of slots whose variant is F508del or listed in the TRIKAFTA USPI Table 6 (Dec 2024). Label-literal
    count (n_eti_alleles) and a stricter count restricted to complete slots (n_eti_alleles_cf).
  * (d) any_non_cf_polymorphism: a slot holds a non-CF-causing variant (that slot is incomplete).
  * Known cis complexes split over the two slots (e.g. S466X in one slot and R1070Q in the other) are flagged; the
    'passenger' slot is treated as incomplete because both variants may sit on the same chromosome.
  * (e) class genotype (authors' decision, 1 Nov 2026; McKone 2006): functional class I–III = severe, IV–VI = mild.
    'mild' if any complete slot is class IV–VI (the milder allele determines phenotype, also when the other slot is
    incomplete); 'severe' if both slots are complete and class I–III; 'incomplete' if any slot is incomplete; otherwise
    'unclassified' (a CF-causing variant without a published class). Variants of unknown class get a class only when the
    variant type makes it unambiguous or the clinical authors assigned one (CLASS_CURATED).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

MAP_CSV = Path(__file__).resolve().parents[1] / "data" / "reference" / "cftr_variant_map.csv"
COMPLETE = {"F508del", "CF-causing", "VCC"}
MISSING_TOKENS = {"unknown", "not done", "nan", "", "none"}
# (anchor, passenger, reason): if one slot is the anchor and the other the passenger, they may be one cis allele
CIS_PAIRS = [
    ("S466X", "R1070Q", "[S466X;R1070Q] is a recognised complex allele (CFTR2: CF-causing)"),
    ("F508del", "L467F", "[F508del;L467F] is a recognised complex allele (CFTR2: CF-causing)"),
    ("3199del6", "I148T", "[I148T;3199del6] is a recognised complex allele"),
    ("G576A", "R668C", "[G576A;R668C] is a recognised complex allele (CFTR2: non-CF-causing)"),
    ("R74W", "D1270N", "[R74W;D1270N] is a recognised complex allele"),
    ("R74W", "V201M", "[R74W;V201M] is a recognised complex allele"),
    ("V201M", "D1270N", "[V201M;D1270N] is a recognised complex allele"),
]
# CF-causing/VCC variants whose functional class is 'unknown' in the map: class from the variant type, or assigned by the
# clinical authors (1 Nov 2026) for CF-causing missense variants without a published class
CLASS_CURATED = {
    "S1455X": "VI (last-exon nonsense, C-terminal truncation; best available)",
    "3849+5G->A": "V (non-canonical splice site; best available)",
    "1898+3A->G": "V (non-canonical splice site; best available)",
    "E1044G": "II (clinician assignment, 1 Nov 2026)",
    "G314R": "II (clinician assignment, 1 Nov 2026)",
    "L571S": "I (clinician assignment, 1 Nov 2026)",
    "M1101R": "I (clinician assignment, 1 Nov 2026)",
    "H1054D": "II (clinician assignment, 1 Nov 2026)",
}
_CACHE: dict = {}


def class_severity(fclass) -> str:
    """'severe' for class I-III, 'mild' for class IV-VI (mixed 'IV/V' is mild, 'I/III' severe), else 'unclassified'."""
    s = "" if _isna(fclass) else str(fclass).strip()
    first = s.split(" ")[0].split("/")
    if first and first[0] in ("I", "II", "III"):
        return "severe"
    if first and first[0] in ("IV", "V", "VI"):
        return "mild"
    return "unclassified"


def _isna(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def load_variant_map(path: Path | str = MAP_CSV, force: bool = False) -> dict[str, dict]:
    path = Path(path)
    if force or path not in _CACHE:
        with open(path, newline="", encoding="utf-8") as f:
            _CACHE[path] = {r["raw_string"]: r for r in csv.DictReader(f)}
    return _CACHE[path]


def raw_key(mut, other) -> str:
    """Effective raw allele string for one slot: OtherMutation when Mutation == 'other'."""
    m = "nan" if _isna(mut) else str(mut)
    if m.strip().lower() == "other":
        return "other" if _isna(other) else str(other)
    return m


def resolve_slot(mut, other, vmap: dict | None = None) -> dict:
    vmap = vmap if vmap is not None else load_variant_map()
    key = raw_key(mut, other)
    row = vmap.get(key)
    if row is not None:
        return {**row, "raw": key, "unmapped": False}
    if key.strip().lower() in MISSING_TOKENS:
        return {"raw": key, "normalised_legacy_name": "(missing)", "slot_status": "missing", "function_group": "unresolved",
                "eti_fda_eligible": "unknown", "confidence": "high", "unmapped": False}
    return {"raw": key, "normalised_legacy_name": f"(unmapped) {key}", "slot_status": "uncertain",
            "function_group": "unresolved", "eti_fda_eligible": "unknown", "confidence": "low", "unmapped": True}


def classify_genotype(m1, o1, m2, o2, vmap: dict | None = None) -> dict:
    """Patient genotype categories from the two registry slots (Mutation1/OtherMutation1, Mutation2/OtherMutation2)."""
    s = [resolve_slot(m1, o1, vmap), resolve_slot(m2, o2, vmap)]
    names = [x["normalised_legacy_name"] for x in s]
    status = [x["slot_status"] for x in s]
    fg = [x["function_group"] for x in s]
    complete = [st in COMPLETE for st in status]

    # cis complexes split across slots -> passenger slot incomplete
    cis_note = ""
    for anchor, passenger, why in CIS_PAIRS:
        for i, j in ((0, 1), (1, 0)):
            if names[i] == anchor and names[j] == passenger:
                complete[j] = False
                cis_note = f"{anchor}/{passenger}: {why}"
    cis = bool(cis_note)

    n_f = sum(st == "F508del" for st in status)
    if all(complete):
        f508 = {2: "F508del homozygous", 1: "F508del heterozygous", 0: "no F508del"}[n_f]
        f508_detail = f508
    else:
        f508 = "incomplete"
        f508_detail = "F508del/incomplete" if n_f >= 1 else ("both slots incomplete" if not any(complete) else "non-F508del/incomplete")

    rf = any(c and g == "RF" for c, g in zip(complete, fg))
    if rf:
        fun = "≥1 RF allele"
        fun_d = "≥1 RF allele" + ("" if all(complete) else " (other slot incomplete)")
    elif all(complete) and all(g in ("F508del", "MF") for g in fg):
        fun = fun_d = "two MF alleles"
    elif not all(complete):
        fun = fun_d = "incomplete"
    else:
        fun = "other or unresolved"
        fun_d = ("≥1 gating allele (no RF)" if "gating" in fg else
                 "other CF-causing (no MF/RF designation)")

    def eti_slot(i, strict):
        x = s[i]
        if cis and not complete[i] and x["slot_status"] in COMPLETE:   # cis passenger: do not count
            return False
        if strict and not complete[i]:
            return False
        return x.get("eti_fda_eligible") == "yes"

    fclass = [CLASS_CURATED.get(n, x.get("functional_class")) for n, x in zip(names, s)]
    sev = [class_severity(fc) for fc in fclass]
    if any(c and v == "mild" for c, v in zip(complete, sev)):
        cls = "mild"
    elif all(complete) and all(v == "severe" for v in sev):
        cls = "severe"
    elif not all(complete):
        cls = "incomplete"
    else:
        cls = "unclassified"

    n_eti = sum(eti_slot(i, False) for i in (0, 1))
    n_eti_cf = sum(eti_slot(i, True) for i in (0, 1))
    eti_undetermined = any(x.get("eti_fda_eligible") == "unknown" for x in s)

    return {
        "allele1": names[0], "allele2": names[1],
        "slot1_status": status[0], "slot2_status": status[1],
        "fg1": fg[0], "fg2": fg[1],
        "f508del_genotype": f508, "f508del_detail": f508_detail, "n_f508del": n_f,
        "functional_genotype": fun, "functional_genotype_detail": fun_d,
        "class1": fclass[0], "class2": fclass[1], "class_genotype": cls,
        "n_eti_alleles": n_eti, "eti_eligible": "yes" if n_eti >= 1 else "no",
        "n_eti_alleles_cf": n_eti_cf, "eti_eligible_cf": "yes" if n_eti_cf >= 1 else "no",
        "eti_undetermined_slot": eti_undetermined,
        "any_non_cf_polymorphism": any(st == "non-CF-causing" for st in status),
        "possible_cis_pair": cis, "cis_note": cis_note,
        "any_low_confidence": any(x.get("confidence") == "low" for x in s),
        "any_unmapped": any(x.get("unmapped") for x in s),
    }


def classify_frame(df, cols=("Mutation1", "OtherMutation1", "Mutation2", "OtherMutation2"), vmap: dict | None = None):
    """Vectorised helper: returns a DataFrame of classify_genotype outputs aligned to df.index."""
    import pandas as pd
    vmap = vmap if vmap is not None else load_variant_map()
    a, b, c, d = (df[k] for k in cols)
    return pd.DataFrame([classify_genotype(w, x, y, z, vmap=vmap) for w, x, y, z in zip(a, b, c, d)], index=df.index)

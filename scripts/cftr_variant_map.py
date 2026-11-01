#!/usr/bin/env python3
"""Curated, reproducible classification of every CFTR variant string in the CF Registry of Türkiye export.

Reads the raw allele strings (Mutation1/OtherMutation1/Mutation2/OtherMutation2 of Old_Files/finaldata.xlsx),
normalises them, matches them to public references and writes one row per distinct raw string to
    data/reference/cftr_variant_map.csv
together with aggregate (non patient-level) summary tables
    data/reference/cftr_variant_summary.md / .json

External references (all cached in data/reference/cftr_external_cache.json so that re-runs are offline):
  * CFTR2 variant list, 30 January 2026 release (cftr2.org; xlsx)
  * FDA TRIKAFTA USPI revised 12/2024 (Table 6, responsive variants)
  * FDA KALYDECO USPI revised 08/2023 (Table 3) and SYMDEKO USPI revised 08/2023 (Table 6) - used only to define
    residual-function (RF) alleles by the modulator-trial convention
  * NCBI RefSeq NM_000492.4 (Bio.Entrez) - used to verify c.-name equivalences and residue identities
  * NCBI ClinVar (Bio.Entrez esearch/esummary) - only for variants absent from CFTR2

PRIVACY: only hard-coded variant-name strings (never patient data) are sent to external services.

Usage:  python scripts/cftr_variant_map.py [--refresh] [--offline]
  --refresh  re-download every external source and overwrite the cache
  --offline  never touch the network (fails if the cache lacks something)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import json
import math
import re
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ANALYSIS = HERE.parents[1]                      # November/analysis
REPO = HERE.parents[3]                          # UKKS
RAW_XLSX = Path(os.environ.get("CFRT_EXPORT", REPO / "Old_Files" / "finaldata.xlsx"))
REF = ANALYSIS / "data" / "reference"
CACHE_JSON = REF / "cftr_external_cache.json"
OUT_CSV = REF / "cftr_variant_map.csv"
SUMMARY_MD = REF / "cftr_variant_summary.md"
SUMMARY_JSON = REF / "cftr_variant_summary.json"
sys.path.insert(0, str(ANALYSIS))

ACCESS_DATE = "2026-10-31"
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")    # required by NCBI E-utilities for uncached lookups (user identification only)
NCBI_TOOL = "ukks_cftr_variant_map"
UA = {"User-Agent": "Mozilla/5.0 (UKKS CFTR variant curation; research use)"}

SOURCES = {
    "cftr2": {"url": "https://cftr2.org/sites/default/files/CFTR2_30January2026.xlsx",
              "landing": "https://cftr2.org/mutations_history", "version": "CFTR2 variant list, 30 January 2026"},
    "fda_eti": {"url": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/212273s013lbl.pdf",
                "version": "TRIKAFTA USPI, NDA 212273 S-013, revised 12/2024 (Table 6)",
                "table": "Table 6: List of CFTR Gene Mutations Responsive to TRIKAFTA", "end": ["⁎ Clinical data", "* Clinical data"]},
    "fda_iva": {"url": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/203188s040,207925s017lbl.pdf",
                "version": "KALYDECO USPI, NDA 203188 S-040 / 207925 S-017, revised 08/2023 (Table 3)",
                "table": "Table 3: List of CFTR Gene Mutations that Produce CFTR Protein and are Responsive to KALYDECO",
                "end": ["* Clinical data exist"]},
    "fda_tezi": {"url": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/210491s011lbl.pdf",
                 "version": "SYMDEKO USPI, NDA 210491 S-011, revised 08/2023 (Table 6)",
                 "table": "Table 6: List of CFTR Gene Mutations that Produce CFTR Protein and are Responsive to SYMDEKO",
                 "end": ["* Clinical data for these mutations"]},
    "refseq": {"accession": "NM_000492.4", "version": "NCBI RefSeq NM_000492.4 (CFTR mRNA), via Bio.Entrez efetch"},
    "clinvar": {"version": "NCBI ClinVar via Bio.Entrez esearch/esummary (queried on access date)"},
}

# ======================================================================================================================
# 1. CURATED DECISIONS (every manual decision lives here, with a short reason)
# ======================================================================================================================
# Tokens meaning "no variant information" -> slot_status 'missing'
MISSING_TOKENS = {"unknown": "registry value 'unknown'", "not done": "registry value 'not done' (genotyping not performed)",
                  "nan": "empty cell", "": "empty cell", "none": "empty cell",
                  "other": "Mutation = 'other' but OtherMutation empty (variant not specified)"}

# Trial/label conventions -------------------------------------------------------------------------------------------
# Gating (class III) variants: the KALYDECO gating set studied clinically (Trials 1 and 4 of the USPI). G970R is
# deliberately excluded: the USPI states it causes a splicing defect with little-to-no protein (treated as MF).
GATING = {"G551D", "G1244E", "G1349D", "G178R", "G551S", "S1251N", "S1255P", "S549N", "S549R"}
# Missense / in-frame variants explicitly treated as minimal function (MF) in the VX17-445-102 / VX-661 F/MF trial
# eligibility lists ("missense mutations ... not responsive to IVA and TEZ/IVA in vitro"). N1303K, I507del, G85E, R347P
# and R560T are named in the Kaftrio EU SmPC (sect. 5.1); the remainder follow the published trial appendix list
# (Middleton 2019; Heijerman 2019) as recalled - flagged 'medium' confidence where not named in the SmPC.
MF_MISSENSE_SMPC = {"N1303K", "I507del", "G85E", "R347P", "R560T"}
MF_MISSENSE_TRIAL = {"A46D", "L467P", "V520F", "A559T", "R560S", "A561E", "Y569D", "L1065P", "R1066C", "R1066M",
                     "L1077P", "H1085R", "M1101K",
                     "G970R"}   # G970R: splice defect, little/no protein (KALYDECO USPI); listed as splice MF (3040G->C) in F/MF trials
# Residual function (RF): ivacaftor-responsive, non-gating CF-causing/VCC variants (KALYDECO Table 3) plus the
# clinically studied residual-function set of the TEZ/IVA label (Symdeko Table 6 '*', = EXPAND trial list).
# Nonsense-type variants that are NOT treated as MF despite their type (reason in value)
NULL_EXCEPTIONS = {"S1455X": "C-terminal nonsense in the last exon (escapes NMD); CFTR2 'varying clinical consequence' - "
                             "not assumed minimal function"}
# Functional class where a specific, commonly cited assignment exists (De Boeck & Amaral 2016; Veit 2016). Others get
# a rule-based 'best available' class (null types -> I, gating -> III, RF -> IV/V) or 'unknown'.
FUNCTIONAL_CLASS = {"F508del": "II", "N1303K": "II", "G85E": "II", "I507del": "II", "E92K": "II", "R1066C": "II",
                    "L1077P": "II", "M1101K": "II", "A46D": "II", "R347P": "IV", "R334W": "IV", "R117H": "IV",
                    "R347H": "IV", "R352Q": "IV", "2789+5G->A": "V", "3849+10kbC->T": "V", "3272-26A->G": "V",
                    "A455E": "V", "711+3A->G": "V", "E831X": "V (aberrant splicing)", "G970R": "I/III (splicing defect)",
                    "M470V": "not applicable", "5T": "V (splicing; polyT)", "5T;TG12": "V (splicing; polyT)",
                    "5T;TG13": "V (splicing; polyT)"}

# Aliases: normalised token -> CFTR2 legacy name. 'seq' = equivalence verified against NM_000492.4 at run time.
#   ("del", a, b, a2, b2): deleting c.a_b gives the same sequence as deleting c.a2_b2
#   ("dupA", a, a2): inserting the base after c.a equals inserting it after c.a2 (duplication in a run)
#   ("sub", pos, ref, alt, protein): substitution at c.pos ref>alt gives protein change (or '=' for synonymous)
ALIASES = {
    "deltaF508": ("F508del", "high", "legacy spelling of F508del", None),
    "[delta]F508": ("F508del", "high", "legacy spelling of F508del", None),
    "F508": ("F508del", "medium", "bare 'F508' read as F508del (no other F508 variant recorded in this registry)", None),
    "[delta]I507": ("I507del", "high", "legacy spelling of I507del", None),
    "V470M": ("M470V", "high", "same polymorphism (c.1408A>G); NM_000492.4 encodes Val470 so it is sometimes written V470M", None),
    "c.1520_1522delTCT": ("F508del", "high", "non-3'-normalised description of F508del (identical product sequence)", ("del", 1520, 1522, 1521, 1523)),
    "c.1520-1522 delTCT": ("F508del", "high", "c.1520_1522del written with '-'; identical product to F508del", ("del", 1520, 1522, 1521, 1523)),
    "c.1520_1522del": ("F508del", "high", "non-3'-normalised description of F508del (identical product sequence)", ("del", 1520, 1522, 1521, 1523)),
    "c.1542_1543delAT": ("1677delTA", "high", "deletion within an ATATA run; identical product to c.1545_1546del (1677delTA)", ("del", 1542, 1543, 1545, 1546)),
    "c.1542-1543del AT": ("1677delTA", "high", "as c.1542_1543delAT; identical product to 1677delTA", ("del", 1542, 1543, 1545, 1546)),
    "c.2046delA": ("2184delA", "high", "deletion within the c.2046-2052 A7 run; identical product to c.2052del (2184delA)", ("del", 2046, 2046, 2052, 2052)),
    "2181delA": ("2184delA", "high", "legacy 2181 = c.2049 inside the A7 run; identical product to 2184delA", ("del", 2049, 2049, 2052, 2052)),
    "c.172_175delGATA": ("306delTAGA", "high", "4-bp deletion in a GATAGA repeat; identical product to c.174_177del (306delTAGA)", ("del", 172, 175, 174, 177)),
    "c.174_177delGATA": ("306delTAGA", "high", "position matches 306delTAGA (c.174_177del); bases written out of phase", ("del", 172, 175, 174, 177)),
    "c.1126-1127insA": ("1259insA", "high", "insertion of A into the c.1127-1130 A-run = c.1130dup (1259insA)", ("dupA", 1126, 1130)),
    "c.2856G>C": ("M952I", "high", "c.2856G>C = p.Met952Ile (verified)", ("sub", 2856, "G", "C", "M952I")),
    "c.3196C->T": ("R1066C", "high", "c.3196C>T = p.Arg1066Cys (verified)", ("sub", 3196, "C", "T", "R1066C")),
    "c.3131A>G": ("E1044G", "high", "c.3131A>G = p.Glu1044Gly (verified)", ("sub", 3131, "A", "G", "E1044G")),
    "c.3131A> G": ("E1044G", "high", "c.3131A>G = p.Glu1044Gly (verified)", ("sub", 3131, "A", "G", "E1044G")),
    "c.3196C>T": ("R1066C", "high", "c.3196C>T = p.Arg1066Cys (verified)", ("sub", 3196, "C", "T", "R1066C")),
    "c.3718_2477C>T": ("3849+10kbC->T", "medium", "typo of c.3718-2477C>T ('_' for '-')", None),
    "c.3712477 C>T": ("3849+10kbC->T", "medium", "typo of c.3718-2477C>T (missing '8-')", None),
    "c.1210-34TG[12]T[5]": ("5T;TG12", "high", "HGVS repeat notation for TG12-5T", None),
    "c.1210-7_1210-6delTT": ("5T;TG11", "low", "strict HGVS on NM_000492.4 (TG11-T7 reference) = 5T;TG11 (CFTR2 non-CF-causing); "
                             "the TG repeat may simply not have been reported - if TG12/TG13 this would be VCC", None),
    "C.1210-7_1210-6del": ("5T;TG11", "low", "as c.1210-7_1210-6delTT (TG repeat not reported)", None),
    "IVS14b+5G>A": ("2789+5G->A", "high", "legacy intron 14b +5 = c.2657+5G>A", None),
    "IVS16+5G>A": ("2789+5G->A", "medium", "new intron numbering (intron 16 = c.2657+) -> c.2657+5G>A; under legacy numbering it would be "
                   "c.2988+5G>A (not a known variant)", None),
    "2657+ 5G>A": ("2789+5G->A", "medium", "c.-number written without 'c.' (c.2657+5G>A = 2789+5G>A)", None),
    "IVS15-15 C>G": ("2752-15C/G", "medium", "new intron 15 -15 = c.2620-15C>G; legacy numbering impossible (reference T at c.2909-15)", None),
    "223C/T": ("R31C", "medium", "legacy 223C>T = c.91C>T = p.Arg31Cys (verified)", ("sub", 91, "C", "T", "R31C")),
    "2553A/G": ("I807M", "medium", "legacy 2553A>G = c.2421A>G = p.Ile807Met (verified)", ("sub", 2421, "A", "G", "I807M")),
    "2694T/G": ("T854T", "high", "legacy 2694T>G = c.2562T>G = p.Thr854= (verified)", ("sub", 2562, "T", "G", "=")),
    "- 33G- >A": ("5UTR-33G->A", "medium", "legacy -33G>A = 5'UTR c.-165G>A", None),
    "R-1283G": ("R1283G", "medium", "stray '-' removed", None),
    "PW401X": ("W401X", "medium", "stray 'P' removed (residue 401 is Trp)", None),
    "1531C/T (L467F)": ("L467F", "high", "legacy 1531C>T = c.1399C>T = p.Leu467Phe (verified)", ("sub", 1399, "C", "T", "L467F")),
    "1531C / T (L467F)": ("L467F", "high", "legacy 1531C>T = c.1399C>T = p.Leu467Phe", ("sub", 1399, "C", "T", "L467F")),
    "c.3870A>G": ("P1290P", "high", "c.3870A>G = p.Pro1290= (verified)", ("sub", 3870, "A", "G", "=")),
    "c.3870 A>G": ("P1290P", "high", "c.3870A>G = p.Pro1290=", ("sub", 3870, "A", "G", "=")),
    "c.3718-24 G>A": ("3850-24G->A", "high", "c.3718-24G>A = legacy 3850-24G>A", None),
    "c.3718-24g>A": ("3850-24G->A", "high", "c.3718-24G>A = legacy 3850-24G>A", None),
    "c.1392+89G>A1": ("1524+89G->A", "medium", "trailing '1' typo; c.1392+89G>A = 1524+89G>A", None),
    "c.1265C>T": ("S422F", "high", "c.1265C>T = p.Ser422Phe (verified)", ("sub", 1265, "C", "T", "S422F")),
    "c.1364C>T": ("A455V", "high", "c.1364C>T = p.Ala455Val (verified; NOT A455E)", ("sub", 1364, "C", "T", "A455V")),
    "c.1689C>A": ("Y563X", "high", "c.1689C>A = p.Tyr563Ter (verified)", ("sub", 1689, "C", "A", "Y563*")),
    "c.325T>G": ("Y109D", "high", "c.325T>G = p.Tyr109Asp (verified)", ("sub", 325, "T", "G", "Y109D")),
    "c.3293G>T": ("W1098L", "high", "c.3293G>T = p.Trp1098Leu (verified)", ("sub", 3293, "G", "T", "W1098L")),
    "c.3587C>A": ("S1196X", "high", "c.3587C>A = p.Ser1196Ter (verified)", ("sub", 3587, "C", "A", "S1196*")),
    "c.3846G>A": ("W1282X", "high", "c.3846G>A = p.Trp1282Ter (verified)", ("sub", 3846, "G", "A", "W1282*")),
    "c.3930G>A": ("W1310X", "high", "c.3930G>A = p.Trp1310Ter (verified)", ("sub", 3930, "G", "A", "W1310*")),
    "c.2976T>A": ("F992L", "high", "c.2976T>A = p.Phe992Leu (verified)", ("sub", 2976, "T", "A", "F992L")),
    "c.473G> A": ("S158N", "high", "c.473G>A = p.Ser158Asn (verified)", ("sub", 473, "G", "A", "S158N")),
    "p.P740L": ("P740L", "high", "one-letter protein name", None),
    "c.717delG": ("849delG", "high", "c.717del = 849delG (CFTR2 cDNA name)", None),
    "1677 del TA": ("1677delTA", "high", "spaces removed", None),
    "c.2052dupA": ("2184insA", "high", "c.2052dup = 2184insA (CFTR2 cDNA name)", None),
}

# Intronic 'IVS' names whose numbering (legacy vs new exon numbering) is ambiguous but where every reading is a
# canonical (+/-1,2) splice-site variant -> minimal function regardless; identity reported as both candidates.
IVS_CANONICAL = {
    "IVS9-1G>A": "c.1393-1G>A (1525-1G>A, legacy intron 9) or c.1210-1G>A (1342-1G>A, new intron 9)",
    "IVS 9-1 G>A": "c.1393-1G>A (1525-1G>A, legacy intron 9) or c.1210-1G>A (1342-1G>A, new intron 9)",
    "IVS9as - 1G>A": "c.1393-1G>A (1525-1G>A) or c.1210-1G>A (1342-1G>A); 'as' = acceptor site",
    "IVS9-1G>T": "c.1393-1G>T (1525-1G>T) or c.1210-1G>T (1342-1G>T)",
    "IVS10-1G>A": "c.1585-1G>A (1717-1G>A, legacy intron 10) or c.1393-1G>A (1525-1G>A, new intron 10)",
    "IVS8+1G>A": "c.1209+1G>A (1341+1G>A, legacy intron 8) or c.1116+1G>A (1248+1G>A, new intron 8)",
    "IVS16+1 G>A": "c.2988+1G>A (3120+1G>A, legacy intron 16) or c.2657+1G>A (2789+1G>A, new intron 16)",
    "IVS6b+2T>C": "c.869+2T>C (1001+2T>C, legacy intron 6b)",
}

# Variants not in CFTR2 (or only resolvable via sequence/ClinVar). vtype drives the MF rule; 'clinvar' is the query
# term sent to ClinVar (variant name only). status_rule: 'clinvar' -> slot status from ClinVar germline class;
# 'null' -> null variant type (nonsense/frameshift/canonical splice/large deletion) treated as CF-causing by rule.
NOT_IN_CFTR2 = {
    "c.4061T>A": dict(c="c.4061T>A", p="p.Met1354Lys", vtype="missense", clinvar="c.4061T>A", seq=("sub", 4061, "T", "A", "M1354K")),
    "C.4061 T>A": dict(c="c.4061T>A", p="p.Met1354Lys", vtype="missense", clinvar="c.4061T>A", seq=("sub", 4061, "T", "A", "M1354K")),
    "c.3874-3C>G": dict(c="c.3874-3C>G", p="p.?", vtype="intronic", clinvar="c.3874-3C>G"),
    "3874-3C>G": dict(c="c.3874-3C>G", p="p.?", vtype="intronic", clinvar="c.3874-3C>G",
                      why="'c.' omitted: c.3874 is the first base of exon 24, whereas legacy 3874 (= c.3742) is exonic, "
                          "so only the c. reading is valid"),
    "711+9A-G": dict(c="c.579+9A>G", p="p.?", vtype="intronic", clinvar="c.579+9A>G",
                     why="legacy 711+9A>G = c.579+9A>G (not in CFTR2)"),
    "4444delC": dict(c="c.4312del", p="p.(Arg1438fs)", vtype="frameshift (last exon)", clinvar=None, seq=("ref", 4312, "C"),
                     why="legacy 4444delC = c.4312delC (reference C verified) in the LAST exon (c.4243 onwards): escapes NMD, "
                         "not in CFTR2 - not assumed null"),
    "3385ins8": dict(c="", p="p.(frameshift, legacy 3385ins8 = c.3253 region, exon 20)", vtype="frameshift", clinvar=None,
                     why="8-bp insertion in exon 20 (frameshift, NMD-competent); exact bases not given"),
    "c.3140-5C>G": dict(c="c.3140-5C>G", p="p.?", vtype="intronic", clinvar="c.3140-5C>G"),
    "c.3149C>T": dict(c="c.3149C>T", p="p.Pro1050Leu", vtype="missense", clinvar="c.3149C>T", seq=("sub", 3149, "C", "T", "P1050L")),
    "c.1718C>T": dict(c="c.1718C>T", p="p.Ser573Phe", vtype="missense", clinvar="c.1718C>T", seq=("sub", 1718, "C", "T", "S573F")),
    "p.Ser573Phe": dict(c="c.1718C>T", p="p.Ser573Phe", vtype="missense", clinvar="c.1718C>T", seq=("sub", 1718, "C", "T", "S573F")),
    "c.1270G>C": dict(c="c.1270G>C", p="p.Gly424Arg", vtype="missense", clinvar="c.1270G>C", seq=("sub", 1270, "G", "C", "G424R")),
    "c.3930G>C": dict(c="c.3930G>C", p="p.Trp1310Cys", vtype="missense", clinvar="c.3930G>C", seq=("sub", 3930, "G", "C", "W1310C")),
    "c.472_473insA": dict(c="c.472_473insA", p="p.Ser158LysfsTer5", vtype="frameshift", clinvar="c.472_473insA"),
    "c.1935delG": dict(c="c.1935del", p="p.Gly646AspfsTer17", vtype="frameshift", clinvar="c.1935del"),
    "c.2633_2634delinsT": dict(c="c.2633_2634delinsT", p="p.Val879fs", vtype="frameshift", clinvar="c.2633_2634delinsT"),
    "c.1323-1344del22": dict(c="c.1323_1344del", p="p.Asp443fs", vtype="frameshift", clinvar="c.1323_1344del"),
    "c.2998_3012del": dict(c="c.2998_3012del", p="p.(in-frame 5-aa deletion)", vtype="inframe", clinvar="c.2998_3012del"),
    "c.3092_3100dupCATATTTCC": dict(c="c.3092_3100dup", p="p.(in-frame 3-aa duplication)", vtype="inframe", clinvar="c.3092_3100dup"),
    "3271+ 18C/T": dict(c="c.3139+18C>T", p="p.?", vtype="intronic", clinvar="c.3139+18C>T"),
    "3791C/T": dict(c="c.3659C>T", p="p.Thr1220Ile", vtype="missense", clinvar="c.3659C>T", seq=("sub", 3659, "C", "T", "T1220I")),
    "T1220I": dict(c="c.3659C>T", p="p.Thr1220Ile", vtype="missense", clinvar="c.3659C>T", seq=("sub", 3659, "C", "T", "T1220I")),
    "4404C/T": dict(c="c.4272C>T", p="p.Tyr1424=", vtype="synonymous", clinvar="c.4272C>T", seq=("sub", 4272, "C", "T", "=")),
    "4521G/A": dict(c="c.4389G>A", p="p.Gln1463=", vtype="synonymous", clinvar="c.4389G>A", seq=("sub", 4389, "G", "A", "=")),
    "Q1463Q": dict(c="c.4389G>A", p="p.Gln1463=", vtype="synonymous", clinvar="c.4389G>A", seq=("sub", 4389, "G", "A", "=")),
    "p.Leu989TyrfsTer9": dict(c="", p="p.Leu989TyrfsTer9", vtype="frameshift", clinvar="p.Leu989TyrfsTer9"),
    "pG1123del": dict(c="", p="p.Gly1123del", vtype="inframe", clinvar="p.Gly1123del"),
    "3126del4": dict(c="", p="p.(frameshift, legacy 3126del4)", vtype="frameshift", clinvar=None),
    "3363delGT": dict(c="", p="p.(frameshift, legacy 3363delGT)", vtype="frameshift", clinvar=None),
    "V1160X": dict(c="", p="p.Val1160Ter", vtype="nonsense", clinvar="p.Val1160Ter"),
    "Y38H": dict(c="", p="p.Tyr38His", vtype="missense", clinvar="p.Tyr38His"),
    "T360R": dict(c="", p="p.Thr360Arg", vtype="missense", clinvar="p.Thr360Arg"),
    "Y515H": dict(c="", p="p.Tyr515His", vtype="missense", clinvar="p.Tyr515His"),
    "F533L": dict(c="", p="p.Phe533Leu", vtype="missense", clinvar="p.Phe533Leu"),
    "A412S": dict(c="", p="p.Ala412Ser", vtype="missense", clinvar="p.Ala412Ser"),
    "E1228G": dict(c="", p="p.Glu1228Gly", vtype="missense", clinvar="p.Glu1228Gly"),
    "N1432K": dict(c="", p="p.Asn1432Lys", vtype="missense", clinvar="p.Asn1432Lys"),
    "F994C": dict(c="", p="p.Phe994Cys", vtype="missense", clinvar="p.Phe994Cys"),
    "A1136V": dict(c="", p="p.Ala1136Val", vtype="missense", clinvar="p.Ala1136Val"),
    "R1102K": dict(c="", p="p.Arg1102Lys", vtype="missense", clinvar="p.Arg1102Lys"),
    "S557Y": dict(c="", p="p.Ser557Tyr", vtype="missense", clinvar="p.Ser557Tyr"),
    "D58G": dict(c="", p="p.Asp58Gly", vtype="missense", clinvar="p.Asp58Gly"),
    "IVS23+147 G>A": dict(c="", p="p.?", vtype="intronic", clinvar=None,
                          why="deep intronic; intron numbering (legacy vs new) ambiguous - identity not resolvable"),
    "3T;TG12": dict(c="", p="p.?", vtype="intronic", clinvar=None, why="rare 3T poly-T allele; not in CFTR2; clinical significance unknown"),
    "CFTRdup4-8": dict(c="", p="p.?", vtype="duplication", clinvar=None,
                       why="large duplication not in CFTR2; in-frame under both exon numberings (843 or 936 bp) - effect unknown"),
}

# Strings whose identity cannot be resolved (garbled / impossible residue / incomplete). Kept as 'uncertain'.
UNRESOLVABLE = {
    "E508X": "residue 508 is Phe on NM_000492.4 - 'E508X' cannot exist (possibly E585X or F508del)",
    "WR82X": "residue 82 is Met - not interpretable (possibly W882X)",
    "K732": "incomplete name; residue 732 is Leu (possibly L732X)",
    "Y1092": "incomplete name (probably Y1092X, but variant not stated)",
    "1948T": "uninterpretable legacy string",
    "W1282K": "impossible by a single substitution (TGG>AAG); probably W1282X",
    "G42X": "residue 42 is Ser - not interpretable",
    "R325Q": "residue 325 is Tyr - not interpretable (possibly R352Q)",
    "L511Q": "residue 511 is Ser - not interpretable",
    "K864S": "residue 864 is Ile - not interpretable",
    "R202Q": "residue 202 is Trp - not interpretable",
    "V407M": "residue 407 is Glu - not interpretable (possibly V470M polymorphism)",
    "c.1330G>C": "reference base at c.1330 is A on NM_000492.4 - description invalid",
    "c.1645G>A": "reference base at c.1645 is A on NM_000492.4 - description invalid (S549R is c.1645A>C)",
    "c.470G>A": "reference base at c.470 is T on NM_000492.4 - description invalid",
    "816delCTC": "legacy 816 = c.684; NM_000492.4 has TTT at c.684-686 and no CTC nearby - description invalid "
                 "(an in-frame 3-bp deletion either way)",
    "1342- 11TTT- >G": "uninterpretable poly-T/TG description (legacy 1342-11 lies in the IVS8 TG/T tract; possibly "
                       "1342-11T>G = 5T;TG12, but not stated)",
    "G178R, S466X": "two different CF-causing variants written in one allele slot (phase unknown; other slot holds R1070Q, "
                    "suggesting a split [S466X;R1070Q] complex allele)",
    "Q378Afs ( c.1130-1131insA) ----2789 +5G>A": "two variants (c.1130dupA frameshift and 2789+5G>A) written in one allele slot; "
                                                "probably the two alleles of a compound heterozygote but phase not recorded",
}

# Multi-variant strings: decision on the allele as a whole. 'name' must resolve via CFTR2/aliases.
COMPLEX = {
    "deltaF508; M470V": ("F508del", "high", "F508del with the benign M470V polymorphism in the same field"),
    "deltaF508; V470M; c.1393-61 A>G": ("F508del", "high", "F508del with benign polymorphisms (V470M, c.1393-61A>G) in the same field"),
    "1677delTA, M470V": ("1677delTA", "high", "1677delTA with the benign M470V polymorphism in the same field"),
    "D1152H; I1051V": ("D1152H", "medium", "D1152H plus I1051V (CFTR2: no interpretation); D1152H defines the allele"),
    "D1152H ;I1051V": ("D1152H", "medium", "D1152H plus I1051V (CFTR2: no interpretation); D1152H defines the allele"),
    "D1152H, I1051V": ("D1152H", "medium", "D1152H plus I1051V (CFTR2: no interpretation); D1152H defines the allele"),
    "D1152H;I1051V": ("D1152H", "medium", "D1152H plus I1051V (CFTR2: no interpretation); D1152H defines the allele"),
    "L732X; Q493P": ("L732X", "medium", "L732X (nonsense) with Q493P (no interpretation) in the same field; nonsense defines the allele"),
    "D110H; D58G": ("D110H", "medium", "D110H with D58G (not in CFTR2) in the same field; D110H defines the allele"),
    "3199del6; I148T": ("3199del6", "high", "known complex allele I148T;3199del6 - the CF-causing component is 3199del6 (I148T benign)"),
    "7T/7T-11TG/11TG": ("7T", "high", "poly-T/TG genotype 7T/7T, TG11/TG11 = reference (normal) tracts on both chromosomes"),
}

# Exon-level deletions / Turkish free text
FREE_TEXT = {"19 ve 21. ekzonlar arasında delesyon": ("CFTRdele19-21", "low",
                                                    "Turkish 'deletion between exons 19 and 21': read as deletion of exons 19-21 "
                                                    "(could also mean exon 20 only); large deletion either way"),
             "Exon13 insT": ("Exon13 insT", "low", "unspecified single-T insertion in exon 13 = frameshift (exact variant unknown)")}

# Complex alleles in a single slot that contain an ETI-listed variant in cis with a null variant: eligibility unknown.
ETI_COMPLEX_UNKNOWN = {"S466X;R1070Q": "R1070Q is Table-6 listed but lies in cis with the S466X nonsense (no protein)",
                       "3199del6; I148T": "only the benign I148T component is Table-6 listed; 3199del6 itself is not"}

# Specific function-group decisions that override the rules (reason given)
FUNCTION_OVERRIDE = {
    "5T;TG12": ("unresolved", "VCC splice-modulating poly-T allele; not part of MF/RF trial definitions"),
    "5T;TG13": ("unresolved", "VCC splice-modulating poly-T allele; not part of MF/RF trial definitions"),
    "5T": ("unresolved", "VCC splice-modulating poly-T allele (TG not stated); not part of MF/RF trial definitions"),
    "CFTRdele2": ("MF", "exon 2 deletion (111 bp, in-frame) - CFTR2 CF-causing; large deletion treated as class I/MF, "
                        "but it is in-frame and not on the published MF list - clinician review"),
    "S466X;R1070Q": ("MF", "complex allele with the S466X nonsense in cis - no full-length protein expected"),
}
FUNCTIONAL_CLASS["S466X;R1070Q"] = "I"
# Display names for curated non-CFTR2 variants written in several ways
NOT_IN_CFTR2_NAME = {"c.4061T>A": "M1354K", "C.4061 T>A": "M1354K", "c.3149C>T": "P1050L", "c.1718C>T": "S573F",
                     "p.Ser573Phe": "S573F", "c.1270G>C": "G424R", "c.3930G>C": "W1310C", "3791C/T": "T1220I",
                     "T1220I": "T1220I", "4404C/T": "Y1424Y", "4521G/A": "Q1463Q", "Q1463Q": "Q1463Q",
                     "3271+ 18C/T": "3271+18C>T", "pG1123del": "G1123del", "p.Leu989TyrfsTer9": "p.Leu989TyrfsTer9",
                     "c.472_473insA": "c.472_473insA", "c.1935delG": "c.1935del", "c.2633_2634delinsT": "c.2633_2634delinsT",
                     "c.1323-1344del22": "c.1323_1344del", "c.2998_3012del": "c.2998_3012del",
                     "c.3092_3100dupCATATTTCC": "c.3092_3100dup", "c.3874-3C>G": "c.3874-3C>G", "c.3140-5C>G": "c.3140-5C>G",
                     "3874-3C>G": "c.3874-3C>G", "711+9A-G": "711+9A>G", "4444delC": "4444delC", "3385ins8": "3385ins8"}


def _nk_dict(d: dict) -> dict:
    return {nkey(k): v for k, v in d.items()}


def lookup(d: dict, s: str):
    """Exact raw-string match first, then normalised-key match (case/space/arrow insensitive)."""
    if s in d:
        return s, d[s]
    nd = {nkey(k): k for k in d}
    k = nd.get(nkey(s))
    return (k, d[k]) if k is not None else (None, None)

# ======================================================================================================================
# 2. Normalisation helpers
# ======================================================================================================================
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H",
       "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V", "Ter": "X"}


def nkey(s) -> str:
    """Canonical matching key for an allele token (case/whitespace/arrow/'/' insensitive; c.-del bases dropped)."""
    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
    s = s.strip().replace("→", "->").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"-*>", ">", s)                                  # '->', '-->', '- >' -> '>'
    s = re.sub(r"([0-9])([ACGTacgt])-([ACGTacgt])$", r"\1\2>\3", s)  # '2789+5G-A' -> '2789+5G>A'
    s = re.sub(r"([ACGTacgt])/([ACGTacgt])$", r"\1>\2", s)       # '4521G/A' -> '4521G>A'
    if s[:2].lower() == "c.":
        s = "c." + s[2:].upper()
        s = re.sub(r"^c\.(\d+)-(\d+)(DEL|INS|DUP)", r"c.\1_\2\3", s)
        s = re.sub(r"(DEL|DUP)[ACGT]+$", r"\1", s)
        return s
    if s[:2].lower() == "p.":
        body = s[2:]
        for k, v in AA3.items():
            body = body.replace(k, v)
        return "P." + body.upper()
    return s.upper()


def short_protein(p) -> str | None:
    """'p.Gly542X' -> 'G542X'; 'p.Glu528=' -> 'E528E'."""
    m = re.match(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|X|=)$", str(p))
    if not m:
        return None
    a, n, b = m.groups()
    a1 = AA3.get(a)
    b1 = a1 if b == "=" else ("X" if b == "X" else AA3.get(b))
    return f"{a1}{n}{b1}" if a1 and b1 else None


def hgvs_p(p) -> str:
    p = "" if p is None or (isinstance(p, float) and math.isnan(p)) else str(p)
    return re.sub(r"X(\d*)\b", r"Ter\1", p) if p.startswith("p.") else p


# ======================================================================================================================
# 3. External sources (cached)
# ======================================================================================================================
def _download(url: str, dest: Path) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def fetch_cftr2() -> dict:
    src = SOURCES["cftr2"]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cftr2.xlsx"
        sha = _download(src["url"], p)
        raw = pd.read_excel(p, sheet_name=0, header=None)
    hdr_row = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Variant legacy name")][0]
    header = [str(x) for x in raw.iloc[hdr_row].tolist()]
    preamble = [str(x) for x in raw.iloc[:hdr_row, 0].dropna().tolist()]
    body = raw.iloc[hdr_row + 1:].copy()
    body.columns = ["legacy", "protein", "cdna", "alt", "n_alleles", "freq", "det_prev", "det_cur", "changed"]
    body = body[body.det_cur.notna() & body.legacy.notna()]
    def _cell(k, v):
        if isinstance(v, float) and math.isnan(v):
            return None
        if k == "n_alleles":
            try:
                return int(v)
            except (TypeError, ValueError):
                return str(v)                                   # e.g. 'see note below' (I1027T complex)
        return str(v).strip()
    rows = [{k: _cell(k, v) for k, v in r.items() if k in ("legacy", "protein", "cdna", "alt", "n_alleles", "det_prev", "det_cur")}
            for r in body.to_dict("records")]
    return {"url": src["url"], "landing": src["landing"], "version": src["version"], "accessed": ACCESS_DATE,
            "sha256": sha, "preamble": preamble, "header": header, "rows": rows}


def _label_tokens(pdf_path: Path, table_title: str, end_markers: list[str]) -> list[tuple[str, str]]:
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    lines = [ln.strip() for page in doc for ln in page.get_text().splitlines()]
    starts = [i for i, ln in enumerate(lines) if ln.startswith(table_title)]
    if not starts:
        raise RuntimeError(f"table '{table_title}' not found")
    i0 = starts[-1] + 1
    out, basis = [], "listed"
    for ln in lines[i0:]:
        if any(ln.startswith(e) for e in end_markers):
            break
        if not ln or ln.startswith("Reference ID") or re.fullmatch(r"\d+", ln):
            continue
        if ln.startswith("Mutations responsive"):
            basis = ("clinical" if "clinical data" in ln else "in vitro" if "in vitro" in ln else
                     "extrapolation" if "extrapolation" in ln else ln)
            continue
        clinical_mark = "*" in ln or "⁎" in ln
        name = re.sub(r"[\*†‡§⁎\^]", "", ln).strip()
        if not name:
            continue
        out.append((name, basis if basis != "listed" else ("clinical" if clinical_mark else "in vitro")))
    return out


def fetch_label(key: str) -> dict:
    src = SOURCES[key]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "label.pdf"
        sha = _download(src["url"], p)
        import fitz
        first = fitz.open(str(p))[0].get_text()
        rev = re.search(r"Revised:\s*([0-9/]+)", first)
        refid = re.search(r"Reference ID:\s*(\d+)", first)
        toks = _label_tokens(p, src["table"], src["end"])
    return {"url": src["url"], "version": src["version"], "accessed": ACCESS_DATE, "sha256": sha,
            "revised": rev.group(1) if rev else None, "reference_id": refid.group(1) if refid else None,
            "table": src["table"], "variants": [{"name": n, "basis": b} for n, b in toks]}


def _entrez():
    from Bio import Entrez
    Entrez.email, Entrez.tool = NCBI_EMAIL, NCBI_TOOL
    return Entrez


def fetch_refseq() -> dict:
    from Bio import SeqIO
    import Bio
    E = _entrez()
    h = E.efetch(db="nucleotide", id=SOURCES["refseq"]["accession"], rettype="gb", retmode="text")
    rec = SeqIO.read(h, "genbank")
    h.close()
    cds = [f for f in rec.features if f.type == "CDS"][0]
    exons = [[int(f.location.start) + 1, int(f.location.end)] for f in rec.features if f.type == "exon"]
    return {"accession": rec.id, "description": rec.description, "accessed": ACCESS_DATE, "biopython": Bio.__version__,
            "cds_start_1based": int(cds.location.start) + 1, "cds_end_1based": int(cds.location.end),
            "exons_mrna_1based": exons, "seq": str(rec.seq)}


def fetch_clinvar(term: str) -> dict:
    E = _entrez()
    q = f'CFTR[gene] AND "{term}"'
    h = E.esearch(db="clinvar", term=q, retmax=20)
    ids = E.read(h)["IdList"]
    h.close()
    hits = []
    if ids:
        h = E.esummary(db="clinvar", id=",".join(ids), retmode="json")
        d = json.loads(h.read())
        h.close()
        for uid in d["result"]["uids"]:
            r = d["result"][uid]
            g = r.get("germline_classification") or {}
            hits.append({"uid": uid, "accession": r.get("accession"), "title": r.get("title"),
                         "classification": g.get("description"), "review_status": g.get("review_status"),
                         "last_evaluated": g.get("last_evaluated")})
    time.sleep(0.4)
    return {"query": q, "accessed": ACCESS_DATE, "hits": hits}


def load_cache(refresh: bool, offline: bool) -> dict:
    cache = json.loads(CACHE_JSON.read_text()) if CACHE_JSON.exists() and not refresh else {}
    changed = False

    def need(k):
        return k not in cache

    for key, fn in [("cftr2", fetch_cftr2), ("fda_eti", lambda: fetch_label("fda_eti")),
                    ("fda_iva", lambda: fetch_label("fda_iva")), ("fda_tezi", lambda: fetch_label("fda_tezi")),
                    ("refseq", fetch_refseq)]:
        if need(key):
            if offline:
                raise SystemExit(f"--offline but '{key}' missing from {CACHE_JSON}")
            print(f"  fetching {key} ...")
            cache[key] = fn()
            changed = True
    cache.setdefault("clinvar", {})
    terms = sorted({v["clinvar"] for v in NOT_IN_CFTR2.values() if v.get("clinvar")})
    for t in terms:
        if t not in cache["clinvar"]:
            if offline:
                raise SystemExit(f"--offline but ClinVar term '{t}' missing from cache")
            print(f"  ClinVar: {t}")
            cache["clinvar"][t] = fetch_clinvar(t)
            changed = True
    if changed:
        cache["_meta"] = {"written": dt.datetime.now().isoformat(timespec="seconds"), "access_date": ACCESS_DATE,
                          "note": "Public reference data only (no patient data). Delete or run --refresh to re-download."}
        REF.mkdir(parents=True, exist_ok=True)
        CACHE_JSON.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return cache


# ======================================================================================================================
# 4. Reference indices
# ======================================================================================================================
class Refs:
    def __init__(self, cache: dict):
        self.cftr2 = cache["cftr2"]["rows"]
        self.by_key: dict[str, dict] = {}
        for r in self.cftr2:                 # first (most frequent) row wins for a key
            keys = [r["legacy"], short_protein(r["protein"]) or ""]
            keys += (r["cdna"] or "").split("|")
            keys += [a.strip() for a in (r["alt"] or "").split(",")]
            for k in keys:
                if k and k != "nan":
                    self.by_key.setdefault(nkey(k), r)
        self.by_legacy = {r["legacy"]: r for r in self.cftr2}
        self.eti = {nkey(v["name"]): v for v in cache["fda_eti"]["variants"]}
        self.iva = {nkey(v["name"]): v for v in cache["fda_iva"]["variants"]}
        self.tezi = {nkey(v["name"]): v for v in cache["fda_tezi"]["variants"]}
        rs = cache["refseq"]
        self.cds = rs["seq"][rs["cds_start_1based"] - 1: rs["cds_end_1based"]]
        self.clinvar = cache["clinvar"]

    # --- sequence utilities (NM_000492.4) ---
    def _aa(self, s):
        from Bio.Seq import Seq
        return str(Seq(s[: len(s) // 3 * 3]).translate())

    def verify(self, spec) -> str:
        kind = spec[0]
        c = self.cds
        if kind == "del":
            _, a, b, a2, b2 = spec
            ok = (c[: a - 1] + c[b:]) == (c[: a2 - 1] + c[b2:])
        elif kind == "dupA":
            _, a, a2 = spec
            ok = (c[:a] + "A" + c[a:]) == (c[:a2] + "A" + c[a2:])
        elif kind == "ref":
            _, pos, base = spec
            ok = c[pos - 1] == base
        elif kind == "sub":
            _, pos, ref, alt, prot = spec
            if c[pos - 1] != ref:
                raise AssertionError(f"reference mismatch at c.{pos}: {c[pos-1]} != {ref}")
            w, m = self._aa(c), self._aa(c[: pos - 1] + alt + c[pos:])
            diff = [i for i in range(len(w)) if w[i] != m[i]]
            got = "=" if not diff else f"{w[diff[0]]}{diff[0]+1}{m[diff[0]]}"
            ok = got == prot
        else:
            raise ValueError(kind)
        if not ok:
            raise AssertionError(f"sequence check failed: {spec}")
        return "verified on NM_000492.4"

    def label_hit(self, idx: dict, row: dict | None, extra_names=()) -> dict | None:
        names = list(extra_names)
        if row:
            names += [row["legacy"], short_protein(row["protein"]) or ""]
            names += [a.strip() for a in (row["alt"] or "").split(",")]
        for n in names:
            if n and nkey(n) in idx:
                return idx[nkey(n)]
        return None

    def clinvar_summary(self, term):
        rec = self.clinvar.get(term)
        if not rec:
            return None, "not queried"
        hits = rec["hits"]
        if not hits:
            return None, "no ClinVar record"
        # exact HGVS (c. or p.) must appear in the record title; compound/haplotype records are ignored
        exact = [h for h in hits if term in (h["title"] or "") and ";" not in (h["title"] or "")]
        if not exact:
            return None, f"ClinVar: no exact record for {term} ({len(hits)} non-exact hits ignored)"
        classes = sorted({h["classification"] for h in exact if h["classification"]})
        acc = ",".join(h["accession"] for h in exact[:3])
        cls = classes[0] if len(classes) == 1 else None
        return cls, f"ClinVar {acc}: {'; '.join(classes)}" + (" (records disagree)" if len(classes) > 1 else "")


# ======================================================================================================================
# 5. Classification of one allele token
# ======================================================================================================================
STATUS_MAP = {"CF-causing": "CF-causing", "Varying clinical consequence": "varying clinical consequence",
              "Non CF-causing": "non-CF-causing", "No interpretation available": "unknown significance"}
SLOT_FROM_CFTR2 = {"CF-causing": "CF-causing", "varying clinical consequence": "VCC", "non-CF-causing": "non-CF-causing",
                   "unknown significance": "uncertain"}


def vtype_of(row: dict | None, name: str = "") -> str:
    leg = (row or {}).get("legacy") or name
    prot = (row or {}).get("protein") or ""
    cdna = (row or {}).get("cdna") or ""
    if re.match(r"(?i)^cftr(dele|del)", leg):
        return "large deletion"
    if re.match(r"(?i)^cftrdup", leg):
        return "duplication"
    if ";" in leg or prot.startswith("p.["):
        return "complex"
    if re.search(r"fsX|fsTer", prot):
        return "frameshift"
    if re.search(r"X$|Ter$|\*$", prot):
        return "nonsense"
    if prot == "p.?" and re.search(r"c\.1A|c\.2T|c\.3G", cdna):
        return "start loss"
    if re.search(r"[+-][12](?:[ACGT]>|del|dup|ins|_)", cdna.split("|")[0]):
        return "canonical splice"
    if re.search(r"\d[+-]\d", cdna):
        return "intronic/non-canonical splice"
    if prot.endswith("=") or re.match(r"^[A-Z]\d+([A-Z])$", leg) and leg[0] == leg[-1]:
        return "synonymous"
    if re.search(r"del$|dup$|ins", prot):
        return "in-frame indel"
    if re.match(r"^c\.-", cdna) or cdna.startswith("polyT") or re.match(r"^\d+T(;TG\d+)?$", leg):
        return "UTR/poly-T"
    if re.match(r"^p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}$", prot):
        return "missense"
    return "other"


NULL_TYPES = {"frameshift", "nonsense", "canonical splice", "large deletion", "start loss"}


def classify_row(row: dict | None, refs: Refs, name: str, conf: str, source: str, notes: list[str],
                 status_override: str | None = None, vtype: str | None = None, hgvs=("", "")) -> dict:
    """Derive all attributes for an identified variant (CFTR2 row or curated non-CFTR2 variant)."""
    cftr2_status = STATUS_MAP.get(row["det_cur"], "unknown significance") if row else "not in CFTR2"
    vt = vtype or vtype_of(row, name)
    legacy = row["legacy"] if row else name
    # ---- slot status
    if legacy == "F508del":
        slot = "F508del"
    elif status_override:
        slot = status_override
    elif row:
        slot = SLOT_FROM_CFTR2[cftr2_status]
        if slot == "uncertain" and vt in NULL_TYPES and legacy not in NULL_EXCEPTIONS:
            slot = "CF-causing"
            notes.append(f"CFTR2 has no interpretation but variant type is {vt} (null) -> treated as CF-causing by rule")
            conf = "medium" if conf == "high" else conf
    else:
        slot = "uncertain"
    # ---- label lists
    short = short_protein(row["protein"]) if row else None
    extra = [legacy] + ([short] if short else [])
    eti_hit = refs.label_hit(refs.eti, row, extra)
    iva_hit = refs.label_hit(refs.iva, row, extra)
    tezi_hit = refs.label_hit(refs.tezi, row, extra)
    # ---- function group
    detail = ""
    if slot == "F508del":
        fg, detail = "F508del", "F508del"
    elif slot == "non-CF-causing":
        fg, detail = "non-CF-causing", "non-CF-causing (benign/polymorphism per CFTR2 or ClinVar)"
    elif slot in ("uncertain", "missing"):
        fg, detail = "unresolved", "pathogenicity not established (no CFTR2 interpretation / not in CFTR2)"
        if iva_hit or tezi_hit:
            detail += "; listed as modulator-responsive in FDA labels (would be RF/other if CF-causing)"
    elif legacy in FUNCTION_OVERRIDE:
        fg, why = FUNCTION_OVERRIDE[legacy]
        detail = f"curated: {why}"
        if legacy == "CFTRdele2":
            conf = "medium" if conf == "high" else conf
    elif legacy in GATING or short in GATING:
        fg, detail = "gating", "gating (class III; KALYDECO clinical gating set)"
    elif legacy in MF_MISSENSE_SMPC or legacy in MF_MISSENSE_TRIAL:
        fg = "MF"
        detail = ("MF: missense/in-frame named as minimal function in Kaftrio SmPC (study 445-102)" if legacy in MF_MISSENSE_SMPC
                  else "MF: missense/in-frame on the VX-445-102/VX-661 F/MF eligibility list (as recalled; verify)")
        if legacy not in MF_MISSENSE_SMPC:
            conf = "medium" if conf == "high" else conf
    elif (iva_hit and legacy not in GATING) or (tezi_hit and tezi_hit["basis"] == "clinical" and legacy != "F508del"):
        fg = "RF"
        detail = "RF: " + ("ivacaftor-responsive (KALYDECO Table 3)" if iva_hit else "") + \
                 ("; " if iva_hit and tezi_hit and tezi_hit["basis"] == "clinical" else "") + \
                 ("TEZ/IVA residual-function trial variant (SYMDEKO Table 6 *)" if tezi_hit and tezi_hit["basis"] == "clinical" else "")
    elif vt in NULL_TYPES and legacy not in NULL_EXCEPTIONS:
        fg = "MF"
        detail = f"MF by trial rule: {vt} (no CFTR protein expected)"
        if vt == "large deletion" and legacy != "CFTRdele2,3":     # CFTRdele2,3 is named in the Kaftrio SmPC MF list
            conf = "medium" if conf == "high" else conf
    else:
        fg = "unresolved"
        detail = "CF-causing/VCC but not MF, RF or gating by the trial definitions"
        if legacy in NULL_EXCEPTIONS:
            detail = "curated: " + NULL_EXCEPTIONS[legacy]
        elif tezi_hit:
            detail += " (TEZ/IVA- or ETI-responsive in vitro: corrector-responsive processing variant)"
        elif legacy == "R334W":
            detail += " (classic class IV conductance variant - 'residual function' under class-based schemes)"
    # ---- functional class
    if legacy in FUNCTIONAL_CLASS:
        fclass = FUNCTIONAL_CLASS[legacy]
    elif fg == "non-CF-causing":
        fclass = "not applicable"
    elif fg == "gating":
        fclass = "III"
    elif fg == "MF" and vt in NULL_TYPES:
        fclass = "I" + (" (in-frame deletion; best available)" if legacy == "CFTRdele2" else "")
    elif fg == "MF":
        fclass = "II (best available)"
    elif fg == "RF":
        fclass = "IV/V (residual function; best available)"
    else:
        fclass = "unknown"
    # ---- ETI eligibility (FDA Table 6, Dec 2024)
    if slot == "F508del":
        eti, eti_basis = "yes", "F508del"
    elif legacy in ETI_COMPLEX_UNKNOWN:
        eti, eti_basis = "unknown", ETI_COMPLEX_UNKNOWN[legacy]
    elif eti_hit:
        eti, eti_basis = "yes", f"Table 6 '{eti_hit['name']}' ({eti_hit['basis']})"
    elif conf == "low":
        eti, eti_basis = "unknown", "identity uncertain"
    else:
        eti, eti_basis = "no", "not in Table 6"
    if eti == "yes" and slot == "non-CF-causing":
        notes.append("listed in TRIKAFTA Table 6 although CFTR2/ClinVar classify it non-CF-causing")
    if row is None:
        c_, p_ = hgvs
    else:
        c_, p_ = (row["cdna"] or ""), hgvs_p(row["protein"])
    return dict(normalised_legacy_name=legacy, hgvs_c=c_, hgvs_p=p_, cftr2_status=cftr2_status, functional_class=fclass,
                function_group=fg, function_group_detail=detail, slot_status=slot, variant_type=vt,
                eti_fda_eligible=eti, eti_basis=eti_basis,
                ivacaftor_label="yes" if iva_hit else "no", tezacaftor_ivacaftor_label="yes" if tezi_hit else "no",
                cftr2_alleles=row["n_alleles"] if row else None, source=source, confidence=conf, note="; ".join(notes))


def missing_row(raw: str, why: str) -> dict:
    return dict(normalised_legacy_name="(missing)", hgvs_c="", hgvs_p="", cftr2_status="not in CFTR2", functional_class="",
                function_group="unresolved", function_group_detail="no variant information", slot_status="missing",
                variant_type="missing", eti_fda_eligible="unknown", eti_basis="no variant information",
                ivacaftor_label="", tezacaftor_ivacaftor_label="", cftr2_alleles=None, source="registry token",
                confidence="high", note=why)


def uncertain_row(raw: str, why: str, name: str | None = None) -> dict:
    clean = re.sub(r"\s+", " ", raw).strip()
    name = name or f"(unresolved) {clean}"
    return dict(normalised_legacy_name=name, hgvs_c="", hgvs_p="", cftr2_status="not in CFTR2", functional_class="unknown",
                function_group="unresolved", function_group_detail="variant identity not resolvable",
                slot_status="uncertain", variant_type="unresolved", eti_fda_eligible="unknown",
                eti_basis="identity uncertain", ivacaftor_label="", tezacaftor_ivacaftor_label="", cftr2_alleles=None,
                source="curated (UNRESOLVABLE)", confidence="low", note=why)


def deletion_spec(tok: str):
    """'CFTR dele 18 -20' -> ('18-20', [18,19,20]) ; returns None if not an exon-deletion string."""
    s = re.sub(r"\s+", "", tok).lower()
    m = re.match(r"^(?:cftr|ctfr)-?dele(.+)$", s)
    if not m:
        return None
    spec = m.group(1)
    nums = [int(x) for x in re.findall(r"\d+", spec)]
    if "," in spec:
        exons = nums
    elif "-" in spec and len(nums) >= 2:
        exons = list(range(nums[0], nums[-1] + 1))
    else:
        exons = nums
    contiguous = exons == list(range(exons[0], exons[-1] + 1))
    if len(exons) == 1:
        txt = str(exons[0])
    elif len(exons) == 2 or not contiguous:              # CFTR2 style: 'CFTRdele2,3', 'CFTRdele22,23'
        txt = ",".join(str(n) for n in exons)
    else:                                                 # CFTR2 style: 'CFTRdele2-4', 'CFTRdele22-24'
        txt = f"{exons[0]}-{exons[-1]}"
    return txt, exons


def classify_token(raw: str, refs: Refs) -> dict:
    s = "" if raw is None else str(raw)
    low = s.strip().lower()
    if low in MISSING_TOKENS:
        return missing_row(s, MISSING_TOKENS[low])
    k_un, why_un = lookup(UNRESOLVABLE, s)
    if k_un is not None:
        return uncertain_row(s, why_un)
    k_ft, ft = lookup(FREE_TEXT, s)
    if ft and ft[0] == "Exon13 insT":
        r = classify_row(None, refs, "Exon13 insT", "low", "curated (FREE_TEXT)", [ft[2]],
                         status_override="CF-causing", vtype="frameshift", hgvs=("", "p.(frameshift)"))
        return r
    notes: list[str] = []
    # --- exon deletions / duplications
    tok = ft[0] if ft else s
    dspec = deletion_spec(tok)
    if dspec:
        txt, exons = dspec
        name = f"CFTRdele{txt}"
        row = refs.by_key.get(nkey(name))
        conf = "high" if (row and txt in ("2", "2,3")) else "medium"
        if ft:
            conf = ft[1]
            notes.append(ft[2])
        if not row:
            notes.append("large exon deletion not listed in CFTR2 under this name; exon numbering (legacy vs new) not stated; "
                         "multi-exon/whole-exon deletion treated as null (class I) by rule")
        else:
            notes.append("CFTR2 legacy exon numbering assumed")
        r = classify_row(row, refs, name, conf, "CFTR2" if row else "rule: large deletion", notes,
                         status_override=None if row else "CF-causing", vtype="large deletion")
        if name == "CFTRdele2,3" and not ft:
            r["confidence"] = "high"
        return r
    # --- multi-variant strings
    k_cx, cx = lookup(COMPLEX, s)
    if cx:
        target, conf, why = cx
        if target == "7T":
            return dict(normalised_legacy_name="7T/7T;TG11/TG11 (normal)", hgvs_c="", hgvs_p="", cftr2_status="not in CFTR2",
                        functional_class="not applicable", function_group="non-CF-causing", function_group_detail="reference poly-T/TG tracts",
                        slot_status="non-CF-causing", variant_type="UTR/poly-T", eti_fda_eligible="no",
                        eti_basis="not a variant", ivacaftor_label="no", tezacaftor_ivacaftor_label="no", cftr2_alleles=None,
                        source="curated (COMPLEX)", confidence=conf, note=why)
        row = refs.by_key[nkey(target)]
        r = classify_row(row, refs, target, conf, "curated (COMPLEX) + CFTR2", [why])
        if k_cx in ETI_COMPLEX_UNKNOWN:
            r["eti_fda_eligible"], r["eti_basis"] = "unknown", ETI_COMPLEX_UNKNOWN[k_cx]
        return r
    # --- IVS canonical splice with ambiguous numbering
    k_iv, iv = lookup(IVS_CANONICAL, s)
    if iv:
        notes.append(f"intron numbering ambiguous: {iv}; every reading is a canonical splice-site variant")
        disp = re.sub(r"\s+", "", k_iv).replace("IVS9as-1G>A", "IVS9-1G>A")
        cands = " or ".join(re.findall(r"c\.[0-9+\-]+[ACGT]>[ACGT]", iv))
        r = classify_row(None, refs, disp, "medium", "curated (IVS_CANONICAL)", notes,
                         status_override="CF-causing", vtype="canonical splice", hgvs=(cands, "p.?"))
        return r
    # --- aliases (with optional sequence verification)
    k_al, al = lookup(ALIASES, s)
    if al:
        target, conf, why, spec = al
        notes.append(why)
        if spec:
            notes.append(refs.verify(spec))
        row = refs.by_key.get(nkey(target))
        if row is None:
            raise KeyError(f"alias target {target!r} not in CFTR2")
        return classify_row(row, refs, target, conf, "curated alias -> CFTR2", notes)
    # --- curated non-CFTR2 variants
    k_nc, spec = lookup(NOT_IN_CFTR2, s)
    if spec:
        if spec.get("seq"):
            notes.append(refs.verify(spec["seq"]))
        if spec.get("why"):
            notes.append(spec["why"])
        conf, status = "medium", None
        src = "curated (NOT_IN_CFTR2)"
        if spec.get("clinvar"):
            cls, txt = refs.clinvar_summary(spec["clinvar"])
            notes.append(txt)
            src += " + ClinVar"
            if cls and re.search(r"(?i)^(pathogenic|likely pathogenic|pathogenic/likely pathogenic)$", cls):
                status = "CF-causing"
            elif cls and re.search(r"(?i)^(benign|likely benign|benign/likely benign)$", cls):
                status = "non-CF-causing"
        if status is None and spec["vtype"] in NULL_TYPES:
            status = "CF-causing"
            notes.append(f"{spec['vtype']} -> null variant, treated as CF-causing by rule")
        if status is None:
            status = "uncertain"
            if spec["vtype"] in ("intronic", "duplication") and not spec.get("clinvar"):
                conf = "low"
        name = NOT_IN_CFTR2_NAME.get(k_nc) or re.sub(r"\s+", "", k_nc)
        return classify_row(None, refs, name, conf, src, notes, status_override=status, vtype=spec["vtype"],
                            hgvs=(spec["c"], spec["p"]))
    # --- generic normalisation: strip parenthetical codon / base annotations
    t = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    if t != s:
        notes.append(f"parenthetical annotation removed ('{s[len(t):].strip()}')")
    t = re.sub(r"^p(?=[A-Z]\d)", "", t)
    k = nkey(t)
    row = refs.by_key.get(k)
    if row is None and k.startswith("P."):          # p.Xxx one-letter after conversion
        row = refs.by_key.get(k[2:])
    if row is not None:
        trivial = (nkey(row["legacy"]) == k)
        conf = "high"
        how = "CFTR2 legacy name" if trivial else "CFTR2 cDNA/protein/alternative name"
        if not trivial:
            notes.append(f"matched CFTR2 via {how}")
        if re.search(r"\s|- ?>|-->", s):
            notes.append("spacing/arrow normalised")
        return classify_row(row, refs, row["legacy"], conf, f"CFTR2 ({how})", notes)
    return uncertain_row(s, "no match in CFTR2 and no curated decision (new string?)")


# ======================================================================================================================
# 6. Legacy regex approach (copied verbatim from scripts/00_build_cohort.py for the comparison table)
# ======================================================================================================================
_UNKNOWN = {"unknown", "not done", "nan", "", "none"}
_F508_RE = re.compile(r"^(\[?delta\]?)?f508(del)?$", re.I)
_PTC_RE = re.compile(r"(X$|X\b|del[ACGT]+$|ins[ACGT]+|AA-?\s*>\s*G|CFTRdele)", re.I)
_RF_ALLELES = re.compile(r"^(2789\+\s*5G\s*-?>\s*A|3849\+10KBC\s*-?>\s*T|D1152H|R117H|R334W|A455E|P205S|L997F|D110H|E92K|F1052V|"
                         r"3272-26A\s*-?>\s*G|R347H|S1235R|G178R|R352Q|D1270N|T338I|L206W|F1074L|S945L|V232D|R1070W|I148T)$", re.I)


def legacy_allele_status(mut, other):
    s = str(mut).strip()
    first = s.split(";")[0].strip()
    if _F508_RE.match(first):
        return "F508"
    if s.lower() in _UNKNOWN:
        return "UNK"
    if s.lower() == "other":
        o = str(other).strip()
        return "F508" if _F508_RE.match(o.split(";")[0].strip()) else "NON"
    return "NON"


def legacy_allele_class(mut, other):
    s = str(mut).strip(); first = s.split(";")[0].strip()
    if _F508_RE.match(first):
        return "F508del"
    if s.lower() in _UNKNOWN:
        return "unknown"
    name = str(other).strip() if s.lower() == "other" else first
    if name.lower() in _UNKNOWN or name.lower() == "other":
        return "other_unclassified"
    if _PTC_RE.search(name) or re.search(r"\+1G|\-1G|\+2T|\-2A|CFTRdele", name, re.I):
        return "PTC_or_MF"
    if _RF_ALLELES.match(name.replace(" ", "")) or _RF_ALLELES.match(name):
        return "residual"
    return "MF_missense_or_other"


def legacy_problem(row: dict, old_status: str, old_class: str) -> str:
    new_slot, fg = row["slot_status"], row["function_group"]
    probs = []
    if old_status == "NON" and new_slot in ("non-CF-causing", "uncertain", "missing"):
        probs.append({"non-CF-causing": "polymorphism counted as a real second mutation",
                      "uncertain": "unresolved/VUS string counted as an identified second mutation",
                      "missing": "unspecified 'other' counted as an identified mutation"}[new_slot])
    if old_status == "NON" and new_slot == "F508del":
        probs.append("F508del written as c.-name missed")
    if old_status == "F508" and new_slot != "F508del":
        probs.append("wrongly counted as F508del")
    if old_class == "residual" and fg != "RF":
        probs.append(f"regex RF but now {fg}")
    if old_class in ("PTC_or_MF", "MF_missense_or_other") and fg == "RF":
        probs.append("RF allele missed (regex MF)")
    if old_class == "MF_missense_or_other" and fg in ("non-CF-causing",):
        probs.append("polymorphism assumed minimal function")
    if old_class == "MF_missense_or_other" and fg == "unresolved" and new_slot in ("CF-causing", "VCC"):
        probs.append("non-MF CF-causing variant assumed minimal function")
    if old_class == "PTC_or_MF" and fg not in ("MF", "F508del") and new_slot not in ("missing",):
        probs.append(f"regex PTC/MF but now {fg}")
    return "; ".join(probs)


# ======================================================================================================================
# 7. Main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    print("[1] external references")
    cache = load_cache(args.refresh, args.offline)
    refs = Refs(cache)
    print(f"    CFTR2 rows={len(refs.cftr2)} | ETI Table 6={len(cache['fda_eti']['variants'])} | "
          f"IVA Table 3={len(cache['fda_iva']['variants'])} | TEZ/IVA Table 6={len(cache['fda_tezi']['variants'])}")

    print("[2] raw strings")
    raw = pd.read_excel(RAW_XLSX, usecols=["PatCode", "DateVisitYear", "Mutation1", "OtherMutation1", "Mutation2", "OtherMutation2"])
    slots = []                                           # (PatCode, effective raw string, source column, mut, other)
    for mcol, ocol in [("Mutation1", "OtherMutation1"), ("Mutation2", "OtherMutation2")]:
        for pc, m, o in zip(raw.PatCode, raw[mcol], raw[ocol]):
            ms = "nan" if (isinstance(m, float) and math.isnan(m)) else str(m)
            has_o = not (o is None or (isinstance(o, float) and math.isnan(o)))
            eff = str(o) if (ms.strip().lower() == "other" and has_o) else ms
            slots.append((pc, eff, mcol if eff == ms else ocol, m, o))
    sl = pd.DataFrame(slots, columns=["PatCode", "raw", "col", "mut", "other"])
    agg = sl.groupby("raw").agg(n_alleles=("PatCode", "size"), n_patients=("PatCode", "nunique"),
                                columns=("col", lambda x: ",".join(sorted(set(x))))).reset_index()
    n_other_placeholder = int((sl.mut.astype(str).str.strip().str.lower() == "other").sum())

    print("[3] classify")
    rows = []
    for r in agg.itertuples():
        c = classify_token(r.raw, refs)
        if r.raw.strip().lower() == "other":
            c["note"] = (c["note"] + f"; {n_other_placeholder} slots carried the 'other' placeholder in total, "
                         f"{r.n_alleles} of them without an OtherMutation string").strip("; ")
        first = sl[sl.raw == r.raw].iloc[0]
        old_s = legacy_allele_status(first.mut, first.other)
        old_c = legacy_allele_class(first.mut, first.other)
        rows.append({"raw_string": r.raw, "n_alleles": int(r.n_alleles), "n_patients": int(r.n_patients),
                     **c, "raw_columns": r.columns, "legacy_regex_status": old_s, "legacy_regex_class": old_c,
                     "legacy_regex_problem": legacy_problem(c, old_s, old_c)})
    vm = pd.DataFrame(rows).sort_values(["n_alleles", "raw_string"], ascending=[False, True])
    cols = ["raw_string", "n_alleles", "n_patients", "normalised_legacy_name", "hgvs_c", "hgvs_p", "cftr2_status",
            "functional_class", "function_group", "eti_fda_eligible", "source", "confidence", "note",
            "slot_status", "function_group_detail", "variant_type", "eti_basis", "ivacaftor_label",
            "tezacaftor_ivacaftor_label", "cftr2_alleles", "raw_columns", "legacy_regex_status", "legacy_regex_class",
            "legacy_regex_problem"]
    vm = vm[cols]
    vm.to_csv(OUT_CSV, index=False)
    print(f"    wrote {OUT_CSV.relative_to(REPO)}  ({len(vm)} distinct raw strings)")
    unmatched = vm[vm.source.str.contains("UNRESOLVABLE") | vm.note.str.contains("no curated decision")]
    print(f"    unresolved identities: {len(unmatched)} strings / {int(unmatched.n_alleles.sum())} allele slots")
    new = vm[vm.note.str.contains("no curated decision")]
    if len(new):
        print("    !! strings without any decision (add to curated dicts):", new.raw_string.tolist())

    print("[4] summary (aggregate only)")
    summarise(raw, vm, cache)


def summarise(raw: pd.DataFrame, vm: pd.DataFrame, cache: dict):
    import importlib
    import lib.cftr as cftr
    importlib.reload(cftr)
    vmap = cftr.load_variant_map(OUT_CSV, force=True)
    raw = raw.sort_values(["PatCode", "DateVisitYear"]).reset_index(drop=True)
    g = pd.DataFrame([cftr.classify_genotype(a, b, c, d, vmap=vmap) for a, b, c, d in
                      zip(raw.Mutation1, raw.OtherMutation1, raw.Mutation2, raw.OtherMutation2)])
    g["PatCode"] = raw.PatCode.values
    g["year"] = raw.DateVisitYear.values
    first = g.groupby("PatCode", sort=False).head(1)
    last = g.groupby("PatCode", sort=False).tail(1)
    n_pt = len(first)
    out = {"n_rows": int(len(raw)), "n_patients": int(n_pt)}

    def vc(df, col, order=None):
        v = df[col].value_counts()
        if order:
            v = v.reindex([o for o in order if o in v.index] + [o for o in v.index if o not in order])
        return {str(k): int(x) for k, x in v.items()}

    f_order = ["F508del homozygous", "F508del heterozygous", "no F508del", "incomplete"]
    fn_order = ["two MF alleles", "≥1 RF allele", "other or unresolved", "incomplete"]
    for lab, df in [("first_record", first), ("last_record", last)]:
        out[lab] = {
            "f508del_genotype": vc(df, "f508del_genotype", f_order),
            "f508del_detail": vc(df, "f508del_detail"),
            "functional_genotype": vc(df, "functional_genotype", fn_order),
            "functional_genotype_detail": vc(df, "functional_genotype_detail"),
            "n_eti_alleles": vc(df, "n_eti_alleles"), "eti_eligible": vc(df, "eti_eligible"),
            "n_eti_alleles_cf": vc(df, "n_eti_alleles_cf"), "eti_eligible_cf": vc(df, "eti_eligible_cf"),
            "any_non_cf_polymorphism": vc(df, "any_non_cf_polymorphism"),
            "possible_cis_pair": vc(df, "possible_cis_pair"),
            "any_low_confidence": vc(df, "any_low_confidence"),
        }
    # within-patient changes across years
    chg = g.groupby("PatCode").agg(f=("f508del_genotype", "nunique"), fn=("functional_genotype", "nunique"),
                                   e=("eti_eligible", "nunique"))
    out["patients_with_changing_category_across_years"] = {"f508del_genotype": int((chg.f > 1).sum()),
                                                           "functional_genotype": int((chg.fn > 1).sum()),
                                                           "eti_eligible": int((chg.e > 1).sum())}
    fi, la = first.set_index("PatCode"), last.set_index("PatCode")
    out["first_incomplete_but_last_complete"] = int(((fi.f508del_genotype == "incomplete") & (la.f508del_genotype != "incomplete")).sum())
    # allele-level (patient level, first record)
    al = pd.concat([first[["allele1", "slot1_status", "fg1", "PatCode"]].set_axis(["allele", "slot", "fg", "PatCode"], axis=1),
                    first[["allele2", "slot2_status", "fg2", "PatCode"]].set_axis(["allele", "slot", "fg", "PatCode"], axis=1)])
    out["allele_slots_first_record"] = {"total": int(len(al)), **{k: int(v) for k, v in al.slot.value_counts().items()}}
    ident = al[al.slot != "missing"]
    info = vm.drop_duplicates("normalised_legacy_name").set_index("normalised_legacy_name")
    top = ident.allele.value_counts().head(30)
    top_rows = []
    for name, n in top.items():
        i = info.loc[name]
        top_rows.append({"allele": name, "n_alleles": int(n), "pct_identified": round(100 * n / len(ident), 1),
                         "n_patients": int(ident[ident.allele == name].PatCode.nunique()),
                         "cftr2_status": i.cftr2_status, "function_group": i.function_group,
                         "functional_class": i.functional_class, "eti": i.eti_fda_eligible, "confidence": i.confidence})
    out["top30_alleles_first_record"] = top_rows
    unres = al[al.slot == "uncertain"]
    out["unresolved_first_record"] = {"allele_slots": int(len(unres)), "patients": int(unres.PatCode.nunique()),
                                      "by_allele": {str(k): int(v) for k, v in unres.allele.value_counts().head(40).items()}}
    out["distinct_raw_strings"] = int(len(vm))
    out["raw_strings_by_slot_status"] = {k: int(v) for k, v in vm.slot_status.value_counts().items()}
    out["raw_strings_by_function_group"] = {k: int(v) for k, v in vm.function_group.value_counts().items()}
    out["raw_strings_by_confidence"] = {k: int(v) for k, v in vm.confidence.value_counts().items()}
    out["unresolved_strings_all_rows"] = {"strings": int((vm.slot_status == "uncertain").sum()),
                                          "allele_slots": int(vm.loc[vm.slot_status == "uncertain", "n_alleles"].sum())}
    lp = vm[vm.legacy_regex_problem.fillna("") != ""]
    out["legacy_regex_problems"] = {"strings": int(len(lp)), "allele_slots_all_rows": int(lp.n_alleles.sum())}
    # problem categories and patient-level cross-tab old regex vs new (first record)
    pc = Counter()
    for r in lp.itertuples():
        for p in r.legacy_regex_problem.split("; "):
            pc[(p.split(" but now ")[0] + (" but now ..." if " but now " in p else ""))] += r.n_alleles
    out["legacy_regex_problem_categories_slots"] = dict(pc.most_common())
    f0 = raw.groupby("PatCode", sort=False).head(1)

    def old_g4(m1, o1, m2, o2):
        a1, a2 = legacy_allele_status(m1, o1), legacy_allele_status(m2, o2)
        if "UNK" in (a1, a2):
            return "unknown"
        return {2: "hom", 1: "het", 0: "non"}[(a1 == "F508") + (a2 == "F508")]

    def old_fun(m1, o1, m2, o2):
        a = [legacy_allele_class(m1, o1), legacy_allele_class(m2, o2)]
        if "residual" in a:
            return "≥1 residual-function allele"
        if "unknown" in a:
            return "incomplete"
        return "two minimal-function alleles" if all(x in ("F508del", "PTC_or_MF", "MF_missense_or_other") for x in a) else "unclassified"
    og = [old_g4(*x) for x in zip(f0.Mutation1, f0.OtherMutation1, f0.Mutation2, f0.OtherMutation2)]
    of = [old_fun(*x) for x in zip(f0.Mutation1, f0.OtherMutation1, f0.Mutation2, f0.OtherMutation2)]
    ct1 = pd.crosstab(pd.Series(og, name="old genotype4"), first.f508del_genotype.values)
    ct2 = pd.crosstab(pd.Series(of, name="old genotype_functional"), first.functional_genotype.values)
    out["crosstab_old_vs_new_f508del_first_record"] = {i: {c: int(ct1.loc[i, c]) for c in ct1.columns} for i in ct1.index}
    out["crosstab_old_vs_new_functional_first_record"] = {i: {c: int(ct2.loc[i, c]) for c in ct2.columns} for i in ct2.index}
    out["sources"] = {k: {kk: cache[k].get(kk) for kk in ("url", "version", "revised", "reference_id", "accessed", "sha256")
                          if kk in cache[k]} for k in ("cftr2", "fda_eti", "fda_iva", "fda_tezi")}
    out["sources"]["refseq"] = {k: cache["refseq"][k] for k in ("accession", "accessed", "biopython")}
    SUMMARY_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ---- markdown tables
    L = [f"# CFTR genotype summary (auto-generated by scripts/cftr_variant_map.py, {dt.date.today()})", "",
         f"Rows {out['n_rows']}, patients {n_pt}. Patient-level = first record per PatCode (earliest DateVisitYear). "
         "Aggregate counts only.", ""]

    def tbl(title, d_first, d_last):
        L.extend([f"### {title}", "", "| Category | First record n (%) | Last record n (%) |", "|---|---|---|"])
        for k in list(dict.fromkeys(list(d_first) + list(d_last))):
            a, b = d_first.get(k, 0), d_last.get(k, 0)
            L.append(f"| {k} | {a} ({100*a/n_pt:.1f}) | {b} ({100*b/n_pt:.1f}) |")
        L.append("")
    F, La = out["first_record"], out["last_record"]
    tbl("(a) F508del genotype", F["f508del_genotype"], La["f508del_genotype"])
    tbl("(a') F508del detail", F["f508del_detail"], La["f508del_detail"])
    tbl("(b) Functional genotype", F["functional_genotype"], La["functional_genotype"])
    tbl("(b') Functional genotype, detail", F["functional_genotype_detail"], La["functional_genotype_detail"])
    tbl("(c) ETI-eligible alleles (label-literal, FDA Table 6 Dec 2024)", F["n_eti_alleles"], La["n_eti_alleles"])
    tbl("(c) ETI eligible (≥1 label-listed allele)", F["eti_eligible"], La["eti_eligible"])
    tbl("(c') ETI-eligible alleles counting only CF-causing/VCC/F508del slots", F["n_eti_alleles_cf"], La["n_eti_alleles_cf"])
    tbl("(c') ETI eligible, CF-causing/VCC alleles only", F["eti_eligible_cf"], La["eti_eligible_cf"])
    tbl("(d) Any non-CF-causing polymorphism in a slot", F["any_non_cf_polymorphism"], La["any_non_cf_polymorphism"])
    tbl("Possible cis pair split across the two slots", F["possible_cis_pair"], La["possible_cis_pair"])
    L.extend(["### Top 30 alleles (patient level, first record; % of identified allele slots)", "",
              "| # | Allele | n alleles | % | n patients | CFTR2 | Function group | Class | ETI (FDA) | Conf. |",
              "|---|---|---|---|---|---|---|---|---|---|"])
    for i, t in enumerate(top_rows, 1):
        L.append(f"| {i} | {t['allele']} | {t['n_alleles']} | {t['pct_identified']} | {t['n_patients']} | {t['cftr2_status']} | "
                 f"{t['function_group']} | {t['functional_class']} | {t['eti']} | {t['confidence']} |")
    a = out["allele_slots_first_record"]
    L.extend(["", f"Allele slots at first record: {a}", "",
              f"Unresolved (identity or pathogenicity uncertain) at first record: {out['unresolved_first_record']['allele_slots']} "
              f"slots in {out['unresolved_first_record']['patients']} patients; all rows: "
              f"{out['unresolved_strings_all_rows']['strings']} distinct strings / {out['unresolved_strings_all_rows']['allele_slots']} slots.", "",
              "### Previous regex approach vs new classification (patient level, first record)", ""])
    for key, title in [("crosstab_old_vs_new_f508del_first_record", "old genotype4 (rows) x new F508del genotype (columns)"),
                       ("crosstab_old_vs_new_functional_first_record", "old genotype_functional (rows) x new functional genotype (columns)")]:
        ct = out[key]
        colsx = sorted({c for v in ct.values() for c in v})
        L.extend([f"{title}:", "", "| old \\ new | " + " | ".join(colsx) + " |", "|---" * (len(colsx) + 1) + "|"])
        for i, v in ct.items():
            L.append(f"| {i} | " + " | ".join(str(v.get(c, 0)) for c in colsx) + " |")
        L.append("")
    L.extend(["Problem categories (allele slots, all rows): " +
              "; ".join(f"{k}: {v}" for k, v in out["legacy_regex_problem_categories_slots"].items()), "",
              "### Raw strings the previous regex approach (00_build_cohort.py) mis-handles", "",
              "| Raw string | n slots (all rows) | n patients | Regex status/class | New classification | Problem |",
              "|---|---|---|---|---|---|"])
    for r in lp.sort_values("n_alleles", ascending=False).itertuples():
        L.append(f"| `{r.raw_string}` | {r.n_alleles} | {r.n_patients} | {r.legacy_regex_status}/{r.legacy_regex_class} | "
                 f"{r.normalised_legacy_name}: {r.slot_status}, {r.function_group} | {r.legacy_regex_problem} |")
    L.extend(["", f"Patients whose category changes across registry years: {out['patients_with_changing_category_across_years']}; "
              f"incomplete at first record but complete at last: {out['first_incomplete_but_last_complete']}."])
    SUMMARY_MD.write_text("\n".join(L) + "\n")
    print(f"    wrote {SUMMARY_MD.relative_to(REPO)} and {SUMMARY_JSON.name}")
    print("    first record:", json.dumps(F["f508del_genotype"], ensure_ascii=False), json.dumps(F["functional_genotype"], ensure_ascii=False),
          json.dumps(F["eti_eligible"]))


if __name__ == "__main__":
    main()

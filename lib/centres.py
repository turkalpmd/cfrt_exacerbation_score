"""Harmonise registry centre identifiers (codes changed format between years; 2019 rows lack centre names).
Row name := CentreDescrip if it is a real name, else the name learned for that CentreCode in other rows.
Institution := name without department suffix (paediatric/adult units of one hospital = one institution).
Patient centre := modal institution over the patient's rows (ties -> most recent)."""
import re

import pandas as pd

DEPT = re.compile(r"\s+(Erişkin|Çocuk|Cocuk|Pediatri|Göğüs)\b.*$", re.I)


def _is_name(s: str) -> bool:
    return bool(s) and not s.isdigit() and not s.upper().startswith("KF-") and s.lower() != "nan"


def patient_institution(raw: pd.DataFrame) -> pd.Series:
    r = raw[["PatCode", "DateVisitYear", "CentreCode", "CentreDescrip"]].copy()
    r["code"] = r.CentreCode.astype(str).str.strip(); r["desc"] = r.CentreDescrip.astype(str).str.strip()
    named = r[r.desc.map(_is_name)]
    code2name = named.groupby("code").desc.agg(lambda s: s.mode().iloc[0]).to_dict()
    r["name"] = [d if _is_name(d) else code2name.get(c, code2name.get(d)) for c, d in zip(r.code, r.desc)]
    r["inst"] = r.name.map(lambda n: DEPT.sub("", n).strip() if isinstance(n, str) else None)
    r = r.dropna(subset=["inst"]).sort_values("DateVisitYear")
    return r.groupby("PatCode").inst.agg(lambda s: s.value_counts().sort_index(kind="stable").idxmax() if s.value_counts().max() > 1 else s.iloc[-1])


def patient_centre_unit(raw: pd.DataFrame) -> pd.Series:
    """Registry centre at unit level (paediatric and adult departments of one hospital kept separate, as registered in the
    CFRT); patient centre := modal unit over the patient's rows (ties -> most recent)."""
    r = raw[["PatCode", "DateVisitYear", "CentreCode", "CentreDescrip"]].copy()
    r["code"] = r.CentreCode.astype(str).str.strip(); r["desc"] = r.CentreDescrip.astype(str).str.strip()
    named = r[r.desc.map(_is_name)]
    code2name = named.groupby("code").desc.agg(lambda s: s.mode().iloc[0]).to_dict()
    r["unit"] = [d if _is_name(d) else code2name.get(c, code2name.get(d)) for c, d in zip(r.code, r.desc)]
    r = r.dropna(subset=["unit"]).sort_values("DateVisitYear")
    return r.groupby("PatCode").unit.agg(lambda s: s.value_counts().sort_index(kind="stable").idxmax() if s.value_counts().max() > 1 else s.iloc[-1])

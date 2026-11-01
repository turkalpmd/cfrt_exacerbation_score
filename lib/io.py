"""JSON/CSV writers with provenance."""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import numpy as np


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=Path(__file__).parent).strip()
    except Exception:
        return "unknown"


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def write_json(obj: dict, path: Path, script: str):
    obj = {"_provenance": {"script": script, "git": _git_hash(),
                           "time": _dt.datetime.now().isoformat(timespec="seconds")}, **obj}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, cls=_Enc))
    return path

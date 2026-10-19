"""Readers for the repository's markdown result tables and a few JSONs; every figure panel cites the file it reads (see manifest)."""
import re, json, os
from style import ROOT
def md_table(path):
    rows = [l.strip().replace("\\|", "¦") for l in open(f"{ROOT}/{path}") if l.strip().startswith("|")]; head = [h.strip() for h in rows[0].strip("|").split("|")]; out = []
    for l in rows[2:]:
        cells = [c.strip() for c in l.strip("|").split("|")]
        if len(cells) == len(head): out.append(dict(zip(head, cells)))
    return out
def num(s):
    """first number in a cell ('0.095 (0.12)' -> 0.095, '+21' -> 21, '2.4x' -> 2.4, '-' -> None)"""
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s.replace("−", "-")); return float(m.group()) if m else None
def nums(s): return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", s.replace("−", "-"))]

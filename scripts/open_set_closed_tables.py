#!/usr/bin/env python3
"""Closed-set control vs open set, configuration 1 (scripts/open_set_closed.py). Per seed and type t, image AUROC on the
SAME test images: 50 test normals + the real anomalies of type t that are NOT budget references of that seed.
  closed  one run per seed and arm, trained on all 6 types (closed_c1/<arm>_s1/seed_<s>/image_scores.json)
  open    fold t of the configuration-1 score dump (score_dist_c1/<arm>_s1/seed_<s>/fold_<t>/image_scores.json), same anomalies
Epoch rules: last epoch; best-0x per type (each arm picks its own epoch on that type's 0x, as the open-set tables do);
mean of epochs 15-20. Differences are paired by seed (same canvases, budget, placements, normals, training seed).
Without closed runs the open side alone is printed (preview).

  python scripts/open_set_closed_tables.py [--open DIR] [--closed DIR] [--manifests results/open_set_v2/manifest]
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "results/open_set_v2"
T = ["holes", "breakage", "burnt", "colour_diff", "colour_same", "scratches"]
LV = ["0x", "1x", "2x", "3x", "4x"]
ARMS = {"diffusion_in": "Target-object", "diffusion_cross": "Cross-object"}


def budgets(src: Path) -> dict[int, set[str]]:
    """Budget references per seed from the manifests: a folder of seed_<s>.json or a tar archive of manifest/seed_<s>.json."""
    out = {}
    if src.is_dir():
        for f in sorted(src.glob("seed_*.json")):
            man = json.loads(f.read_text(encoding="utf-8")); out[int(man["seed"])] = {r for t in T for r in man["budget"][t]}
        return out
    with tarfile.open(src) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".json") and "/manifest/seed_" in m.name:
                man = json.load(tf.extractfile(m)); out[int(man["seed"])] = {r for t in T for r in man["budget"][t]}
    return out


def load(f: Path) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """files ('A<id>' for anomalies), labels, scores[level] = [epoch, image] for the run's primary aggregation."""
    res = json.load(open(f.parent / "results.json", encoding="utf-8"))
    prim = [v.get("primary_agg") for k, v in res.items() if "@" not in k and isinstance(v, dict)][0] or "default"
    S: dict[str, dict[int, np.ndarray]] = {lv: {} for lv in LV}; files = labels = None
    for e in json.load(open(f)):
        if e["agg"] != prim or e["shift"] not in LV or e["epoch"] is None:
            continue
        S[e["shift"]][e["epoch"]] = np.array(e["scores"], float)
        fl = [("A" + Path(x).stem) if "Anomaly" in x else Path(x).name for x in e["files"]]
        if files is None:
            files, labels = fl, np.array(e["labels"])
        assert fl == files and list(e["labels"]) == list(labels), f"file order differs in {f}"
    assert all((l == 1) == x.startswith("A") for l, x in zip(labels, files))
    return files, labels, {lv: np.stack([S[lv][ep] for ep in sorted(S[lv])]) for lv in LV}


def curves(run, keep: set[str]) -> np.ndarray:
    """[level, epoch] AUROC on the normals + the anomalies in keep."""
    files, y, S = run; m = np.array([l == 0 or x in keep for x, l in zip(files, y)])
    assert (y[m] == 1).sum() == len(keep), (len(keep), (y[m] == 1).sum())
    return np.array([[roc_auc_score(y[m], S[lv][e][m]) for e in range(S[lv].shape[0])] for lv in LV])


def rules(A: np.ndarray) -> dict[str, tuple[float, float]]:
    e = int(np.argmax(A[0]))
    return {"last": (A[0, -1], A[1:, -1].mean()), "best-0×": (A[0, e], A[1:, e].mean()),
            "mean ep 15–20": (A[0, 14:20].mean(), A[1:, 14:20].mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", type=Path, default=B / "score_dist_c1")
    ap.add_argument("--closed", type=Path, default=B / "closed_c1")
    ap.add_argument("--manifests", type=Path, default=B / "manifest")
    a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); bud = budgets(a.manifests)
    for arm, nm in ARMS.items():
        seeds = sorted(int(d.name[5:]) for d in (a.open / f"{arm}_s1").glob("seed_*"))
        R = {"open": {}, "closed": {}}                                  # [seed][type] -> rules
        for s in seeds:
            cr = a.closed / f"{arm}_s1" / f"seed_{s}" / "image_scores.json"
            closed = load(cr) if cr.exists() else None
            for t in T:
                op = load(a.open / f"{arm}_s1" / f"seed_{s}" / f"fold_{t}" / "image_scores.json")
                keep = {x for x, l in zip(op[0], op[1]) if l == 1 and x[1:] not in bud[s]}
                R["open"].setdefault(s, {})[t] = rules(curves(op, keep))
                if closed is not None:
                    R["closed"].setdefault(s, {})[t] = rules(curves(closed, keep))
        both = sorted(R["closed"])
        print(f"\n## {nm}: open seeds {len(seeds)}, closed seeds {len(both)} {both}\n")
        for rule in ("last", "best-0×", "mean ep 15–20"):
            print(f"### {rule} (0× / 1–4×; mean over seeds)\n")
            print("| | " + " | ".join(T) + " | macro 0× | macro 1–4× | 2 hard 0× | 2 hard 1–4× |"); print("|---|" + "---|" * 10)
            sides = [("open (type withheld)", R["open"], seeds)] + ([("closed (type in training)", R["closed"], both)] if both else [])
            for lab, D, ss in sides:
                A = np.array([[D[s][t][rule] for t in T] for s in ss])           # [seed, type, (0x, 1-4x)]
                print(f"| {lab} | " + " | ".join(f"{A[:, j, 0].mean():.3f} / {A[:, j, 1].mean():.3f}" for j in range(6))
                      + f" | {A[..., 0].mean():.3f} | {A[..., 1].mean():.3f} | {A[:, 4:, 0].mean():.3f} | {A[:, 4:, 1].mean():.3f} |")
            if both:
                O = np.array([[R["open"][s][t][rule] for t in T] for s in both]); C = np.array([[R["closed"][s][t][rule] for t in T] for s in both])
                D = C - O
                def st(x): return f"{x.mean():+.3f} ± {x.std(ddof=1) / np.sqrt(len(x)):.3f} ({(x > 0).sum()}/{len(x)})" if len(x) > 1 else f"{x.mean():+.3f}"
                print(f"| **closed − open, paired** (± SE, seeds closed better) | " + " | ".join(f"{D[:, j, 0].mean():+.3f} / {D[:, j, 1].mean():+.3f}" for j in range(6))
                      + f" | {st(D[..., 0].mean(1))} | {st(D[..., 1].mean(1))} | {st(D[:, 4:, 0].mean(1))} | {st(D[:, 4:, 1].mean(1))} |")
            print()


if __name__ == "__main__":
    main()

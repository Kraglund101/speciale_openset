#!/usr/bin/env python3
"""Result tables of the open-set experiment (results/open_set_v2/runs/<arm>/seed_s/fold_k/results.json).
Image AUROC; primary = composite (mean of 0x-4x), mean of epochs 15-20; also clean (0x) epochs 15-20 and last epoch.
Per arm: mean over all finished (seed, fold) runs, and the paired difference to normal-only (m1) on the same seed and
fold. Per fold: the primary metric of every arm.

  python scripts/open_set_tables.py     -> results/open_set_v2/TABLES.md
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import open_set_steps as S  # noqa: E402

LEVELS = ["0x", "1x", "2x", "3x", "4x"]
LABEL = {"m1": "normal-only (M1)", "diffusion_in": "target-object", "diffusion_cross": "cross-object", "dtd": "DTD", "cutmix": "CutMix"}


def metrics(f: Path) -> dict | None:
    if not f.exists():
        return None
    r = json.load(open(f, encoding="utf-8")); k0 = [k for k in r if k.startswith("Mode") and " @ " not in k]
    if not k0 or any(f"{k0[0]} @ {lv}" not in r for lv in LEVELS):
        return None
    a = np.array([r[f"{k0[0]} @ {lv}"]["img_aurocs"] for lv in LEVELS]); n = a.shape[1]
    if n < 2:
        return None
    late = slice(max(n - 6, 0), n); c = a.mean(0)
    return {"comp_late": float(c[late].mean()), "clean_late": float(a[0, late].mean()), "comp_last": float(c[-1]), "clean_last": float(a[0, -1]),
            "blur_late": float(a[3:, late].mean()), "epochs": n}


def main() -> None:
    runs = S.OUT / "runs"; seeds = sorted({int(p.name[5:]) for p in runs.glob("*/seed_*")}) if runs.exists() else []
    M = {S.run_name(m, s_): {(sd, k): metrics(runs / S.run_name(m, s_) / f"seed_{sd}" / f"fold_{k}" / "results.json") for sd in seeds for k in S.CLASSES}
         for m, s_ in S.ARMS}
    M = {a: {k: v for k, v in d.items() if v} for a, d in M.items()}
    out = ["# Open-set results (image AUROC)", "",
           f"Seeds found: {seeds}. Late = mean of the last 6 epochs (epochs 15-20 of a 20-epoch run). Composite = mean of 0x-4x. "
           "Difference to M1 is paired on seed and fold.", "",
           "| arm | stage | runs | composite, late | clean (0x), late | composite, last | strong blur (3x-4x), late | composite vs M1 (paired) | runs better than M1 |",
           "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: f"{st.mean(v):.3f}" + (f" ± {st.pstdev(v):.3f}" if len(v) > 1 else "")
    for m, s_ in S.ARMS:
        d = M[S.run_name(m, s_)]
        if not d:
            continue
        keys = sorted(d); pair = [(d[k]["comp_late"] - M["m1"][k]["comp_late"]) for k in keys if k in M["m1"]]
        out.append(f"| {LABEL[m]} | {s_ or '–'} | {len(d)} | {f([d[k]['comp_late'] for k in keys])} | {f([d[k]['clean_late'] for k in keys])} | "
                   f"{f([d[k]['comp_last'] for k in keys])} | {f([d[k]['blur_late'] for k in keys])} | "
                   + ("–" if m == "m1" or not pair else f"{st.mean(pair):+.3f}") + " | "
                   + ("–" if m == "m1" or not pair else f"{sum(x > 0 for x in pair)} of {len(pair)}") + " |")
    out += ["", "## Composite, late, per held-out type (mean over seeds)", "", "| arm | stage | " + " | ".join(S.CLASSES) + " |", "|---|---|" + "---|" * len(S.CLASSES)]
    for m, s_ in S.ARMS:
        d = M[S.run_name(m, s_)]
        if d:
            cells = [[d[(sd, k)]["comp_late"] for sd in seeds if (sd, k) in d] for k in S.CLASSES]
            out.append(f"| {LABEL[m]} | {s_ or '–'} | " + " | ".join(f"{st.mean(c):.3f}" if c else "–" for c in cells) + " |")
    (S.OUT / "TABLES.md").write_text("\n".join(out) + "\n", encoding="utf-8"); print("\n".join(out))


if __name__ == "__main__":
    main()

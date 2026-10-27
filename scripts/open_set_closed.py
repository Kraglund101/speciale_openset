#!/usr/bin/env python3
"""Closed-set control for configuration 1 (user 2026-10-23): does seeing the evaluated anomaly type in training help?

Per seed ONE run per arm (diffusion_in = Target-object, diffusion_cross = Cross-object) on ALL 6 types: the 54
configuration-1 synthetics (stage 1 = s2 epoch 0: the 3 budget anomalies of each type on the type's own 9 canvases,
the same set every epoch), i.e. exactly fold k's 45 plus type k's 9 - for every k at once. Everything else is the
launcher's command (open_set_steps.command: hero, seed, train normals, flags); only the lists differ:
  synthetics  54 per epoch (s2 epoch 0 of every type; images already exist - NO generation, NO manifest rebuild)
  test        50 test normals + every real anomaly NOT among the seed's 18 budget references (66 images)
Runner = scripts/score_runner.py (per-image scores), so per-type AUROC is computed afterwards against the same 50 normals.
The matched open-set side is the existing score_dist_c1 dumps of fold k restricted to the same non-budget anomalies.
Output: results/open_set_v2/closed_c1/<arm>_s1/seed_<s>/{results.json,image_scores.json,stdout.log}; checkpoints deleted.

  python scripts/open_set_closed.py check --seeds 42 123 7 99 256 11 22 33 44 55     # lists + 0 missing files, no training
  python scripts/open_set_closed.py run   --seeds 42 123 7 99 256 11 22 33 44 55 --slots 12 --stagger 10
  python scripts/open_set_closed.py run   --seeds 42 --methods diffusion_in --epochs 2 --sub smoke --slots 1 --stagger 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import open_set_scores as SC  # noqa: E402
import open_set_steps as S  # noqa: E402

METHODS = ["diffusion_in", "diffusion_cross"]


def out_dir(sub: str, m: str, s: int) -> Path:
    return S.OUT / "closed_c1" / ("" if sub == "runs" else sub) / S.run_name(m, 1) / f"seed_{s}"


def closed_lists(s: int, method: str) -> dict:
    """Run lists of the closed control for seed s (written to lists/seed_<s>/closed/)."""
    man = S.manifest(s); d = S.OUT / "lists" / f"seed_{s}" / "closed"; d.mkdir(parents=True, exist_ok=True)
    sp = json.loads(S.P.SPLITS.read_text())["normals"]; tn = S.train_normals(s, man)
    pool = sorted(f"{c}.JPG" for t in S.CLASSES for c in man["groups"][t])
    assert len(pool) == 54 and not (set(tn) | set(pool)) & set(sp["test"]) and not set(tn) & set(pool)
    budget = {r for t in S.CLASSES for r in man["budget"][t]}; refs = S.load_refs()
    assert len(budget) == len(S.CLASSES) * S.BUDGET
    test = [{"id": r["id"], "raw": r["raw"], "image": r["ind_img"], "gt_mask": r["gt_mask"]}
            for t in S.CLASSES for r in refs[t] if r["id"] not in budget]
    assert len(test) == sum(len(refs[t]) for t in S.CLASSES) - len(budget)
    ep, canv = [], []
    for t in S.CLASSES:                                               # stage 1 = s2 epoch 0, shared by all folds
        for st in man["sets"]["s2"][t][:S.K]:
            assert "alias" not in st and st["epoch"] == 0 and st["ref"] in man["budget"][t], (t, st)
            tag, c, _ = S.resolve(man, "s2", t, t, st); assert tag == "s2", (t, st, tag)
            i = st["i"]
            ep.append({"image": str(S.ipath(method, s, tag, t, i)), "mask": str(S.ipath(method, s, tag, t, i, "masks")),
                       "canvas": str(S.pdir(s, tag, t, i) / "canvas.png")}); canv.append(f"{c}.JPG")
    assert sorted(canv) == pool, f"seed {s}: closed-set canvases are not exactly the seed's 54"
    for k in S.CLASSES:                                               # = fold k's configuration-1 list + type k's 9
        fk = S.fold_epochs(man, s, k, method, 1)[0]
        assert {x["image"] for x in fk} == {x["image"] for x, t in zip(ep, [t for t in S.CLASSES for _ in range(S.K)]) if t != k}, (s, k)
    epochs = [ep] * S.EPOCH_LIMIT
    (d / "train_normals.txt").write_text("\n".join(tn) + "\n"); (d / "m1_extra.txt").write_text("\n".join(pool) + "\n")
    json.dump(test, open(d / "test_anomalies.json", "w"), indent=1)
    f = d / f"epochs_{method}_s1.json"; json.dump({"epochs": epochs}, open(f, "w"))
    miss = sum(not Path(x[k]).exists() for x in ep for k in ("image", "mask", "canvas"))
    return {"dir": d, "train_normals": d / "train_normals.txt", "m1_extra": d / "m1_extra.txt", "test": d / "test_anomalies.json",
            "n_test": len(test), "epochs": f, "missing": miss}


def command(m: str, s: int, sub: str, extra: list[str]) -> list[str]:
    orig, S.run_lists = S.run_lists, lambda s_, fold, method, step: closed_lists(s_, method)
    try:
        c = S.command(s, S.CLASSES[0], m, 1, extra)                   # fold argument unused by the closed lists
    finally:
        S.run_lists = orig
    assert c[2].endswith("train_uninet.py"), c[:3]
    c[c.index("--output-dir") + 1] = str(out_dir(sub, m, s))
    return [sys.executable, "-u", "-c", SC.BOOT.format(s=str(ROOT / "scripts"), r=str(ROOT / "scripts/score_runner.py"))] + c[3:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "run"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42]); ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--slots", type=int, default=2); ap.add_argument("--stagger", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=S.E); ap.add_argument("--sub", default="runs")
    a = ap.parse_args(); extra = [] if a.epochs == S.E else ["--epochs", str(a.epochs)]
    if a.stage == "check":
        bad = 0
        for s in a.seeds:
            for m in a.methods:
                r = closed_lists(s, m); bad += r["missing"] > 0
                print(f"seed {s} {m}: 54 synthetics per epoch (= each fold's 45 + its type's 9), {r['n_test']} test anomalies, "
                      f"missing files {r['missing']}")
        print(f"check: {bad} problem(s)"); return
    gate = threading.Lock(); env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def one(job: tuple) -> None:
        m, s = job; d = out_dir(a.sub, m, s)
        if SC.done(d, a.epochs):
            return
        with gate:
            cmd = command(m, s, a.sub, extra); time.sleep(a.stagger)
        d.mkdir(parents=True, exist_ok=True); t0 = time.time()
        with open(d / "stdout.log", "w", encoding="utf-8") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        shutil.rmtree(d / "checkpoints", ignore_errors=True)
        print(f"{time.strftime('%m-%d %H:%M')}  [closed_c1] {S.run_name(m, 1)} seed {s}: rc {rc}, "
              f"{(time.time() - t0) / 60:.1f} min, scores {'ok' if SC.done(d, a.epochs) else 'MISSING'}", flush=True)

    jobs = [(m, s) for s in a.seeds for m in a.methods]
    print(f"[closed_c1] {len(jobs)} runs, {a.slots} at a time", flush=True)
    with ThreadPoolExecutor(a.slots) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()

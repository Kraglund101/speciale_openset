#!/usr/bin/env python3
"""Per-image anomaly scores of the configuration-1 open-set runs (score histograms normals vs anomalies, 0x and 1-4x).

Re-trains Normal-only + CutMix + DTD + Target-object + Cross-object, configuration 1, for the given seeds x folds with
EXACTLY the launcher's command (open_set_steps.command: same lists, hero, seed, flags), only the runner is
scripts/score_runner.py, which also writes image_scores.json. Output: results/open_set_v2/score_dist_c1/<arm>/seed_<s>/
fold_<k>/{results.json,image_scores.json,stdout.log}; checkpoints are deleted.

  python scripts/open_set_scores.py --seeds 42 --slots 12 --stagger 10
  python scripts/open_set_scores.py --seeds 42 --folds holes --methods dtd --epochs 2 --sub smoke    # smoke test
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
import open_set_steps as S  # noqa: E402

METHODS = ["m1", "cutmix", "dtd", "diffusion_in", "diffusion_cross"]
BOOT = ("import sys; sys.path.insert(0, {s!r}); from pathlib import Path; import train_uninet as T; "
        "T.RUNNER = Path({r!r}); sys.argv = ['train_uninet.py'] + sys.argv[1:]; raise SystemExit(T.main())")


def out_dir(sub: str, m: str, s: int, k: str) -> Path:
    return S.OUT / "score_dist_c1" / ("" if sub == "runs" else sub) / S.run_name(m, 1) / f"seed_{s}" / f"fold_{k}"


def run_lists_ro(s: int, fold: str, method: str, step: int) -> dict:
    """Server 2026-10-15: S.run_lists without writing. Other jobs read the shared lists concurrently, so the lists on disk
    are only asserted to equal what run_lists would write (same code path), and their paths are returned."""
    man = S.manifest(s); d = S.OUT / "lists" / f"seed_{s}" / f"fold_{fold}"; tn = S.train_normals(s, man)
    Pk = sorted(f"{c}.JPG" for t in S.CLASSES if t != fold for c in man["groups"][t])
    test = [{"id": r["id"], "raw": r["raw"], "image": r["ind_img"], "gt_mask": r["gt_mask"]} for r in S.load_refs()[fold]]
    assert (d / "train_normals.txt").read_text() == "\n".join(tn) + "\n", (s, fold, "train_normals differ")
    assert (d / "m1_extra.txt").read_text() == "\n".join(Pk) + "\n", (s, fold, "m1_extra differ")
    assert json.load(open(d / "test_anomalies.json")) == test, (s, fold, "test list differs")
    out = {"dir": d, "train_normals": d / "train_normals.txt", "m1_extra": d / "m1_extra.txt", "test": d / "test_anomalies.json", "n_test": len(test)}
    if method != "m1":
        ep = json.loads(json.dumps(S.fold_epochs(man, s, fold, method, step))); out["epochs"] = d / f"epochs_{method}_s{step}.json"
        assert json.load(open(out["epochs"]))["epochs"] == ep, (s, fold, method, "epoch list differs")
        out["missing"] = sum(not Path(t[k]).exists() for e in ep for t in e for k in ("image", "mask", "canvas"))
    return out


def command(m: str, s: int, k: str, sub: str, extra: list[str]) -> list[str]:
    orig, S.run_lists = S.run_lists, run_lists_ro      # same command as the launcher, but no list rewrite
    try:
        c = S.command(s, k, m, 1, extra)
    finally:
        S.run_lists = orig
    assert c[2].endswith("train_uninet.py"), c[:3]
    c[c.index("--output-dir") + 1] = str(out_dir(sub, m, s, k))
    return [sys.executable, "-u", "-c", BOOT.format(s=str(ROOT / "scripts"), r=str(ROOT / "scripts/score_runner.py"))] + c[3:]


def done(d: Path, n: int) -> bool:
    try:
        r = json.load(open(d / "results.json", encoding="utf-8")); k = [x for x in r if x.startswith("Mode") and " @ " not in x]
        return bool(k) and len(r[k[0]]["img_aurocs"]) >= n and (d / "image_scores.json").exists()
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42]); ap.add_argument("--folds", nargs="+", default=S.CLASSES)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--slots", type=int, default=2); ap.add_argument("--stagger", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=S.E); ap.add_argument("--sub", default="runs")
    a = ap.parse_args(); extra = [] if a.epochs == S.E else ["--epochs", str(a.epochs)]
    gate = threading.Lock(); env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def one(job: tuple) -> None:
        m, s, k = job; d = out_dir(a.sub, m, s, k)
        if done(d, a.epochs):
            return
        with gate:
            cmd = command(m, s, k, a.sub, extra); time.sleep(a.stagger)
        d.mkdir(parents=True, exist_ok=True); t0 = time.time()
        with open(d / "stdout.log", "w", encoding="utf-8") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        shutil.rmtree(d / "checkpoints", ignore_errors=True)
        print(f"{time.strftime('%m-%d %H:%M')}  [scores_c1] {S.run_name(m, 1)} seed {s} fold {k}: rc {rc}, "
              f"{(time.time() - t0) / 60:.1f} min, scores {'ok' if done(d, a.epochs) else 'MISSING'}", flush=True)

    jobs = [(m, s, k) for s in a.seeds for m in a.methods for k in a.folds]
    print(f"[scores_c1] {len(jobs)} runs, {a.slots} at a time", flush=True)
    with ThreadPoolExecutor(a.slots) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()

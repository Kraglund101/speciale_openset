#!/usr/bin/env python3
"""One launcher for the open-set experiment, for a workstation and for a multi-GPU server (same code, no shell scripts):
plan -> generate -> refine -> dtd -> cutmix -> check -> train -> tables, each step resumable (existing files and
finished runs are skipped).

  # pilot on a single GPU: seed 42, two folds, every arm
  python scripts/open_set_launch.py --profile local --seeds 42 --folds holes breakage

  # smoke test (any machine): 2 epochs of everything, one seed, one fold - checks the whole chain in minutes
  python scripts/open_set_launch.py --profile server --smoke

  # full run
  python scripts/open_set_launch.py --profile server --seeds 42 123 7 99 256 11 22 33 44 55

Several GPUs: --gpus N spreads the generation processes and the trainings over GPU 0..N-1 (CUDA_VISIBLE_DEVICES per
process); the profile's counts are then per GPU (server: 8 generation processes and 8 trainings per GPU). Untested on
more than one GPU (2026-10-03). A GPU split into slices (MIG): a process can use only one slice, so pass the slice
UUIDs from `nvidia-smi -L` with --gpu-ids MIG-... MIG-... and set --gen-procs / --train-slots to what the slices hold.

Profiles only set defaults: --gen-procs (parallel generation processes), --train-slots (trainings at once), --stagger
(seconds between training starts; simultaneous start-ups stall each other). Checkpoints are deleted after every run
unless --keep-checkpoints (a run writes ~1.1 GB). --stages limits what is done, e.g. --stages train tables.
Progress: results/open_set_v2/launch.log; one log per step / run under results/open_set_v2/logs/.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/open_set_v2"
STEPS = str(ROOT / "scripts/open_set_steps.py")
PROFILES = {"local": dict(gen_procs=2, train_slots=2, stagger=90), "server": dict(gen_procs=8, train_slots=8, stagger=45)}
ALL = ["plan", "generate", "refine", "dtd", "cutmix", "check", "train", "tables"]
_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M')}  {msg}"
    with _lock:
        print(line, flush=True); OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "launch.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(cmd: list[str], logfile: Path, env: dict) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a", encoding="utf-8") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))


def done(results: Path, epochs: int) -> bool:
    if not results.exists():
        return False
    try:
        r = json.load(open(results, encoding="utf-8")); k = [k for k in r if k.startswith("Mode") and " @ " not in k]
        return bool(k) and len(r[k[0]].get("img_aurocs", [])) >= epochs
    except Exception:
        return False


def detect_gpus() -> tuple[list[str], bool]:
    """(device ids, is_mig) from `nvidia-smi -L`. MIG slices -> their MIG-... UUIDs (a process can use only one slice);
    several whole GPUs -> their indices; one GPU or no nvidia-smi -> ["0"]."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ["0"], False
    mig = re.findall(r"UUID:\s*(MIG-[0-9A-Za-z/-]+)", out)
    if mig:
        return mig, True
    return [str(i) for i in range(max(1, len(re.findall(r"^GPU \d+:", out, flags=re.M))))], False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="local")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42]); ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=None, help="e.g. m1 diffusion_in_s1 dtd_s3 (default: all 16)")
    ap.add_argument("--stages", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--gen-procs", type=int); ap.add_argument("--train-slots", type=int); ap.add_argument("--stagger", type=int)
    ap.add_argument("--smoke", action="store_true", help="2 epochs, seed 42, fold holes: end-to-end check")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--hero", choices=["caao", "mode3"], default="caao",
                    help="UniNet setting for the synthetic arms; mode3 -> runs_mode3/, m1 is not re-run, no tables stage")
    ap.add_argument("--train-epochs", type=int, default=20,
                    help="epochs per training (default 20); other values write to runs[_<hero>]_e<n>/ and skip the tables stage")
    ap.add_argument("--runs-tag", default="", help="runs go to runs[_<hero>][_e<n>]_<tag>/ (e.g. profile); logs *_<tag>_*; no tables stage")
    ap.add_argument("--cpu-procs", type=int, default=12, help="parallel processes for the CPU stages dtd / cutmix (outputs identical)")
    ap.add_argument("--sets", default=None, help="only these image sets, e.g. s2 (stages 1-2) or s2,s3")
    ap.add_argument("--gpus", type=int, default=0, help="number of GPUs (default 0 = detect with nvidia-smi: slices of a split GPU "
                                                        "and several GPUs are used like multi-GPU, one GPU runs as before); processes are spread over them with CUDA_VISIBLE_DEVICES "
                                                        "(the profile's process counts are per GPU)")
    ap.add_argument("--gpu-ids", nargs="+", default=None, help="explicit device ids instead of 0..N-1, e.g. the MIG-... UUIDs "
                                                               "that `nvidia-smi -L` lists when a GPU is split into slices")
    a = ap.parse_args(); prof = PROFILES[a.profile]
    found, mig = detect_gpus() if not (a.gpu_ids or a.gpus) else ([], False)
    ids = a.gpu_ids or ([str(i) for i in range(a.gpus)] if a.gpus else found); a.gpus = len(ids)
    pin = a.gpus > 1 or bool(a.gpu_ids) or mig                 # one whole GPU: environment untouched
    per = 2 if mig else None                                   # a slice is small: 2 processes per slice unless told otherwise
    gen_procs = a.gen_procs or (per or prof["gen_procs"]) * a.gpus; slots = a.train_slots or (per or prof["train_slots"]) * a.gpus
    stagger = prof["stagger"] if a.stagger is None else a.stagger
    gpu_env = lambda g: dict(env, CUDA_VISIBLE_DEVICES=ids[g]) if pin else env      # one GPU: leave the environment alone
    free_gpus: queue.Queue = queue.Queue()
    for i in range(slots):
        free_gpus.put(i % a.gpus)
    env = dict(os.environ, PYTHONIOENCODING="utf-8"); epochs = 20
    if a.sets:
        env["OPEN_SET_SETS"] = a.sets
    env["OPEN_SET_HERO"] = a.hero
    if a.runs_tag:
        env["OPEN_SET_RUNS_TAG"] = a.runs_tag
        a.stages = [x for x in a.stages if x != "tables"]               # open_set_tables.py reads runs/ only
    if a.train_epochs != 20:
        env["OPEN_SET_TRAIN_EPOCHS"] = str(a.train_epochs); epochs = a.train_epochs
        a.stages = [x for x in a.stages if x != "tables"]               # open_set_tables.py reads runs/ (20-epoch) only
    if a.smoke:
        a.seeds, a.folds, epochs = [42], a.folds or ["holes"], 2; env["OPEN_SET_EPOCH_LIMIT"] = "2"
    os.environ.update({k: v for k, v in env.items() if k in ("OPEN_SET_EPOCH_LIMIT", "OPEN_SET_SETS", "OPEN_SET_HERO", "OPEN_SET_TRAIN_EPOCHS", "OPEN_SET_RUNS_TAG")})
    sys.path.insert(0, str(ROOT / "scripts"))
    import open_set_steps as S                                             # after the env var is set
    folds = a.folds or S.CLASSES; fargs = ["--folds", *folds] if a.folds else []
    arms = [(m, st) for m, st in S.ARMS if (a.arms is None or S.run_name(m, st) in a.arms) and not (a.hero != "caao" and m == "m1")]
    if a.hero != "caao":
        a.stages = [x for x in a.stages if x != "tables"]           # open_set_tables.py reads runs/ (caao) only
    py = [sys.executable, "-u", STEPS]; L = OUT / "logs"
    log(f"=== launch profile {a.profile} hero {a.hero}{' SMOKE' if a.smoke else ''}: seeds {a.seeds}, folds {folds}, {len(arms)} arms, stages {a.stages}, "
        f"gen procs {gen_procs}, train slots {slots}, gpus {a.gpus}{' (MIG slices)' if mig else ''}{' pinned: ' + ' '.join(ids) if pin else ''}")
    for s in a.seeds:
        sa = ["--seeds", str(s), *fargs]
        for st in [x for x in a.stages if x in ("plan", "generate", "refine", "dtd", "cutmix", "check")]:
            t0 = time.time(); n = gen_procs if st in ("generate", "refine") else a.cpu_procs if st in ("dtd", "cutmix") else 1
            with ThreadPoolExecutor(n) as ex:
                rcs = list(ex.map(lambda k: run(py + [st, *sa] + (["--shard", str(k), str(n)] if n > 1 else []), L / f"{st}_seed{s}_{k}.log",
                                                gpu_env(k % a.gpus)), range(n)))
            log(f"seed {s} {st}: rc {rcs} in {(time.time() - t0) / 60:.1f} min")
            if any(rcs) and st != "check":
                log(f"seed {s}: STOP, {st} failed (see {L})"); raise SystemExit(1)
        if "train" in a.stages:
            jobs = [(m, stp, k) for k in folds for m, stp in arms]; gate = threading.Lock()

            def one(job: tuple) -> None:
                m, stp, k = job; name = S.run_name(m, stp); out = OUT / S.RUNS_SUB / name / f"seed_{s}" / f"fold_{k}"
                if done(out / "results.json", epochs):
                    return
                with gate:                                               # stagger the start-ups
                    cmd = S.command(s, k, m, stp, []); time.sleep(stagger)
                g = free_gpus.get()
                try:
                    t0 = time.time(); rc = run(cmd, L / f"train_{name}{'' if a.hero == 'caao' else '_' + a.hero}{'' if epochs == 20 or a.smoke else f'_e{epochs}'}{'_' + a.runs_tag if a.runs_tag else ''}_seed{s}_{k}.log", gpu_env(g))
                finally:
                    free_gpus.put(g)
                if not a.keep_checkpoints:
                    shutil.rmtree(out / "checkpoints", ignore_errors=True)
                log(f"seed {s} fold {k} {name} [{a.hero}{'' if epochs == 20 else f' e{epochs}'}]: rc {rc}, {(time.time() - t0) / 60:.1f} min")

            log(f"seed {s}: training {len(jobs)} runs, {slots} at a time")
            with ThreadPoolExecutor(slots) as ex:
                list(ex.map(one, jobs))
    if "tables" in a.stages:
        run([sys.executable, str(ROOT / "scripts/open_set_tables.py")], L / "tables.log", env)
    log("=== launch finished")


if __name__ == "__main__":
    main()

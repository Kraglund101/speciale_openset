#!/usr/bin/env python3
"""Type-inclusive extension of the open-set experiment, configuration 1 (2026-10-23).

Question: does making the evaluated (held-out) anomaly type available during synthesis and detector training improve
detection of that type, measured on identical test images?

Per fold k (held-out type k), 3 real cashew anomalies of type k are RESERVED (same 3 for all seeds and both methods;
chosen by ground-truth defect area, see select_reserved). They leave the evaluation set and become references:

  A  open-set baseline        existing configuration-1 models (diffusion_in_s1 / diffusion_cross_s1): per-image scores
                               from results/open_set_v2/score_dist_c1/ (same command, seeds, lists), re-scored on the
                               reduced test set - NOT retrained.
  B  type-inclusive target    the 45 original synthetics (5 available types, unchanged files) + 9 new type-k synthetics:
                               diffusion_in, references = the 3 reserved cashew anomalies; 54 per epoch.
  C  type-inclusive cross     the 45 original cross-object synthetics + 9 new type-k ones: conditioning image = the
                               existing per-seed cross-object donor of each reserved anomaly (man["pairing"]), placed
                               mask from the reserved cashew anomaly (as in the original cross arm); 54 per epoch.

The 9 new placements of type k use the 9 configuration-1 slots of type k of the seed's manifest (sets.s2[k][:9]: the
canvases of group k, which fold k's original runs do not use) with the original budget references mapped 1:1 (sorted)
onto the reserved ones. Everything else (normals, hero caao, seeds, generator, noise 0.4 / 50 even steps, masks 0.25 x
p90, epochs, test shifts) is the existing pipeline. B / C score the FULL original test list (like A); evaluation drops
the 3 reserved images afterwards, so A, B and C are evaluated by the same code on identical images.

Output (only new paths): results/open_set_v2/typeincl/{reserved.json, test_lists/, prep/, diffusion_in/,
diffusion_cross/, lists/, runs/, TYPEINCL_TABLES.md}.

  python scripts/open_set_typeincl.py select
  python scripts/open_set_typeincl.py plan --seeds 42 ...
  python scripts/open_set_typeincl.py generate --seeds ... [--shard i n]
  python scripts/open_set_typeincl.py refine --seeds ... [--shard i n]
  python scripts/open_set_typeincl.py lists --seeds ...
  python scripts/open_set_typeincl.py check --seeds ...
  python scripts/open_set_typeincl.py train --seeds ... [--folds ..] [--epochs 2 --sub smoke] --slots 12 --stagger 10
  python scripts/open_set_typeincl.py eval
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import open_set_steps as S  # noqa: E402
import pregenerate_synthetic as P  # noqa: E402

T = S.OUT / "typeincl"
SEEDS = S.SEEDS10
ARMS = ["diffusion_in", "diffusion_cross"]
# Budget-matched open-set controls (54 per epoch, NO type-k synthetic), user 2026-10-23 "run both":
#   a54dup: the 45 originals + 9 duplicates of 9 of them (no new information; isolates the 54-vs-45 ratio / step effect)
#   a54new: the 45 originals + 9 EXISTING configuration-2 synthetics of the available types (sets.s2 epoch-1 slots, same
#           budget refs, same group canvases, already generated) - more data of the same five types
VARIANTS = ["ti", "a54dup", "a54new"]
EXPECTED_EVAL = {"holes": 17, "breakage": 17, "burnt": 7, "colour_diff": 6, "colour_same": 7, "scratches": 12}
TAG = "ti"
BOOT = ("import sys; sys.path.insert(0, {s!r}); from pathlib import Path; import train_uninet as T; "
        "T.RUNNER = Path({r!r}); sys.argv = ['train_uninet.py'] + sys.argv[1:]; raise SystemExit(T.main())")


def byid() -> dict:
    refs = S.load_refs(); return {r["id"]: r for k in refs for r in refs[k]}


def reserved() -> dict:
    return json.loads((T / "reserved.json").read_text())["reserved"]


# ---------------------------------------------------------------- select
def select_reserved() -> None:
    """3 per type, by ground-truth defect area (no AUROC involved): the images at the 1/6, 1/2 and 5/6 positions of the
    area-sorted list (small / medium / large). If all 3 share one raw subtype while the type has several, the middle pick
    is replaced by the median-area image of the other subtype. Written once; never re-chosen."""
    f = T / "reserved.json"
    if f.exists():
        print(f"{f} exists - kept"); return
    refs = S.load_refs(); out = {}; info = {}
    for t in S.CLASSES:
        rows = sorted(((float((np.array(Image.open(r["gt_mask"]).convert("L")) > 0).mean()), r["id"], r["raw"]) for r in refs[t]))
        n = len(rows); pos = [round((n - 1) * q) for q in (1 / 6, 1 / 2, 5 / 6)]; pick = [rows[p] for p in pos]
        subs = sorted({r[2] for r in rows})
        if len(subs) > 1 and len({p[2] for p in pick}) == 1:
            other = [r for r in rows if r[2] != pick[0][2]]; pick[1] = other[len(other) // 2]
        out[t] = sorted(p[1] for p in pick)
        info[t] = [{"id": p[1], "raw": p[2], "gt_area_pct": round(100 * p[0], 4)} for p in pick]
    T.mkdir(parents=True, exist_ok=True)
    json.dump({"rule": select_reserved.__doc__, "reserved": out, "details": info}, open(f, "w"), indent=1)
    for t in S.CLASSES:                                   # reduced test lists (identical for every seed: test = all type-k)
        full = [{"id": r["id"], "raw": r["raw"], "image": r["ind_img"], "gt_mask": r["gt_mask"]} for r in refs[t]]
        red = [x for x in full if x["id"] not in out[t]]
        assert len(red) == EXPECTED_EVAL[t], (t, len(red))
        d = T / "test_lists" / f"fold_{t}"; d.mkdir(parents=True, exist_ok=True)
        json.dump(full, open(d / "test_anomalies_full.json", "w"), indent=1)
        json.dump(red, open(d / "test_anomalies_reduced.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


# ---------------------------------------------------------------- plan
def new_slots(man: dict, k: str) -> list[dict]:
    """The 9 configuration-1 slots of type k (sets.s2[k][:9]) with budget refs mapped 1:1 (sorted) to the reserved."""
    R = reserved()[k]; B = sorted(man["budget"][k]); m = dict(zip(B, R)); out = []
    for st in man["sets"]["s2"][k][:S.K]:
        out.append({"i": st["i"], "canvas": st["canvas"], "ref": m[st["ref"]], "orig_ref": st["ref"]})
    assert sorted(x["canvas"] for x in out) == sorted(man["groups"][k]), (k, "slots are not group k's canvases")
    return out


def pdir(s: int, k: str, i: int) -> Path:
    return T / "prep" / f"seed_{s}" / k / f"{i:03d}"


def ipath(arm: str, s: int, k: str, i: int, kind: str = "images") -> Path:
    return T / arm / f"seed_{s}" / k / kind / f"{i:03d}.png"


def stage_plan(seeds: list[int]) -> None:
    ids = byid()
    for s in seeds:
        man = S.manifest(s); n = 0
        for k in S.CLASSES:
            for sl in new_slots(man, k):
                w = pdir(s, k, sl["i"])
                if (w / "meta.json").exists():
                    continue
                w.mkdir(parents=True, exist_ok=True)
                pl = P.place(ids[sl["ref"]], sl["canvas"], s, TAG, k, sl["i"], w)
                P.cached_link(lambda c=sl["canvas"]: Image.open(P.NORMAL_DIR / f"{c}.JPG").convert("RGB"), f"canvas_{sl['canvas']}", w / "canvas.png")
                json.dump({"idx": sl["i"], "normal": f"{sl['canvas']}.JPG", "is_hard": False, "seed": pl["seed"], "cls": k, "ref": sl["ref"],
                           "orig_ref": sl["orig_ref"], "tag": TAG, "placement": pl}, open(w / "meta.json", "w"), indent=1); n += 1
        print(f"plan seed {s}: {n} new placements", flush=True)


# ---------------------------------------------------------------- generate / refine
def jobs(seeds: list[int]):
    ids = byid()
    for s in seeds:
        man = S.manifest(s)
        for k in S.CLASSES:
            for sl in new_slots(man, k):
                r = ids[sl["ref"]]; dn = man["pairing"][k][sl["ref"]]
                yield s, k, sl["i"], "diffusion_in", r["ind_img"], r["ind_mask"], ""
                yield s, k, sl["i"], "diffusion_cross", dn["cr_img"], dn["cr_mask"], dn["donor_id"]


def stage_generate(seeds: list[int], shard: tuple[int, int]) -> None:
    stamp = json.loads((S.OUT / "generator_settings.json").read_text())
    assert stamp == {"ckpt": P.CKPT, "mask_factor": P.MASK_FACTOR, **P.GEN}, ("generator settings differ from the main run", stamp)
    todo = [j for n, j in enumerate(jobs(seeds)) if n % shard[1] == shard[0] and not ipath(j[3], j[0], j[1], j[2]).exists()]
    print(f"generate shard {shard}: {len(todo)} images", flush=True)
    if not todo:
        return
    P.gc.setup_experiment("ResNet"); pipe, ip, t2i = P.gc.load_models(P.h.CKPTS[P.CKPT]); P.gc.EXP = Path(tempfile.mkdtemp(prefix="tigen_"))
    print(f"generator: {P.GEN}", flush=True)
    for n, (s, k, i, arm, img, msk, _d) in enumerate(todo):
        meta = json.loads((pdir(s, k, i) / "meta.json").read_text()); rid = f"{arm[10:]}-s{s}-{TAG}-{k}-{i:03d}"
        dst = ipath(arm, s, k, i); dst.parent.mkdir(parents=True, exist_ok=True)
        P.gc.generate_one(pipe, ip, t2i, canvas_id=meta["normal"].split(".")[0], ref_id=rid, difficulty="easy", defect_map={rid: "defect"},
                          seed=meta["seed"], device="cuda", layout="A", ref_img_override=Path(img), ref_mask_override=Path(msk),
                          placed_mask_override=pdir(s, k, i) / "placed_mask.png", save_raw=True,
                          canvas_override=P.NORMAL_DIR / meta["normal"], caption_override=" ", **P.GEN)
        shutil.copy(P.gc.EXP / "anomaly/imgs/easy" / f"{rid}.png", dst)
        if n % 50 == 0:
            print(f"  generated {n + 1}/{len(todo)}", flush=True)
    print(f"generate shard {shard}: done", flush=True)


def stage_refine(seeds: list[int], shard: tuple[int, int]) -> None:
    from compute_refined_masks import _load_dinov3_cnx
    from refine_inprocess import refine
    dino = None
    for n, (s, k, i, arm, *_r) in enumerate(jobs(seeds)):
        if n % shard[1] != shard[0]:
            continue
        img, dst = ipath(arm, s, k, i), ipath(arm, s, k, i, "masks")
        if dst.exists() or not img.exists():
            continue
        dino = dino or _load_dinov3_cnx("cuda"); dst.parent.mkdir(parents=True, exist_ok=True)
        m = refine(dino, img, pdir(s, k, i), None, factor=P.MASK_FACTOR, thesis_reference=True)
        Image.fromarray(m.astype(np.uint8) * 255).save(dst)
    print(f"refine shard {shard}: done", flush=True)


# ---------------------------------------------------------------- lists / check
def orig_lists(s: int, k: str) -> Path:
    return S.OUT / "lists" / f"seed_{s}" / f"fold_{k}"


def lpath(s: int, k: str, arm: str, var: str = "ti") -> Path:
    return T / "lists" / f"seed_{s}" / f"fold_{k}" / f"epochs_{arm}_{var}.json"


def extra9(man: dict, s: int, k: str, arm: str, orig_ep0: list[dict], var: str) -> list[dict]:
    """The 9 extra entries of a control: available types in CLASSES order, round robin (2,2,2,2,1)."""
    avail = [t for t in S.CLASSES if t != k]; cnt = {t: 0 for t in avail}; out = []
    for j in range(S.K):
        t = avail[j % len(avail)]; c = cnt[t]; cnt[t] += 1
        if var == "a54dup":                                      # c-th original entry of type t (positions t*9 + c)
            out.append(dict(orig_ep0[avail.index(t) * S.K + c]))
        else:                                                    # s2 epoch-1 slot c of type t (existing image)
            st = man["sets"]["s2"][t][S.K + c]; tag, cv, _ = S.resolve(man, "s2", k, t, st); i = st["i"]
            out.append({"image": str(S.ipath(arm, s, tag, t, i)), "mask": str(S.ipath(arm, s, tag, t, i, "masks")),
                        "canvas": str(S.pdir(s, tag, t, i) / "canvas.png")})
    return out


def stage_lists(seeds: list[int]) -> None:
    for s in seeds:
        man = S.manifest(s)
        for k in S.CLASSES:
            new = sorted(new_slots(man, k), key=lambda x: x["i"])
            for arm in ARMS:
                orig = json.load(open(orig_lists(s, k) / f"epochs_{arm}_s1.json"))["epochs"]    # the list A was trained on
                assert orig == json.loads(json.dumps(S.fold_epochs(man, s, k, arm, 1))), (s, k, arm, "original list != fold_epochs")
                add = [{"image": str(ipath(arm, s, k, x["i"])), "mask": str(ipath(arm, s, k, x["i"], "masks")),
                        "canvas": str(pdir(s, k, x["i"]) / "canvas.png")} for x in new]
                ep = [e + add for e in orig]
                lpath(s, k, arm).parent.mkdir(parents=True, exist_ok=True); json.dump({"epochs": ep}, open(lpath(s, k, arm), "w"))
                for var in ("a54dup", "a54new"):
                    x9 = extra9(man, s, k, arm, orig[0], var)
                    json.dump({"epochs": [e + x9 for e in orig]}, open(lpath(s, k, arm, var), "w"))
        print(f"lists seed {s}: done", flush=True)


def stage_check(seeds: list[int]) -> int:
    ids = byid(); res = reserved(); bad = []; train, test, pool = P.normal_pools()
    tn_all = {str(x).split(".")[0] for x in train}; te_all = {str(x).split(".")[0] for x in test}
    for k in S.CLASSES:
        red = json.load(open(T / "test_lists" / f"fold_{k}" / "test_anomalies_reduced.json")); rid = {x["id"] for x in red}
        if len(red) != EXPECTED_EVAL[k] or rid & set(res[k]) or len(res[k]) != 3:
            bad.append((k, "reduced test list"))
    for s in seeds:
        man = S.manifest(s)
        for k in S.CLASSES:
            evaluated = {x["id"] for x in json.load(open(T / "test_lists" / f"fold_{k}" / "test_anomalies_reduced.json"))}
            assert json.load(open(orig_lists(s, k) / "test_anomalies.json")) == json.load(open(T / "test_lists" / f"fold_{k}" / "test_anomalies_full.json"))
            for arm in ARMS:
                ep = json.load(open(lpath(s, k, arm)))["epochs"]; orig = json.load(open(orig_lists(s, k) / f"epochs_{arm}_s1.json"))["epochs"]
                for e, (a, o) in enumerate(zip(ep, orig)):
                    if len(a) != 54 or a[:45] != o:
                        bad.append((s, k, arm, e, "not 45 original + 9")); break
                    canv = []
                    for t in a:
                        for q in ("image", "mask", "canvas"):
                            if not Path(t[q]).exists():
                                bad.append((s, k, arm, "missing", t[q]))
                        meta = json.loads((Path(t["canvas"]).parent / "meta.json").read_text()); canv.append(meta["normal"].split(".")[0])
                        if meta["cls"] == k and meta["ref"] not in res[k]:
                            bad.append((s, k, arm, "type-k synthetic from a non-reserved ref", meta["ref"]))
                        if meta["cls"] == k and meta["ref"] in evaluated:
                            bad.append((s, k, arm, "LEAK: evaluated anomaly used as reference", meta["ref"]))
                    n_k = sum(json.loads((Path(t["canvas"]).parent / "meta.json").read_text())["cls"] == k for t in a)
                    if n_k != 9 or any(json.loads((Path(t["canvas"]).parent / "meta.json").read_text())["cls"] == k for t in a[:45]):
                        bad.append((s, k, arm, e, "type k must appear exactly in the 9 new entries", n_k))
                    if len(set(canv)) != 54 or set(canv) & (tn_all | te_all):
                        bad.append((s, k, arm, e, "canvases not 54 distinct non-train/test normals"))
                    if e > 0:
                        break                                       # stage 1: all 20 epochs are the same list (checked below)
                if any(x != ep[0] for x in ep):
                    bad.append((s, k, arm, "epochs differ (stage 1 must be fixed)"))
                for x in new_slots(man, k):                         # cross donors must not be cashew
                    if arm == "diffusion_cross":
                        d = man["pairing"][k][x["ref"]]
                        if "cashew" in d["cr_img"].lower() or d["donor"] == "cashew":
                            bad.append((s, k, "cross donor is cashew", d["donor_id"]))
    for s in seeds:                                                # budget-matched open-set controls
        man = S.manifest(s)
        for k in S.CLASSES:
            for arm in ARMS:
                orig = json.load(open(orig_lists(s, k) / f"epochs_{arm}_s1.json"))["epochs"]
                for var in ("a54dup", "a54new"):
                    ep = json.load(open(lpath(s, k, arm, var)))["epochs"]
                    if any(x != ep[0] for x in ep) or len(ep) != len(orig):
                        bad.append((s, k, arm, var, "epochs differ / count")); continue
                    a = ep[0]
                    if len(a) != 54 or a[:45] != orig[0]:
                        bad.append((s, k, arm, var, "not 45 original + 9")); continue
                    if var == "a54dup" and any(x not in orig[0] for x in a[45:]):
                        bad.append((s, k, arm, var, "duplicate not from the 45"))
                    if var == "a54new" and any(x in orig[0] for x in a[45:]):
                        bad.append((s, k, arm, var, "'new' entry is one of the 45"))
                    for t in a:
                        for q in ("image", "mask", "canvas"):
                            if not Path(t[q]).exists():
                                bad.append((s, k, arm, var, "missing", t[q]))
                        meta = json.loads((Path(t["canvas"]).parent / "meta.json").read_text())
                        if meta["cls"] == k:
                            bad.append((s, k, arm, var, "LEAK: held-out type in an open-set control"))
                        if meta["ref"] not in man["budget"][meta["cls"]]:
                            bad.append((s, k, arm, var, "ref outside the budget", meta["cls"], meta["ref"]))
    print(f"check: {len(bad)} problem(s)", *bad[:20], sep="\n  "); return len(bad)


# ---------------------------------------------------------------- train (score runner, like open_set_scores.py)
def out_dir(sub: str, arm: str, s: int, k: str, var: str = "ti") -> Path:
    return T / ("runs" if sub == "runs" else f"runs_{sub}") / f"{arm}_{var}" / f"seed_{s}" / f"fold_{k}"


def command(arm: str, s: int, k: str, sub: str, extra: list[str], var: str = "ti") -> list[str]:
    def rl(s_, k_, m_, st_):                                          # read-only: original lists + the new epoch list
        d = orig_lists(s_, k_)
        return {"dir": d, "train_normals": d / "train_normals.txt", "m1_extra": d / "m1_extra.txt",
                "test": d / "test_anomalies.json", "n_test": None, "epochs": lpath(s_, k_, m_, var), "missing": 0}
    orig, S.run_lists = S.run_lists, rl
    try:
        c = S.command(s, k, arm, 1, extra)
    finally:
        S.run_lists = orig
    assert c[2].endswith("train_uninet.py"), c[:3]
    c[c.index("--output-dir") + 1] = str(out_dir(sub, arm, s, k, var))
    return [sys.executable, "-u", "-c", BOOT.format(s=str(ROOT / "scripts"), r=str(ROOT / "scripts/score_runner.py"))] + c[3:]


def done(d: Path, n: int) -> bool:
    try:
        r = json.load(open(d / "results.json")); k = [x for x in r if x.startswith("Mode") and " @ " not in x]
        return bool(k) and len(r[k[0]]["img_aurocs"]) >= n and (d / "image_scores.json").exists()
    except Exception:
        return False


def stage_train(seeds, folds, slots, stagger, epochs, sub, variants=("ti",)) -> None:
    extra = [] if epochs == S.E else ["--epochs", str(epochs)]; gate = threading.Lock(); env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def one(job):
        arm, s, k, var = job; d = out_dir(sub, arm, s, k, var)
        if done(d, epochs):
            return
        with gate:
            cmd = command(arm, s, k, sub, extra, var); time.sleep(stagger)
        d.mkdir(parents=True, exist_ok=True); t0 = time.time()
        with open(d / "stdout.log", "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        shutil.rmtree(d / "checkpoints", ignore_errors=True)
        print(f"{time.strftime('%m-%d %H:%M')}  [typeincl] {arm}_{var} seed {s} fold {k}: rc {rc}, {(time.time() - t0) / 60:.1f} min, "
              f"scores {'ok' if done(d, epochs) else 'MISSING'}", flush=True)

    js = [(arm, s, k, var) for s in seeds for var in variants for arm in ARMS for k in folds]
    print(f"[typeincl] {len(js)} runs, {slots} at a time", flush=True)
    with ThreadPoolExecutor(slots) as ex:
        list(ex.map(one, js))


# ---------------------------------------------------------------- eval
LEVELS = ["0x", "1x", "2x", "3x", "4x"]; AGG = "standard_paper"


def scores_auroc(f: Path, drop: set[str]) -> dict:
    """Per level and epoch: image AUROC on the test images whose id is not in `drop` (normals kept)."""
    from sklearn.metrics import roc_auc_score
    d = json.load(open(f)); out = {lv: {} for lv in LEVELS}
    for x in d:
        if x["agg"] != AGG or x["shift"] not in LEVELS:
            continue
        assert not x.get("files_mismatch"), f
        keep = [j for j, p in enumerate(x["files"]) if not (x["labels"][j] == 1 and Path(p).stem in drop)]
        out[x["shift"]][x["epoch"]] = roc_auc_score([x["labels"][j] for j in keep], [x["scores"][j] for j in keep])
    return out


def pick(a: dict) -> tuple[float, float, int]:
    """Original convention: epoch with the best 0x AUROC (first on ties); 0x and mean 1x-4x at that epoch."""
    eps = sorted(a["0x"]); b = max(eps, key=lambda e: (a["0x"][e], -e))
    return a["0x"][b], float(np.mean([a[lv][b] for lv in LEVELS[1:]])), b


def stage_eval(seeds) -> None:
    """A = open-set (score_dist_c1), D = a54dup, N = a54new, I = type-inclusive; all on the reduced test set, best-0x epoch."""
    res = reserved(); R = {}
    base = {"diffusion_in": "diffusion_in_s1", "diffusion_cross": "diffusion_cross_s1"}
    src = lambda arm, s, k: {"A": S.OUT / "score_dist_c1" / base[arm] / f"seed_{s}" / f"fold_{k}" / "image_scores.json",
                             "D": out_dir("runs", arm, s, k, "a54dup") / "image_scores.json",
                             "N": out_dir("runs", arm, s, k, "a54new") / "image_scores.json",
                             "I": out_dir("runs", arm, s, k, "ti") / "image_scores.json"}
    for arm in ARMS:
        for s in seeds:
            for k in S.CLASSES:
                for c, f in src(arm, s, k).items():
                    if f.exists():
                        R[(arm, s, k, c)] = pick(scores_auroc(f, set(res[k])))
    se = lambda z: z.std(ddof=1) / np.sqrt(len(z)) if len(z) > 1 else float("nan")
    f = lambda z: f"{z.mean():.3f} +- {se(z):.3f}" if len(z) else "-"
    fd = lambda z: f"{z.mean():+.3f} +- {se(z):.3f}" if len(z) else "-"
    name = {"A": "open-set (45)", "D": "open-set 45+9 dup", "N": "open-set 45+9 same-type new", "I": "type-inclusive (45+9 type k)"}
    lab = {"diffusion_in": "target-object", "diffusion_cross": "cross-object"}
    pairs = [("I", "A", "PRIMARY type-incl - open-set"), ("D", "A", "budget: dup - open-set"), ("N", "A", "same-type data: new - open-set"),
             ("I", "D", "type-incl - dup"), ("I", "N", "type-incl - new (budget and data matched)")]
    out = ["# Type-inclusive vs open-set (configuration 1), identical reduced test images", "",
           f"Reserved per type (removed from evaluation, references of the type-inclusive models): {json.dumps(res)}",
           "Epoch = best 0x on the reduced set (original convention; nuisance 1-4x = mean of 1x-4x at that epoch).",
           "A = existing open-set models (score_dist_c1 reruns, not retrained). D/N/I = new runs (54 synthetics per epoch).",
           "Values: mean +- SE over seeds; differences are paired (same seed and fold).", ""]
    for arm in ARMS:
        out += [f"## {lab[arm]}", ""]
        for lvl, idx in (("0x", 0), ("1-4x", 1)):
            out += [f"### {lvl} AUROC", "", "| held-out type | n | " + " | ".join(name[c] for c in "ADNI") + " | " + " | ".join(p[2] for p in pairs) + " |",
                    "|---|---|" + "---|" * (4 + len(pairs))]
            for k in S.CLASSES:
                ss = [s for s in seeds if all((arm, s, k, c) in R for c in "ADNI")]
                val = {c: np.array([R[(arm, s, k, c)][idx] for s in ss]) for c in "ADNI"}
                out.append(f"| {k} | {len(ss)} | " + " | ".join(f(val[c]) for c in "ADNI") + " | " + " | ".join(fd(val[a] - val[b]) for a, b, _ in pairs) + " |")
            ss = [s for s in seeds if all((arm, s, k, c) in R for k in S.CLASSES for c in "ADNI")]
            mac = {c: np.array([np.mean([R[(arm, s, k, c)][idx] for k in S.CLASSES]) for s in ss]) for c in "ADNI"}
            out += [f"| **macro (6 folds)** | {len(ss)} | " + " | ".join(f(mac[c]) for c in "ADNI") + " | " + " | ".join(fd(mac[a] - mac[b]) for a, b, _ in pairs) + " |", ""]
        for k in ("colour_same", "scratches"):
            rows = [(s, R[(arm, s, k, "A")], R[(arm, s, k, "I")]) for s in seeds if (arm, s, k, "A") in R and (arm, s, k, "I") in R]
            if rows:
                out += [f"**{lab[arm]}, {k}, per seed** (open-set -> type-inclusive; selected epoch):", ""] + \
                       [f"- seed {s}: 0x {a[0]:.3f} -> {b[0]:.3f} (ep {a[2]} -> {b[2]}), 1-4x {a[1]:.3f} -> {b[1]:.3f}" for s, a, b in rows] + [""]
    (T / "TYPEINCL_TABLES.md").write_text("\n".join(out) + "\n"); print("\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "plan", "generate", "refine", "lists", "check", "train", "eval"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS); ap.add_argument("--folds", nargs="+", default=S.CLASSES)
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1]); ap.add_argument("--slots", type=int, default=12)
    ap.add_argument("--variants", nargs="+", default=["ti"], choices=VARIANTS)
    ap.add_argument("--stagger", type=int, default=10); ap.add_argument("--epochs", type=int, default=S.E); ap.add_argument("--sub", default="runs")
    a = ap.parse_args(); sh = tuple(a.shard)
    if a.stage == "select":
        select_reserved()
    elif a.stage == "plan":
        stage_plan(a.seeds)
    elif a.stage == "generate":
        stage_generate(a.seeds, sh)
    elif a.stage == "refine":
        stage_refine(a.seeds, sh)
    elif a.stage == "lists":
        stage_lists(a.seeds)
    elif a.stage == "check":
        raise SystemExit(1 if stage_check(a.seeds) else 0)
    elif a.stage == "train":
        stage_train(a.seeds, a.folds, a.slots, a.stagger, a.epochs, a.sub, tuple(a.variants))
    else:
        stage_eval(a.seeds)


if __name__ == "__main__":
    main()

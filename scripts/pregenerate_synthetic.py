#!/usr/bin/env python3
"""Pregenerate the synthetic cashew training pools (plan: ~/.claude/plans/peppy-whistling-canyon.md).

Per seed s (default 42 123 7 99 256) and class (holes, breakage, burnt, colour_diff, colour_same, scratches):
  - ONE fixed pool of 45 canvases per seed from the spare normals (all 500 minus splits.json normals.train (100) and
    normals.test (50)), shared by all 6 leave-one-type-out folds (M1 = 100 train normals + these 45).
  - 2026-10-01 (user): per fold (held-out type k) and epoch, a seeded permutation re-partitions the 45 into 5 groups of
    9, one per train type (schedule()) -> images are generated PER FOLD. The block below (fixed 9 per class) describes
    the superseded v1 schedule, except the per-type reference counts, which are unchanged.
  - 180 steps per class (n = the class's real anomalies, u = 180/n uses each, q = u // 9, r = u % 9):
      base: every anomaly on EVERY canvas q times (full permutation);
      leftover (colour_diff r=2, scratches r=3): each anomaly on r RANDOM distinct canvases, drawn so every canvas gets
      the same number of extras -> each canvas exactly 20 x. Order: split into 20 epochs (steps 9e..9e+8) by a random
      bipartite matching (epoch_order) -> every epoch = 9 different canvases AND 9 different anomalies.
    image seed = hash(s, class, t) drives placement AND diffusion noise.
  - footprint = the reference cashew's OWN VisA GT mask, placed (flip / rot90 / scale 0.6-1.0 / position,
    place_easy_mask) inside the canvas FG (BiRefNet), scale 0.5-1.0; NO piece filtering. SAME footprint/canvas/seed in both arms.
  - diffusion_in: CLIP reference = the cashew defect; diffusion_cross: = a random same-type donor (pipe_fryum /
    fire_hood pit for holes), redrawn per seed, every used donor used equally often (pairing(); no CLIP similarity).
  - mask = refine(factor 0.6, vae_reference=True) = 0.6 x p90 vs the decoded canvas. No rejection.
Generator: core-50k @20k, visual CFG g7, noise 0.7, 50 steps, band 2; caption --caption template|none.

Layout (results/open_set_data/):
  manifest/seed_{s}.json, normals/seed_{s}/m1_canvases.txt, normals/train_normals.txt, ref_masks/
  prep/seed_{s}/fold_{k}/{cls}/{i:03d}/{canvas,placed_mask,ref_image,ref_mask}.png + meta.json   (shared by both arms)
  diffusion_{in,cross}/seed_{s}/fold_{k}/{cls}/{images,masks}/{i:03d}.png

Stages (each resumable: existing files are skipped):
  python scripts/pregenerate_synthetic.py plan     [--seeds ...] [--steps 180]
  python scripts/pregenerate_synthetic.py folds                              (leave-one-type-out test folds -> folds.json)
  python scripts/pregenerate_synthetic.py generate [--seeds ...] [--steps N] [--classes ...] [--caption none]
  python scripts/pregenerate_synthetic.py refine   [--seeds ...] [--steps N]
  python scripts/pregenerate_synthetic.py dtd      [--seeds ...] [--steps N]   (DRAEM default, beta ~ U(0.2, 1.0))
  python scripts/pregenerate_synthetic.py cutmix   [--seeds ...] [--steps N]   (derangement pi, patch inside donor cashew)
  python scripts/pregenerate_synthetic.py check    [--seeds ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import generate_cashew as gc  # noqa: E402
import data_sources as h  # noqa: E402
from data_sources import CASHEW_MASKS, DATA, REAL_ANOMALIES  # noqa: E402
from src.utils.placement_utils import place_easy_mask  # noqa: E402

OUT = ROOT / "results/open_set_data"
CLASSES = ["holes", "breakage", "burnt", "colour_diff", "colour_same", "scratches"]
SEEDS = [42, 123, 7, 99, 256]
PER_CLASS_CANVASES, STEPS = 9, 180          # per type: 9 canvases per epoch, 180 synthetics = 20 epochs x 9
POOL_CANVASES = 54                          # canvas pool per seed = 6 types x 9; dealt anew every epoch (user 2026-10-03)
EPOCHS = STEPS // PER_CLASS_CANVASES
SCALE = (0.5, 1.0)
MAX_PLACEMENT_TRIES = 50
# raw_background=True = B (thesis as written): the decoded generation is blended into the RAW canvas, a*G + (1-a)*raw
# (user 2026-09-27: all seeds regenerated this way). Masks are refined against the MATCHED reference (stage_refine).
# Noise strength 0.4 with 50 evenly spaced denoising steps from its start timestep (381): chosen by the user 2026-10-04
# after the noise-strength test (results/noise_strength_test/TABLES.md: flat from 0.2-0.4 up, worse below).
# Until then: noise_strength 0.7 on the stock 50-step grid (= 35 steps, even_steps False) = GEN_070.
GEN = dict(noise_strength=0.4, num_steps=50, even_steps=True, guidance_scale=7.0, band_mode=2, cfg_mode="visual", raw_background=True)
GEN_070 = dict(GEN, noise_strength=0.7, even_steps=False)
MASK_FACTOR = 0.25  # refined label = factor x p90; 0.25 chosen 2026-09-30 (mask ladder: scripts/inpaint_region_test.py)
CKPT = "clean-20k"   # 2026-09-30: clean generator (was "core-50k @20k")
FG_DIR = gc.CASHEW_ROOT / "birefnet_masks" / "Normal"
NORMAL_DIR = gc.CASHEW_ROOT / "Data" / "Images" / "Normal"
SPLITS = ROOT / "data" / "splits.json"


def image_seed(s: int, fold: str, cls: str, i: int, attempt: int = 0) -> int:
    """Deterministic per-image seed (placement + diffusion noise); attempt > 0 = placement retry."""
    return int(hashlib.sha256(f"{s}|{fold}|{cls}|{i}|{attempt}".encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def sub(meth: str, s: int, fold: str | None, cls: str) -> Path:
    """<method>/seed_s/<type> (method = prep | diffusion_in | diffusion_cross | dtd | cutmix). Images are shared by all
    folds (fold=None); a fold name gives the superseded per-fold layout <method>/seed_s/fold_<k>/<type>."""
    return OUT / meth / f"seed_{s}" / cls if fold is None else OUT / meth / f"seed_{s}" / f"fold_{fold}" / cls


def fold_types(fold: str) -> list[str]:
    return [c for c in CLASSES if c != fold]


def iter_steps(man: dict, steps: int = STEPS):
    """(None, type, step) over the 6 types of a seed manifest (None = images shared by all folds)."""
    for cls in CLASSES:
        for st in man["steps"][cls][:steps]:
            yield None, cls, st


def normal_pools() -> tuple[list[str], list[str], list[str]]:
    """(train normals, test normals, spare canvas pool) as stems; the pool excludes train AND test normals."""
    sp = json.loads(SPLITS.read_text())["normals"]
    train = sorted(Path(x).stem for x in sp["train"]); test = sorted(Path(x).stem for x in sp["test"])
    allN = sorted(p.stem for p in NORMAL_DIR.glob("*.JPG"))
    pool = [x for x in allN if x not in set(train) | set(test)]
    assert len(allN) == 500 and len(train) == 100 and len(test) == 50 and len(pool) == 350, (len(allN), len(pool))
    return train, test, pool


def load_refs() -> dict[str, list[dict]]:
    """Per class: the real cashew anomalies (id, label, GT mask, in-dist CLIP reference). The 84 ids, their VisA label
    and type come from data/real_anomalies.json (cross-object donors: random per seed, see pairing())."""
    h.OUT = OUT
    refs = {k: [] for k in CLASSES}
    for r in json.loads(REAL_ANOMALIES.read_text(encoding="utf-8")):
        refs[r["type"]].append({
            "id": r["id"], "raw": r["label"], "cls": r["type"], "gt_mask": str(CASHEW_MASKS / f"{r['id']}.png"),
            "ind_img": str(gc.CASHEW_ROOT / r["image"]), "ind_mask": str(h.normalise_mask(CASHEW_MASKS / f"{r['id']}.png", f"cashew_{r['id']}"))})
    for k in refs:
        refs[k].sort(key=lambda x: x["id"])
    return refs


_POOLS: dict | None = None
# donors excluded after inspection (user): fire_hood_0045 (pit filled with blue foreign material), fire_hood_0072
EXCLUDED_DONOR_PREFIXES = ("fire_hood_0045_", "fire_hood_0072_")
SPECK_MAX_PX, SPECK_MIN_DIST = 10, 50


def drop_isolated_specks(mask_path: Path, name: str) -> Path:
    """Remove annotation specks from a reference mask: a connected piece is removed ONLY if it is <= SPECK_MAX_PX px AND
    more than SPECK_MIN_DIST px from every other piece of the mask. Real fragments next to a defect (scratch pieces,
    burnt clusters) are never touched. Over all 484 masks this matches exactly 3 pieces (macaroni1 021 9 px,
    pipe_fryum 031 7 px, pipe_fryum 075 1 px). Returns the original path if nothing is removed, else a *_clean.png copy."""
    from scipy import ndimage as nd
    a = np.array(Image.open(mask_path).convert("L")) > 127
    lab, n = nd.label(a); keep = a.copy(); removed = 0
    for j in range(1, n + 1):
        pj = lab == j
        if n > 1 and pj.sum() <= SPECK_MAX_PX and nd.distance_transform_edt(~(a & ~pj))[pj].min() > SPECK_MIN_DIST:
            keep &= ~pj; removed += 1
    if not removed:
        return mask_path
    out = Path(mask_path).with_name(f"{name}_clean.png")
    Image.fromarray(keep.astype(np.uint8) * 255).save(out)
    return out


def donor_pools() -> dict[str, list[dict]]:
    """Same-type cross-object donors (label rules: data_sources.CANON): holes -> RealIAD fire_hood
    'pit'; every other type -> pipe_fryum single-label defects of that type. Masks normalised to 0/255, unique names."""
    global _POOLS
    if _POOLS is None:
        import data_sources as mp
        mp.train_items._caps = {c["image_path"]: c["caption"] for c in json.loads((mp.D / "captions_from_master.json").read_text(encoding="utf-8"))}
        mp.train_items._master = json.loads((mp.D / "master_training.json").read_text(encoding="utf-8"))
        pipe = mp.visa_items("pipe_fryum", [mp.REST / "pipe_fryum/Data/Images/Anomaly"], mp.REST / "pipe_fryum/Data/Masks/Anomaly",
                             mp.REST / "pipe_fryum/image_anno.csv")
        fh = mp.train_items("fire_hood", "holes", lambda e: "fire_hood" in e["image_path"] and (e.get("defect_type") or "").lower() == "pit")
        # breakage top-up (user, 2026-09-26): VisA macaroni1 "chip around edge and corner" (17) -> breakage pool
        # = pipe_fryum corner/edge (15) + macaroni1 chips (17) = 32 >= the 20 cashew breakages -> 1-to-1 like all types
        import csv
        mac = []
        for r in csv.reader(open(mp.REST / "macaroni1/image_anno.csv")):
            if len(r) > 1 and "Anomaly" in r[0] and [t.strip().lower() for t in r[1].split(",") if t.strip()] == ["chip around edge and corner"]:
                stem = Path(r[0]).stem
                img = next((mp.REST / "macaroni1/Data/Images/Anomaly" / f"{stem}{e}" for e in (".JPG", ".jpg", ".png")
                            if (mp.REST / "macaroni1/Data/Images/Anomaly" / f"{stem}{e}").exists()), None)
                msk = mp.REST / "macaroni1/Data/Masks/Anomaly" / f"{stem}.png"
                if img and msk.exists():
                    mac.append({"src": "macaroni1", "id": stem, "img": img, "mask": msk, "cls": "breakage"})
        h.OUT = OUT; _POOLS = {k: [] for k in CLASSES}
        for x in sorted(pipe + fh + mac, key=lambda x: (x["src"], str(x["img"]))):
            if Path(x["img"]).stem.startswith(EXCLUDED_DONOR_PREFIXES):
                continue
            if x["cls"] in _POOLS and (x["cls"] == "holes") == (x["src"] == "fire_hood"):
                name = f"{x['src']}_{Path(x['img']).stem}"
                _POOLS[x["cls"]].append({"donor": x["src"], "donor_id": str(x["id"]), "cr_img": str(x["img"]),
                                         "cr_mask": str(drop_isolated_specks(h.normalise_mask(Path(x["mask"]), name), name))})
    return _POOLS


def pairing(s: int, refs: dict) -> dict:
    """Random same-type cross donor per cashew anomaly, redrawn per seed, every USED donor used equally often
    (RealIAD: one random camera view per physical sample):
    D >= N donors: random 1-to-1; D < N (breakage 15 for 20): m = largest divisor of N that is <= D donors (10), each
    used N/m times (2) -> no donor gets more weight than another."""
    pools = donor_pools(); out = {}
    for cls in CLASSES:
        rng = np.random.default_rng(method_seed(s, cls, -1, "pairing"))
        ids = [r["id"] for r in refs[cls]]; N = len(ids)
        # RealIAD photographs one physical sample from up to 5 cameras (C1-C5): group views by sample and draw ONE random
        # view per sample, so two cashew anomalies never get two views of the same physical defect
        groups: dict[str, list[dict]] = {}
        for x in pools[cls]:
            groups.setdefault(x["donor"] + ":" + re.sub(r"_C\d_.*$", "", Path(x["cr_img"]).stem), []).append(x)   # source:sample
        keys = sorted(groups); D = len(keys)
        m = N if D >= N else max(d for d in range(1, D + 1) if N % d == 0)
        chosen = [groups[keys[j]][int(rng.integers(len(groups[keys[j]])))] for j in rng.permutation(D)[:m]]
        cash = [ids[j] for j in rng.permutation(N)]
        out[cls] = {cid: chosen[j % m] for j, cid in enumerate(cash)}
    return out


def epoch_refs(ids: list[str], u: int, E: int, k: int, rng) -> list[list[str]]:
    """Split 'every anomaly u times' into E epochs of k DIFFERENT anomalies (n*u = E*k, u <= E). Greedy per epoch: all
    'tight' anomalies (remaining uses == remaining epochs) first, then the most-remaining ones, random tie-break ->
    always feasible, every anomaly exactly u times."""
    left = {a: u for a in ids}; out = []
    for e in range(E):
        R = E - e; tight = [a for a in ids if left[a] == R]
        rest = sorted((a for a in ids if 0 < left[a] < R), key=lambda a: (-left[a], rng.random()))
        pick = tight + rest[:k - len(tight)]
        assert len(tight) <= k and len(pick) == k, (len(tight), len(pick))
        for a in pick:
            left[a] -= 1
        out.append(pick)
    assert all(v == 0 for v in left.values())
    return out


def schedule(s: int, refs: dict, pool: list[str], steps: int) -> dict:
    """Seed s (design fixed with the user 2026-10-03): ONE pool of 54 canvases. Every epoch a seeded random permutation
    deals them 9 to each of the 6 types -> every canvas carries exactly one synthetic per epoch, no canvas is tied to a
    type, and the images do not depend on the fold: fold k trains on the other 5 types' images (45 synthetics on 45
    canvases per epoch; type k's 9 canvases sit out that epoch), M1 on the same 45 canvases clean. Per type: every real
    anomaly used 180/n times, 9 DIFFERENT anomalies per epoch (epoch_refs), randomly matched to that epoch's 9 canvases.
    Step i = 9e + j (epoch e). The same manifest is used by all methods."""
    rng = np.random.default_rng(s)
    canv = sorted(str(x) for x in rng.choice(pool, size=POOL_CANVASES, replace=False))
    assert len(CLASSES) * PER_CLASS_CANVASES == POOL_CANVASES
    prng = np.random.default_rng(method_seed(s, "pool54", -1, "partition")); epochs = []
    for e in range(steps // PER_CLASS_CANVASES):
        perm = [canv[j] for j in prng.permutation(POOL_CANVASES)]
        epochs.append({t: perm[g * PER_CLASS_CANVASES:(g + 1) * PER_CLASS_CANVASES] for g, t in enumerate(CLASSES)})
    out = {"seed": s, "design": "pool54", "pool": canv, "epochs": epochs, "steps": {}}
    for cls in CLASSES:
        ids = [r["id"] for r in refs[cls]]; n = len(ids); u = steps // n
        assert n * u == steps and u <= len(epochs), (cls, n, u)
        rrng = np.random.default_rng(method_seed(s, cls, -1, "refs"))
        blocks = epoch_refs(ids, u, len(epochs), PER_CLASS_CANVASES, rrng); out["steps"][cls] = []
        for e, blk in enumerate(blocks):
            cs = epochs[e][cls]; order = rrng.permutation(len(blk))
            out["steps"][cls] += [{"i": e * PER_CLASS_CANVASES + j, "epoch": e, "ref": blk[order[j]], "canvas": cs[j]}
                                  for j in range(len(blk))]
    return out


def schedule_v1_fixed_groups(s: int, refs: dict, pool: list[str], steps: int) -> dict:
    """SUPERSEDED 2026-10-01 (fixed 9 canvases per type, 54 per seed). Kept for reference only - not called."""
    rng = np.random.default_rng(s)
    canv = [str(x) for x in rng.choice(pool, size=PER_CLASS_CANVASES * len(CLASSES), replace=False)]
    out = {"seed": s, "canvases": {}, "steps": {}}
    for ci, cls in enumerate(CLASSES):
        cc = canv[ci * PER_CLASS_CANVASES:(ci + 1) * PER_CLASS_CANVASES]
        out["canvases"][cls] = cc
        ids = [r["id"] for r in refs[cls]]; n = len(ids); K = PER_CLASS_CANVASES
        u = steps // n; q, rem = divmod(u, K)
        assert n * u == steps and (n * rem) % K == 0, (cls, n, u)
        per_canvas = {c: [a for a in ids for _ in range(q)] for c in cc}          # base: full permutation, q times
        if rem:                                                                    # leftover: random, balanced over canvases
            while True:
                slots = [c for c in cc for _ in range(n * rem // K)]
                rng.shuffle(slots)
                chunks = [slots[j * rem:(j + 1) * rem] for j in range(n)]
                if all(len(set(ch)) == rem for ch in chunks):                     # distinct extra canvases per anomaly
                    break
            for a, ch in zip(ids, chunks):
                for c in ch:
                    per_canvas[c].append(a)
        order = epoch_order(per_canvas, cc, ids, steps // K, rng)
        out["steps"][cls] = [{"i": t, "ref": a, "canvas": c} for t, (a, c) in enumerate(order)]
    return out


def epoch_order(per_canvas: dict, cc: list[str], ids: list[str], E: int, rng) -> list[tuple[str, str]]:
    """Split the (anomaly, canvas) multiset into E epochs, each = one image per canvas with 9 DIFFERENT anomalies.
    Bipartite anomaly-canvas multigraph with max degree E (canvases have degree exactly E) -> by Koenig's edge-colouring
    theorem it splits into E matchings. Greedy per epoch: max-weight matching (scipy) that must cover every canvas and
    every 'tight' anomaly (remaining uses == remaining epochs), random tie-breaking -> always feasible, stays random."""
    from scipy.optimize import linear_sum_assignment
    left = {(a, c): 0 for a in ids for c in cc}
    for c in cc:
        for a in per_canvas[c]:
            left[(a, c)] += 1
    order = []
    for e in range(E):
        rem = E - e
        deg = {a: sum(left[(a, c)] for c in cc) for a in ids}
        W = np.full((len(cc), len(ids)), -1e9)
        for i, c in enumerate(cc):
            for j, a in enumerate(ids):
                if left[(a, c)] > 0:
                    W[i, j] = (1e6 if deg[a] == rem else 0) + 1e3 * deg[a] + rng.random()
        ri, ci = linear_sum_assignment(-W)
        blk = []
        for i, j in zip(ri, ci):
            assert W[i, j] > -1e8, "no feasible matching"
            a, c = ids[j], cc[i]; left[(a, c)] -= 1; blk.append((a, c))
        assert len({a for a, _ in blk}) == len(cc)
        rng.shuffle(blk); order += blk
    assert all(v == 0 for v in left.values())
    return order


def place(ref: dict, canvas: str, s: int, fold: str, cls: str, i: int, work: Path) -> dict:
    """Place the cashew's own GT mask on the canvas FG; retry with derived seeds if it does not fit."""
    fg = h.binary(FG_DIR / f"{canvas}_binary.png") > 0.5
    m = np.array(Image.open(ref["gt_mask"]).convert("L")) > 0
    if m.shape != fg.shape:
        m = h.resize_mask(m, fg.shape)
    for attempt in range(MAX_PLACEMENT_TRIES):
        sd = image_seed(s, fold, cls, i, attempt); rec: list = []
        placed = place_easy_mask(m.astype(np.float32), fg.astype(np.float32), scale_range=SCALE, seed=sd, record=rec)
        if placed is None:
            continue
        fp = (placed > 0.5)
        if not fp.any():
            continue
        Image.fromarray(fp.astype(np.uint8) * 255).save(work / "placed_mask.png")   # exact placed GT mask, no filtering
        return {"seed": sd, "attempt": attempt, "record": rec}
    raise RuntimeError(f"placement failed {MAX_PLACEMENT_TRIES}x: seed {s} fold {fold} {cls} step {i} ref {ref['id']} canvas {canvas}")


def cached_link(src_fn, key: str, dst: Path) -> None:
    """Write a big file (full-res canvas / reference) ONCE into OUT/cache and hard-link it into the prep folder
    (disk is ~98 % full; 5400 copies would be > 20 GB). The refiner / UniNet read prep/*/canvas.png as usual."""
    c = OUT / "cache" / f"{key}.png"
    if not c.exists():
        c.parent.mkdir(parents=True, exist_ok=True); src_fn().save(c)
    if not dst.exists():
        os.link(c, dst)


def stage_plan(seeds: list[int], steps: int) -> None:
    train, test, pool = normal_pools(); refs = load_refs()
    (OUT / "manifest").mkdir(parents=True, exist_ok=True); (OUT / "normals").mkdir(parents=True, exist_ok=True)
    (OUT / "normals" / "train_normals.txt").write_text("\n".join(train) + "\n")
    json.dump(refs, open(OUT / "references.json", "w"), indent=1)
    byid = {r["id"]: r for k in refs for r in refs[k]}
    for s in seeds:
        man = schedule(s, refs, pool, steps)
        man["pairing"] = pairing(s, refs)                         # random same-type cross donor per anomaly, this seed
        d = OUT / "normals" / f"seed_{s}"; d.mkdir(parents=True, exist_ok=True)
        (d / "m1_canvases.txt").write_text("\n".join(man["pool"]) + "\n")      # the 54-pool (a fold uses 45 of them per epoch)
        for fold, cls, st in iter_steps(man):
            work = sub("prep", s, fold, cls) / f"{st['i']:03d}"
            if (work / "meta.json").exists():
                st.update(json.loads((work / "meta.json").read_text())["placement"]); continue
            work.mkdir(parents=True, exist_ok=True); ref = byid[st["ref"]]
            pl = place(ref, st["canvas"], s, "pool54", cls, st["i"], work); st.update(pl)
            cv = st["canvas"]
            cached_link(lambda: Image.open(NORMAL_DIR / f"{cv}.JPG").convert("RGB"), f"canvas_{cv}", work / "canvas.png")
            cached_link(lambda: Image.open(ref["ind_img"]).convert("RGB"), f"ref_{ref['id']}_image", work / "ref_image.png")
            cached_link(lambda: Image.open(ref["ind_mask"]).convert("L"), f"ref_{ref['id']}_mask", work / "ref_mask.png")
            json.dump({"idx": st["i"], "epoch": st["epoch"], "normal": f"{st['canvas']}.JPG", "is_hard": False,
                       "seed": pl["seed"], "cls": cls, "ref": ref["id"], "donor": man["pairing"][cls][ref["id"]]["donor"],
                       "donor_id": man["pairing"][cls][ref["id"]]["donor_id"], "placement": pl},
                      open(work / "meta.json", "w"), indent=1)
        print(f"  plan seed {s}: " + ", ".join(f"{c} {len(man['steps'][c])}" for c in CLASSES), flush=True)
        json.dump(man, open(OUT / "manifest" / f"seed_{s}.json", "w"), indent=1)


def stage_generate(seeds: list[int], steps: int, classes: list[str], caption: str) -> None:
    refs = json.loads((OUT / "references.json").read_text()); byid = {r["id"]: r for k in refs for r in refs[k]}
    gc.setup_experiment("ResNet")
    pipe, ip, t2i = gc.load_models(h.CKPTS[CKPT]); gc.EXP = Path(tempfile.mkdtemp(prefix="pregen_"))
    for s in seeds:
        man = json.loads((OUT / "manifest" / f"seed_{s}.json").read_text()); last = None
        for fold, cls, st in iter_steps(man, steps):
            if cls not in classes:
                continue
            if last and last != (fold, cls):
                print(f"  generate seed {s} {last[1]}: done", flush=True)
            last = (fold, cls)
            prep = sub("prep", s, fold, cls) / f"{st['i']:03d}"; meta = json.loads((prep / "meta.json").read_text())
            ref = byid[meta["ref"]]; dn = man["pairing"][cls][meta["ref"]]
            for arm, img, mask in (("in", ref["ind_img"], ref["ind_mask"]), ("cross", dn["cr_img"], dn["cr_mask"])):
                dst = sub(f"diffusion_{arm}", s, fold, cls) / "images" / f"{st['i']:03d}.png"
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                rid = f"{arm}-s{s}-{cls}-{st['i']:03d}"
                gc.generate_one(pipe, ip, t2i, canvas_id=meta["normal"].split(".")[0], ref_id=rid, difficulty="easy",
                                defect_map={rid: ref["raw"]}, seed=meta["seed"], device="cuda", layout="A",
                                ref_img_override=Path(img), ref_mask_override=Path(mask), placed_mask_override=prep / "placed_mask.png",
                                save_raw=True, canvas_override=NORMAL_DIR / meta["normal"],
                                caption_override=" " if caption == "none" else None, **GEN)
                shutil.copy(gc.EXP / "anomaly/imgs/easy" / f"{rid}.png", dst)
        if last:
            print(f"  generate seed {s} {last[1]}: done", flush=True)


def stage_refine(seeds: list[int], steps: int) -> None:
    from compute_refined_masks import _load_dinov3_cnx
    from refine_inprocess import refine
    dino = _load_dinov3_cnx("cuda")
    for s in seeds:
        man = json.loads((OUT / "manifest" / f"seed_{s}.json").read_text())
        for fold, cls, st in iter_steps(man, steps):
            for arm in ("in", "cross"):
                d = sub(f"diffusion_{arm}", s, fold, cls); img = d / "images" / f"{st['i']:03d}.png"; dst = d / "masks" / img.name
                if dst.exists() or not img.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                # matched to the B image: decoded canvas inside the band by the blend weight a, raw outside
                m = refine(dino, img, sub("prep", s, fold, cls) / img.stem, None, factor=MASK_FACTOR, thesis_reference=True)
                Image.fromarray(m.astype(np.uint8) * 255).save(dst)
        print(f"  refine seed {s}: done", flush=True)


def stage_rawbg(seeds: list[int], steps: int) -> None:
    """B pipeline (thesis as written): diffusion_{arm} (decoded background, as generated) -> diffusion_{arm}_raw.
    image: new = gen + (1-a)(raw - dec(raw)) = a*G + (1-a)*raw, a = the generator's band-2 blend map
    (make_background_variants.alpha_512), raw = prep canvas.png at 512 (= the JPG the generator painted on), dec = seeded
    VAE latent SAMPLE round-trip (the generator's own draw is not reproducible -> residual ~0.5/255, only inside the
    band; outside the inpainting region the image is set to the raw canvas exactly).
    Aborts a seed if the blend map does not match the generation (ring just outside the band > 3/255).
    mask: refined MASK_FACTOR x p90 against the MATCHED blended reference (refine thesis_reference=True: decoded inside the
    band by a, raw outside) - the originals in diffusion_{arm} stay untouched."""
    raise SystemExit("rawbg: not ported to the per-fold layout (2026-10-01); unused - images are generated as B directly")


def stage_check(seeds: list[int]) -> None:
    """Per seed: pool = 54 distinct canvases (no train / test normal); every epoch the 6 types' groups partition the 54
    exactly (9 each); per type: every anomaly used exactly 180/n times, 9 different anomalies per epoch, step canvas =
    its epoch group; prep matches the manifest; all 4 methods have image + mask."""
    train, test, pool = normal_pools(); bad = 0
    refs = json.loads((OUT / "references.json").read_text())
    for s in seeds:
        man = json.loads((OUT / "manifest" / f"seed_{s}.json").read_text()); P = man["pool"]
        if len(set(P)) != POOL_CANVASES or set(P) & (set(train) | set(test)):
            print(f"  seed {s}: pool not {POOL_CANVASES} distinct / contains a train-test normal!"); bad += 1
        for e, grp in enumerate(man["epochs"]):
            if sorted(c for t_ in CLASSES for c in grp[t_]) != sorted(P) or any(len(grp[t_]) != PER_CLASS_CANVASES for t_ in CLASSES):
                print(f"  seed {s} epoch {e}: groups do not partition the pool"); bad += 1
        for cls in CLASSES:
            stp = man["steps"][cls]; n = len(refs[cls]); use = Counter(st["ref"] for st in stp)
            if len(use) != n or set(use.values()) != {STEPS // n}:
                print(f"  seed {s} {cls}: reference use {dict(use)}"); bad += 1
            for e in range(len(man["epochs"])):
                blk = stp[e * PER_CLASS_CANVASES:(e + 1) * PER_CLASS_CANVASES]
                if len({x["ref"] for x in blk}) != len(blk) or sorted(x["canvas"] for x in blk) != sorted(man["epochs"][e][cls]):
                    print(f"  seed {s} {cls} epoch {e}: repeated anomaly or wrong canvases"); bad += 1
            for st in stp:
                meta = json.loads((sub("prep", s, None, cls) / f"{st['i']:03d}" / "meta.json").read_text())
                if meta["ref"] != st["ref"] or meta["normal"] != f"{st['canvas']}.JPG":
                    print(f"  seed {s} {cls} {st['i']}: prep/manifest mismatch"); bad += 1
                for meth in ("diffusion_in", "diffusion_cross", "dtd", "cutmix"):
                    for sd in ("images", "masks"):
                        if not (sub(meth, s, None, cls) / sd / f"{st['i']:03d}.png").exists():
                            bad += 1
        print(f"  seed {s}: checked", flush=True)
    print(f"check: {bad} problem(s)")


def stage_folds() -> None:
    """Leave-one-type-out test folds (seed-independent): fold k trains on the synthetics of the OTHER 5 types and is
    tested on the fixed 50 test normals + the REAL anomalies of type k. Type k's synthetics (the only ones imitating
    type k's real anomalies) are not used in fold k, so no test anomaly is imitated in its own fold's training data."""
    train, test, _ = normal_pools(); refs = json.loads((OUT / "references.json").read_text())
    folds = {}
    for k in CLASSES:
        folds[k] = {"train_types": [c for c in CLASSES if c != k],
                    "test_normals": [f"{x}.JPG" for x in test],
                    "test_anomalies": [{"id": r["id"], "raw": r["raw"], "image": r["ind_img"], "gt_mask": r["gt_mask"]}
                                       for r in refs[k]]}
    out = {"normals_dir": str(NORMAL_DIR), "train_normals": [f"{x}.JPG" for x in train], "test_normals": [f"{x}.JPG" for x in test],
           "rule": "fold k: train = 100 train normals + synthetics of train_types (per seed: <method>/seed_s/fold_k/<type>/; "
                   "the 45 canvases of normals/seed_s/m1_canvases.txt are re-partitioned over the 5 train types every epoch); "
                   "test = test_normals + test_anomalies (real VisA cashew anomalies of type k)",
           "folds": folds}
    json.dump(out, open(OUT / "folds.json", "w"), indent=1)
    for k in CLASSES:
        print(f"  fold {k:12s}: train types {len(folds[k]['train_types'])}, test = {len(test)} normals + {len(folds[k]['test_anomalies'])} real {k}")


DTD_DIR = ROOT / "datasets" / "dtd" / "images"


def method_seed(s: int, cls: str, i: int, tag: str) -> int:
    """Per-step seed for the non-diffusion methods (independent stream per method, deterministic)."""
    return int(hashlib.sha256(f"{s}|{cls}|{i}|{tag}".encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def _load_step(prep: Path) -> tuple[np.ndarray, np.ndarray]:
    """Canvas (512 RGB, LANCZOS) + footprint M (512, max-pooled, bool): the same inputs the diffusion arms use."""
    from sheet_utils import mask_resize
    x = np.array(Image.open(prep / "canvas.png").convert("RGB").resize((512, 512), Image.LANCZOS)).astype(np.float32)
    M = mask_resize(np.array(Image.open(prep / "placed_mask.png").convert("L")) > 127, 512)
    return x, M


def _indist_mask(s: int, cls: str, i: int) -> np.ndarray:
    """The in-distribution diffusion arm's REFINED mask for this step (512 bool). DTD and CutMix paste into exactly this
    region (user 2026-09-27: same supervision region as diffusion_in -> less confounding than the placed footprint).
    Requires stage refine for this seed to have run first (run_seeds.sh order: generate -> refine -> dtd -> cutmix)."""
    p = OUT / "diffusion_in" / f"seed_{s}" / cls / "masks" / f"{i:03d}.png"
    if not p.exists():
        raise SystemExit(f"missing in-dist refined mask {p} - run generate + refine for seed {s} first")
    return np.array(Image.open(p).convert("L")) > 127


def _draem_augmenter(seed: int):
    """DRAEM texture augmentation (same list as RealNet realnet_dataset.rand_augment): 3 of 10, no replacement."""
    import imgaug.augmenters as iaa
    augmenters = [
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
        iaa.pillike.EnhanceSharpness(),
        iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
        iaa.Solarize(0.5, threshold=(32, 128)),
        iaa.Posterize(),
        iaa.Invert(),
        iaa.pillike.Autocontrast(),
        iaa.pillike.Equalize(),
        iaa.Affine(rotate=(-45, 45)),
    ]
    idx = np.random.default_rng(seed).choice(len(augmenters), 3, replace=False)
    aug = iaa.Sequential([augmenters[j] for j in idx])
    aug.seed_(seed)
    return aug, [int(j) for j in idx]


def stage_dtd(seeds: list[int], steps: int, oracle: bool = False) -> None:
    """DRAEM default: x_syn = (1-M) x + M [(1-b) x + b t], t = augmented DTD texture, b ~ U(0.2, 1.0); M = the PLACED GT
    mask of the step (also the label). oracle=True (method 'dtd_oracle'): the texture is drawn from the step anomaly's
    top-20 most similar DTD textures (dtd_oracle_textures.json); b, augmenters and seed are IDENTICAL to 'dtd'."""
    files = sorted(str(p) for p in DTD_DIR.glob("*/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert len(files) > 5000, f"DTD not found / incomplete at {DTD_DIR} ({len(files)} files)"
    top = json.loads((OUT / "dtd_oracle_textures.json").read_text()) if oracle else None
    name = "dtd_oracle" if oracle else "dtd"
    for s in seeds:
        man = json.loads((OUT / "manifest" / f"seed_{s}.json").read_text()); log = {}
        for fold, cls, st in iter_steps(man, steps):
            d = sub(name, s, fold, cls); i = st["i"]; dst = d / "images" / f"{i:03d}.png"
            sd = method_seed(s, cls, i, "dtd"); rng = np.random.default_rng(sd)
            tex = files[int(rng.integers(len(files)))]; beta = float(rng.uniform(0.2, 1.0))
            if oracle:
                cand = top[st["ref"]]
                tex = str(DTD_DIR / cand[int(np.random.default_rng(method_seed(s, cls, i, "dtd_oracle")).integers(len(cand)))]["texture"])
            aug, idx = _draem_augmenter(sd)
            log[f"{cls}/{i:03d}"] = {"texture": str(Path(tex).relative_to(DTD_DIR)), "augmenters": idx, "beta": beta, "seed": sd}
            if dst.exists():
                continue
            x, M = _load_step(sub("prep", s, fold, cls) / f"{i:03d}")   # M = placed GT mask (= label)
            t = np.array(Image.open(tex).convert("RGB").resize((512, 512), Image.LANCZOS))
            t = aug(image=t).astype(np.float32)
            m = M[..., None].astype(np.float32)
            y = (1 - m) * x + m * ((1 - beta) * x + beta * t)
            dst.parent.mkdir(parents=True, exist_ok=True); (d / "masks").mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.clip(y, 0, 255).round().astype(np.uint8)).save(dst)
            Image.fromarray(M.astype(np.uint8) * 255).save(d / "masks" / f"{i:03d}.png")
        print(f"  {name} seed {s}: done", flush=True)
        (OUT / name / f"seed_{s}").mkdir(parents=True, exist_ok=True)
        json.dump(log, open(OUT / name / f"seed_{s}" / "choices.json", "w"), indent=1)


def stage_cutmix(seeds: list[int], steps: int) -> None:
    """x_syn = (1-M) x + M shift(x_donor). Donor = a seeded random canvas of the seed's fixed 45-pool, excluding the
    step's own canvas (not restricted to the type's epoch group; user 2026-10-01). Shift: align the two cashews' centres
    (FG centroids), then take the nearest shift that puts M completely inside the donor's cashew (FG) -> pasted pixels
    are always cashew surface, from the corresponding spot on the nut. Label = the placed GT mask."""
    from scipy.signal import fftconvolve
    from sheet_utils import mask_resize
    for s in seeds:
        man = json.loads((OUT / "manifest" / f"seed_{s}.json").read_text()); log = {"steps": {}}; P = man["pool"]
        for fold, cls, st in iter_steps(man, steps):
            d = sub("cutmix", s, fold, cls); i = st["i"]; dst = d / "images" / f"{i:03d}.png"
            others = [c for c in P if c != st["canvas"]]
            donor = others[int(np.random.default_rng(method_seed(s, cls, i, "cutmix-donor")).integers(len(others)))]
            x, M = _load_step(sub("prep", s, fold, cls) / f"{i:03d}")   # M = placed GT mask (= label)
            ys, xs = np.nonzero(M); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            Mb = M[y0:y1, x0:x1].astype(np.float32)
            fg = mask_resize(np.array(Image.open(FG_DIR / f"{donor}_binary.png").convert("L")) > 127, 512).astype(np.float32)
            cnt = fftconvolve(fg, Mb[::-1, ::-1], mode="valid")                  # M pixels on donor cashew, per top-left
            valid = np.argwhere(cnt >= Mb.sum() - 0.5)
            fg_i = mask_resize(np.array(Image.open(FG_DIR / f"{st['canvas']}_binary.png").convert("L")) > 127, 512)
            dy, dx = np.argwhere(fg > 0.5).mean(0) - np.argwhere(fg_i).mean(0)
            ty, tx = y0 + dy, x0 + dx
            if len(valid):
                Y, X = valid[int(np.argmin((valid[:, 0] - ty) ** 2 + (valid[:, 1] - tx) ** 2))]; full = True
            else:
                Y, X = np.unravel_index(int(cnt.argmax()), cnt.shape); full = False
            log["steps"][f"{cls}/{i:03d}"] = {"donor": donor, "patch_top_left": [int(Y), int(X)], "fits_inside_donor_cashew": full}
            if dst.exists():
                continue
            xd = np.array(Image.open(NORMAL_DIR / f"{donor}.JPG").convert("RGB").resize((512, 512), Image.LANCZOS)).astype(np.float32)
            y = x.copy(); sub_ = y[y0:y1, x0:x1]; pat = xd[Y:Y + (y1 - y0), X:X + (x1 - x0)]
            mb = Mb[..., None]; y[y0:y1, x0:x1] = (1 - mb) * sub_ + mb * pat
            dst.parent.mkdir(parents=True, exist_ok=True); (d / "masks").mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.clip(y, 0, 255).round().astype(np.uint8)).save(dst)
            Image.fromarray(M.astype(np.uint8) * 255).save(d / "masks" / f"{i:03d}.png")
        print(f"  cutmix seed {s}: done", flush=True)
        (OUT / "cutmix" / f"seed_{s}").mkdir(parents=True, exist_ok=True)
        json.dump(log, open(OUT / "cutmix" / f"seed_{s}" / "choices.json", "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "folds", "generate", "refine", "rawbg", "dtd", "dtd_oracle", "cutmix", "check"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--caption", choices=["template", "none"], default="template")
    a = ap.parse_args()
    if a.stage == "plan":
        stage_plan(a.seeds, STEPS)                   # always the full 180-step schedule (steps only limit generation)
    elif a.stage == "generate":
        stage_generate(a.seeds, a.steps, a.classes.split(","), a.caption)
    elif a.stage == "refine":
        stage_refine(a.seeds, a.steps)
    elif a.stage == "rawbg":
        stage_rawbg(a.seeds, a.steps)
    elif a.stage == "folds":
        stage_folds()
    elif a.stage == "dtd":
        stage_dtd(a.seeds, a.steps)
    elif a.stage == "dtd_oracle":
        stage_dtd(a.seeds, a.steps, oracle=True)
    elif a.stage == "cutmix":
        stage_cutmix(a.seeds, a.steps)
    else:
        stage_check(a.seeds)


if __name__ == "__main__":
    main()

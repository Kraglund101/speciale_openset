#!/usr/bin/env python3
"""Open-set (leave-one-type-out) experiment, five-stage design (user 2026-10-03). Root: results/open_set_v2.

Per seed s: 54 canvases (spare normals; no train / test normal) in 6 fixed groups G_t of 9, one per anomaly type; a
BUDGET of 3 real anomalies per type (balanced over the 10 seeds: anomaly_plan_b3.json); matched cross-object donors
1-to-1 (pregenerate_synthetic.pairing). Fold k trains on the 5 other types: 45 synthetics per epoch on its 45 canvases
P_k = union of G_t, t != k (9 per type), 20 epochs; test = 50 test normals + the REAL anomalies of type k.
  stage 1  the 3 budget anomalies of each type (3 uses each) on the type's 9 canvases, ONE set repeated every epoch
  stage 2  same anomalies, same canvases, re-rendered every epoch (new placement + noise, re-paired within the group)
  stage 3  same anomalies, re-rendered, on ANY of the fold's 45 canvases (re-dealt among its 5 types every epoch)
  stage 4  ALL cashew anomalies of the type (9-20), donors matched 1-to-1, canvases as stage 3 (DTD / CutMix use their masks)
  stage 5  cross-object only: stage 4's canvases, masks and seeds, donor drawn from the UNRESTRICTED pool (ext_pools)
Image sets ("tags"): s2 (stages 1-2; shared by all folds; stage 1 = its epoch 0); s3, s4 (one deal of the 54 canvases
per epoch, shared by the folds) + s3_fold_k / s4_fold_k: the ~1 in 6 synthetics of a deal that sit on a canvas of G_k
are re-painted for fold k on the free canvases of P_k -> every fold sees exactly its own 45 canvases, each once per
epoch; s5 / s5_fold_k = s4's placements with the stage-5 donor. s3 epoch 0 is stage 1's set (alias of s2).
Methods: m1 (100 train normals + P_k clean); diffusion_in, diffusion_cross, dtd, cutmix at stages 1-4 (dtd / cutmix at
stage 4 added 2026-10-09: the budget also sets the MASKS they fill); diffusion_cross at stage 5. CutMix is generated per fold (donor = another canvas of P_k).
Model: train_uninet.py --hero caao (lambda 0.2, no SG) / m1, no photometric, fixed test shifts.

  python scripts/open_set_steps.py plan     --seeds 42 [--folds holes breakage]
  python scripts/open_set_steps.py generate --seeds 42 [--folds ..] [--shard 0 2]      (diffusion_in + diffusion_cross)
  python scripts/open_set_steps.py refine   --seeds 42 [--folds ..] [--shard 0 2]
  python scripts/open_set_steps.py dtd | cutmix --seeds 42 [--folds ..]
  python scripts/open_set_steps.py check    --seeds 42 [--folds ..]
  python scripts/open_set_steps.py command  --seed 42 --fold holes --method diffusion_in --step 2   (prints the run command)
  python scripts/open_set_steps.py arms                                                             (lists method/stage arms)
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import pregenerate_synthetic as P  # noqa: E402

# Generator setting: pregenerate_synthetic.GEN (since 2026-10-04: noise strength 0.4, 50 even denoising steps = the
# noise-strength test arm ns_040_s50; the sampler lives in src/inference/generate.py, EvenStepDDIM).

OUT = ROOT / "results/open_set_v2"
CLASSES, K, E, BUDGET = P.CLASSES, 9, 20, 3
assert K % BUDGET == 0
SEEDS10 = [42, 123, 7, 99, 256, 11, 22, 33, 44, 55]
CASHEW = P.gc.CASHEW_ROOT
ARMS = [("m1", 0)] + [(m, st) for st in (1, 2, 3, 4) for m in ("diffusion_in", "diffusion_cross", "dtd", "cutmix")] \
    + [("diffusion_cross", 5)]
SET_OF_STEP = {1: "s2", 2: "s2", 3: "s3", 4: "s4", 5: "s4"}
# stage-5 donor pools: the matched pools + exactly the additions the user accepted on 2026-10-03
APPROVED = {"scratches": ["fryum", "macaroni1", "macaroni2"], "colour_same": ["macaroni1", "macaroni2"], "colour_diff": ["macaroni2"],
            "burnt": ["fryum"]}
POOL_SIZE = {"holes": 130, "breakage": 32, "scratches": 56, "colour_same": 64, "colour_diff": 22, "burnt": 24}
_EXT: dict | None = None
# OPEN_SET_EPOCH_LIMIT=n (smoke tests): only the first n epochs are placed / generated / trained; the manifest itself
# is always the full 20-epoch plan, so a smoke test uses the first epochs of the real data.
EPOCH_LIMIT = int(os.environ.get("OPEN_SET_EPOCH_LIMIT", E))
# OPEN_SET_SETS="s2" (or "s2,s3"): restrict plan / generate / refine / dtd / cutmix to these image sets (partial pilots).
ONLY_SETS = tuple(x for x in os.environ.get("OPEN_SET_SETS", "s2,s3,s4").split(",") if x)
# OPEN_SET_HERO=mode3: train the synthetic arms with train_uninet.py --hero mode3 instead of caao; runs then go to
# runs_mode3/ (never mixed with the caao runs in runs/). m1 always uses --hero m1.
HERO = os.environ.get("OPEN_SET_HERO", "caao")
RUNS_SUB = "runs" if HERO == "caao" else f"runs_{HERO}"
# OPEN_SET_TRAIN_EPOCHS=n (2026-10-15, e.g. 50): train n epochs instead of E; runs go to <RUNS_SUB>_e<n>/ so the 20-epoch runs
# are never touched. The 20 per-epoch synthetic pools are cycled by the runner (EpochPoolSampler: pool epoch % 20).
TRAIN_EPOCHS = int(os.environ.get("OPEN_SET_TRAIN_EPOCHS", E))
if TRAIN_EPOCHS != E:
    RUNS_SUB += f"_e{TRAIN_EPOCHS}"
# OPEN_SET_RUNS_TAG=<tag> (2026-10-21, e.g. "profile"): runs go to <RUNS_SUB>_<tag>/ - for extra seeds / tests that must not
# enter runs/ (open_set_tables.py reads every seed found in runs/).
RUNS_TAG = os.environ.get("OPEN_SET_RUNS_TAG", "")
if RUNS_TAG:
    RUNS_SUB += f"_{RUNS_TAG}"


def rebase(p) -> str:
    """Make an absolute path written on another machine valid here (everything lives under the project root)."""
    p = str(p)
    if Path(p).exists():
        return p
    q = p.replace("\\", "/"); j = q.lower().find("/speciale/")
    return str(ROOT / q[j + len("/speciale/"):]) if j >= 0 else p


def load_refs() -> dict:
    refs = P.load_refs()
    for v in refs.values():
        for r in v:
            for k in ("ind_img", "gt_mask", "ind_mask"):
                r[k] = rebase(r[k])
    return refs


def phys(x: dict) -> str:
    """Physical-defect key of a donor image (Real-IAD photographs one defect from up to 5 cameras)."""
    return x["donor"] + ":" + re.sub(r"_C\d_.*$", "", Path(x["cr_img"]).stem)


def ext_pools() -> dict[str, dict[str, list[dict]]]:
    """type -> {physical defect: [donor views]} for stage 5."""
    global _EXT
    if _EXT is None:
        import data_sources as DC                                       # label maps of the VisA products
        base = P.donor_pools(); _EXT = {}
        for t in CLASSES:
            g: dict[str, list[dict]] = {}
            for x in base[t]:
                g.setdefault(phys(x), []).append({k: x[k] for k in ("donor", "donor_id", "cr_img", "cr_mask")})
            for obj in APPROVED.get(t, []):
                for img, msk, _lab in DC.visa_rows(obj).get(t, []):
                    name = f"{obj}_{img.stem}"; m = P.drop_isolated_specks(P.h.normalise_mask(msk, name), name)
                    g[f"{obj}:{img.stem}"] = [{"donor": obj, "donor_id": img.stem, "cr_img": str(img), "cr_mask": str(m)}]
            assert len(g) == POOL_SIZE[t], (t, len(g), POOL_SIZE[t])
            _EXT[t] = g
    return _EXT


# ---------------------------------------------------------------- plan
def anomaly_plan(refs: dict) -> dict:
    """Which BUDGET real anomalies of each type every seed uses: every anomaly used as equally often as possible over the
    10 seeds (fewest-used first, random tie-break) -> all real anomalies are used, none dominates. Written once."""
    f = ROOT / "data" / f"anomaly_plan_b{BUDGET}.json"
    if f.exists():
        return json.loads(f.read_text())
    rng = np.random.default_rng(2026); plan = {str(s): {} for s in SEEDS10}; use = {}
    for cls in CLASSES:
        ids = [r["id"] for r in refs[cls]]; cnt = {a: 0 for a in ids}
        for s in SEEDS10:
            pick = sorted(ids, key=lambda a: (cnt[a], rng.random()))[:BUDGET]
            for a in pick:
                cnt[a] += 1
            plan[str(s)][cls] = sorted(pick)
        use[cls] = f"{min(cnt.values())}-{max(cnt.values())}"
    OUT.mkdir(parents=True, exist_ok=True); json.dump(plan, open(f, "w"), indent=1)
    print(f"anomaly plan (budget {BUDGET}): uses per real anomaly over the 10 seeds:", use)
    return plan


def extra_budget(s: int, refs: dict) -> dict:
    """Budget anomalies for a seed outside the 10-seed plan (2026-10-21, e.g. the profiling seed 66): same rule as
    anomaly_plan - fewest-used first over the 10 planned seeds, random tie-break (rng seeded by s) - stored in
    anomaly_plan_b<BUDGET>_extra.json; the 10-seed plan file is never changed."""
    f = OUT / f"anomaly_plan_b{BUDGET}_extra.json"; extra = json.loads(f.read_text()) if f.exists() else {}
    if str(s) not in extra:
        plan = anomaly_plan(refs); rng = np.random.default_rng([2026, s]); extra[str(s)] = {}
        for cls in CLASSES:
            cnt = {r["id"]: 0 for r in refs[cls]}
            for p_ in plan.values():
                for a in p_[cls]:
                    cnt[a] += 1
            extra[str(s)][cls] = sorted(sorted(cnt, key=lambda a: (cnt[a], rng.random()))[:BUDGET])
        json.dump(extra, open(f, "w"), indent=1)
    return extra[str(s)]


def build_manifest(s: int, refs: dict, pool: list[str]) -> dict:
    rng = np.random.default_rng([s, 1]); R = K // BUDGET
    canv = [str(x) for x in rng.choice(pool, size=len(CLASSES) * K, replace=False)]
    groups = {t: sorted(canv[g * K:(g + 1) * K]) for g, t in enumerate(CLASSES)}
    budget = anomaly_plan(refs).get(str(s)) or extra_budget(s, refs)
    s2 = {}                                                            # same anomalies, same canvases, re-paired per epoch
    for t in CLASSES:
        r2 = np.random.default_rng(P.method_seed(s, t, -1, "s2")); A, G, steps = budget[t], groups[t], []
        for e in range(E):
            if e % BUDGET == 0:                                        # canvas c -> anomaly (lab[c] + e) % BUDGET: every
                lab = r2.permutation(np.repeat(np.arange(BUDGET), R))  # canvas gets every anomaly once per BUDGET epochs
            for jj, c in enumerate(r2.permutation(K)):
                steps.append({"i": e * K + jj, "epoch": e, "ref": A[(int(lab[c]) + e) % BUDGET], "canvas": G[int(c)]})
        s2[t] = steps

    def dealt(name: str, per_epoch: dict) -> tuple[dict, dict]:
        """One deal of the 54 canvases per epoch (shared by the folds) + the per-fold re-paints onto P_k."""
        r = np.random.default_rng(P.method_seed(s, name, -1, "deal")); shared = {t: [] for t in CLASSES}; D = []
        for e in range(E):
            if name == "s3" and e == 0:                                # stages 1-3 start from the same images
                for t in CLASSES:
                    shared[t] += [{**st, "alias": "s2"} for st in s2[t][:K]]
                D.append({t: list(groups[t]) for t in CLASSES}); continue
            perm = [canv[j] for j in r.permutation(len(canv))]; d = {t: perm[g * K:(g + 1) * K] for g, t in enumerate(CLASSES)}; D.append(d)
            for t in CLASSES:
                o = r.permutation(K)
                shared[t] += [{"i": e * K + jj, "epoch": e, "ref": per_epoch[t][e][int(o[jj])], "canvas": d[t][jj]} for jj in range(K)]
        rep = {}
        for k in CLASSES:
            rk = np.random.default_rng(P.method_seed(s, f"{name}/{k}", -1, "repaint")); Gk = set(groups[k]); over = {t: {} for t in CLASSES if t != k}
            for e in range(E):
                free = [c for c in D[e][k] if c not in Gk]; free = [free[j] for j in rk.permutation(len(free))]
                vac = [(t, st) for t in CLASSES if t != k for st in shared[t][e * K:(e + 1) * K] if st["canvas"] in Gk]
                assert len(vac) == len(free), (name, k, e, len(vac), len(free))
                for (t, st), c in zip(vac, free):
                    over[t][str(st["i"])] = c
            rep[k] = over
        return shared, rep

    s3, rep3 = dealt("s3", {t: [list(budget[t]) * R] * E for t in CLASSES})
    all4 = {}
    for t in CLASSES:
        ids = [r["id"] for r in refs[t]]; n = len(ids); assert (E * K) % n == 0
        all4[t] = P.epoch_refs(ids, E * K // n, E, K, np.random.default_rng(P.method_seed(s, t, -1, "s4refs")))
    s4, rep4 = dealt("s4", all4)
    pools5 = ext_pools(); s5 = {}                                      # stage 5: donor per s4 step, every donor ~equally often
    for t in CLASSES:
        r5 = np.random.default_rng(P.method_seed(s, t, -1, "s5donors")); keys = sorted(pools5[t]); cnt = {k: 0 for k in keys}; lst = []
        for e in range(E):
            pick = sorted(keys, key=lambda k: (cnt[k], r5.random()))[:K]
            for k in pick:
                cnt[k] += 1
            for j in r5.permutation(K):
                views = pools5[t][pick[int(j)]]; lst.append(views[int(r5.integers(len(views)))])
        s5[t] = lst
    return {"seed": s, "budget_size": BUDGET, "pool": sorted(canv), "groups": groups, "budget": budget, "pairing": P.pairing(s, refs),
            "sets": {"s2": s2, "s3": s3, "s4": s4}, "repaint": {"s3": rep3, "s4": rep4}, "s5_donor": s5}


def manifest(s: int) -> dict:
    return json.loads((OUT / "manifest" / f"seed_{s}.json").read_text())


def entries(man: dict, folds: list[str] | None, sets: tuple = ("s2", "s3", "s4")):
    """(tag, type, i, ref, canvas) of every distinct placement (aliases skipped; re-paints only for `folds`)."""
    sets = tuple(x for x in sets if x in ONLY_SETS)
    for name in sets:
        for t in CLASSES:
            for st in man["sets"][name][t]:
                if "alias" not in st and st["epoch"] < EPOCH_LIMIT:
                    yield name, t, st["i"], st["ref"], st["canvas"]
    for name in sets:
        for k, over in man["repaint"].get(name, {}).items():
            if folds and k not in folds:
                continue
            for t, d in over.items():
                by = {st["i"]: st for st in man["sets"][name][t]}
                for i, c in d.items():
                    if by[int(i)]["epoch"] < EPOCH_LIMIT:
                        yield f"{name}_fold_{k}", t, int(i), by[int(i)]["ref"], c


def resolve(man: dict, name: str, fold: str, t: str, st: dict) -> tuple[str, str, str]:
    """(image-set tag, canvas, set) of step `st` of type t as fold `fold` uses it."""
    if "alias" in st:
        return "s2", st["canvas"], "s2"
    over = man["repaint"].get(name, {}).get(fold, {}).get(t, {})
    return (f"{name}_fold_{fold}", over[str(st["i"])], name) if str(st["i"]) in over else (name, st["canvas"], name)


def pdir(s: int, tag: str, cls: str, i: int) -> Path:
    return OUT / "prep" / f"seed_{s}" / tag / cls / f"{i:03d}"


def ipath(arm: str, s: int, tag: str, cls: str, i: int, kind: str = "images") -> Path:
    return OUT / arm / f"seed_{s}" / tag / cls / kind / f"{i:03d}.png"


def cmpath(s: int, fold: str, name: str, cls: str, i: int, kind: str = "images") -> Path:
    return OUT / "cutmix" / f"seed_{s}" / f"fold_{fold}" / name / cls / kind / f"{i:03d}.png"


def diffusion_jobs(man: dict, byid: dict, tag: str, cls: str, i: int, ref: str) -> list[tuple[str, str, str, str]]:
    """(arm, image tag, reference image, reference mask) to render for one placement."""
    r = byid[ref]; dn = man["pairing"][cls][ref]
    jobs = [("diffusion_in", tag, r["ind_img"], r["ind_mask"]), ("diffusion_cross", tag, dn["cr_img"], dn["cr_mask"])]
    if tag.startswith("s4"):                                           # stage 5: same placement, unrestricted donor
        d5 = man["s5_donor"][cls][i]; jobs.append(("diffusion_cross", tag.replace("s4", "s5", 1), d5["cr_img"], d5["cr_mask"]))
    return jobs


def stage_plan(seeds: list[int], folds: list[str] | None) -> None:
    train, test, pool = P.normal_pools(); refs = load_refs(); byid = {r["id"]: r for k in refs for r in refs[k]}
    (OUT / "manifest").mkdir(parents=True, exist_ok=True)
    for s in seeds:
        f = OUT / "manifest" / f"seed_{s}.json"
        man = manifest(s) if f.exists() else build_manifest(s, refs, pool)
        assert man.get("budget_size") == BUDGET, f"seed {s}: manifest was planned with another budget"
        if not f.exists():
            json.dump(man, open(f, "w"), indent=1)
        n = 0
        for tag, cls, i, ref, canvas in entries(man, folds):
            work = pdir(s, tag, cls, i); n += 1
            if (work / "meta.json").exists():
                continue
            work.mkdir(parents=True, exist_ok=True)
            pl = P.place(byid[ref], canvas, s, tag, cls, i, work)
            P.cached_link(lambda: Image.open(P.NORMAL_DIR / f"{canvas}.JPG").convert("RGB"), f"canvas_{canvas}", work / "canvas.png")
            json.dump({"idx": i, "normal": f"{canvas}.JPG", "is_hard": False, "seed": pl["seed"], "cls": cls, "ref": ref, "tag": tag,
                       "placement": pl}, open(work / "meta.json", "w"), indent=1)
        print(f"plan seed {s}: {n} placements (folds {folds or 'all'})", flush=True)


# ---------------------------------------------------------------- images
def generator_stamp() -> None:
    """Never mix images of different generator settings: the settings the existing diffusion images were made with are
    kept in OUT/generator_settings.json; a mismatch (or images without a stamp) stops generation."""
    cur = {"ckpt": P.CKPT, "mask_factor": P.MASK_FACTOR, **P.GEN}; f = OUT / "generator_settings.json"
    have = any(d.is_dir() and any(d.rglob("*.png")) for d in (OUT / "diffusion_in", OUT / "diffusion_cross"))
    if f.exists():
        old = json.loads(f.read_text())
        if old != cur:
            raise SystemExit(f"STOP: {OUT} holds images made with {old}, current settings are {cur}. Move the old "
                             f"diffusion_in / diffusion_cross / runs folders away (e.g. to _superseded_...) and delete {f.name}.")
    elif have and os.environ.get("OPEN_SET_ACCEPT_EXISTING") != "1":   # =1: I know the existing images were made with cur
        raise SystemExit(f"STOP: {OUT} holds diffusion images without {f.name} (made before 2026-10-04, noise strength 0.7). "
                         f"Move diffusion_in / diffusion_cross / runs away before generating with {cur} "
                         f"(or, if they WERE made with exactly these settings, set OPEN_SET_ACCEPT_EXISTING=1 once).")
    else:
        OUT.mkdir(parents=True, exist_ok=True); tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cur, indent=1)); os.replace(tmp, f)       # atomic: parallel shards start together


def stage_generate(seeds: list[int], folds: list[str] | None, shard: tuple[int, int]) -> None:
    refs = load_refs(); byid = {r["id"]: r for k in refs for r in refs[k]}
    generator_stamp()
    P.gc.setup_experiment("ResNet")
    pipe, ip, t2i = P.gc.load_models(P.h.CKPTS[P.CKPT]); P.gc.EXP = Path(tempfile.mkdtemp(prefix="os2gen_"))
    gen = P.GEN
    print(f"generator: {gen}", flush=True)
    for s in seeds:
        man = manifest(s)
        for n, (tag, cls, i, ref, canvas) in enumerate(entries(man, folds)):
            if n % shard[1] != shard[0]:
                continue
            prep = pdir(s, tag, cls, i); seed = None
            for arm, itag, img, mask in diffusion_jobs(man, byid, tag, cls, i, ref):
                dst = ipath(arm, s, itag, cls, i)
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True); seed = seed or json.loads((prep / "meta.json").read_text())["seed"]
                rid = f"{arm[10:]}-s{s}-{itag}-{cls}-{i:03d}"
                P.gc.generate_one(pipe, ip, t2i, canvas_id=canvas, ref_id=rid, difficulty="easy", defect_map={rid: "defect"}, seed=seed,
                                  device="cuda", layout="A", ref_img_override=Path(img), ref_mask_override=Path(mask),
                                  placed_mask_override=prep / "placed_mask.png", save_raw=True,
                                  canvas_override=P.NORMAL_DIR / f"{canvas}.JPG", caption_override=" ", **gen)
                shutil.copy(P.gc.EXP / "anomaly/imgs/easy" / f"{rid}.png", dst)
            if n % 500 == 0:
                print(f"  generate seed {s} shard {shard[0]}/{shard[1]}: entry {n}", flush=True)
        print(f"generate seed {s} shard {shard[0]}/{shard[1]}: done", flush=True)


def stage_refine(seeds: list[int], folds: list[str] | None, shard: tuple[int, int]) -> None:
    from compute_refined_masks import _load_dinov3_cnx
    from refine_inprocess import refine
    refs = load_refs(); byid = {r["id"]: r for k in refs for r in refs[k]}; dino = None
    for s in seeds:
        man = manifest(s)
        for n, (tag, cls, i, ref, canvas) in enumerate(entries(man, folds)):
            if n % shard[1] != shard[0]:
                continue
            for arm, itag, _img, _mask in diffusion_jobs(man, byid, tag, cls, i, ref):
                img, dst = ipath(arm, s, itag, cls, i), ipath(arm, s, itag, cls, i, "masks")
                if dst.exists() or not img.exists():
                    continue
                dino = dino or _load_dinov3_cnx("cuda"); dst.parent.mkdir(parents=True, exist_ok=True)
                m = refine(dino, img, pdir(s, tag, cls, i), None, factor=P.MASK_FACTOR, thesis_reference=True)
                Image.fromarray(m.astype(np.uint8) * 255).save(dst)
        print(f"refine seed {s} shard {shard[0]}/{shard[1]}: done", flush=True)


def stage_dtd(seeds: list[int], folds: list[str] | None, shard: tuple[int, int] = (0, 1)) -> None:
    """DRAEM blend into the placed GT mask (= label); stages 1-4 (sets s2, s3, s4)."""
    files = sorted(str(p) for p in P.DTD_DIR.glob("*/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert len(files) > 5000, f"DTD incomplete at {P.DTD_DIR}"
    for s in seeds:
        man = manifest(s)
        for n, (tag, cls, i, ref, canvas) in enumerate(entries(man, folds, sets=("s2", "s3", "s4"))):
            dst = ipath("dtd", s, tag, cls, i)
            if n % shard[1] != shard[0] or dst.exists():       # each item is seeded on its own: sharding changes nothing
                continue
            sd = P.method_seed(s, f"{tag}/{cls}", i, "dtd"); rng = np.random.default_rng(sd)
            tex = files[int(rng.integers(len(files)))]; beta = float(rng.uniform(0.2, 1.0)); aug, _ = P._draem_augmenter(sd)
            x, M = P._load_step(pdir(s, tag, cls, i))
            t = aug(image=np.array(Image.open(tex).convert("RGB").resize((512, 512), Image.LANCZOS))).astype(np.float32)
            m = M[..., None].astype(np.float32); y = (1 - m) * x + m * ((1 - beta) * x + beta * t)
            dst.parent.mkdir(parents=True, exist_ok=True); md = ipath("dtd", s, tag, cls, i, "masks"); md.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.clip(y, 0, 255).round().astype(np.uint8)).save(dst); Image.fromarray(M.astype(np.uint8) * 255).save(md)
        print(f"dtd seed {s} shard {shard[0]}/{shard[1]}: done", flush=True)


def stage_cutmix(seeds: list[int], folds: list[str] | None, shard: tuple[int, int] = (0, 1)) -> None:
    """Per fold: patch from another canvas of the fold's 45 (report: 'resampled from the fixed target-domain canvas pool,
    excluding the current canvas'), cut at the centroid-aligned position nearest to 'fully on the donor cashew', pasted
    into the placed GT mask (= label). Stages 1-4 (sets s2, s3, s4)."""
    from scipy.signal import fftconvolve
    from sheet_utils import mask_resize
    fgc: dict[str, np.ndarray] = {}

    def fg(c: str) -> np.ndarray:
        if c not in fgc:
            fgc[c] = mask_resize(np.array(Image.open(P.FG_DIR / f"{c}_binary.png").convert("L")) > 127, 512)
        return fgc[c]

    n = -1
    for s in seeds:
        man = manifest(s)
        for fold in (folds or CLASSES):
            types = [t for t in CLASSES if t != fold]; Pk = sorted(c for t in types for c in man["groups"][t])
            for name in (x for x in ("s2", "s3", "s4") if x in ONLY_SETS):
                for t in types:
                    for st in man["sets"][name][t]:
                        if "alias" in st or st["epoch"] >= EPOCH_LIMIT:
                            continue
                        i = st["i"]; dst = cmpath(s, fold, name, t, i); n += 1
                        if n % shard[1] != shard[0] or dst.exists():   # each item is seeded on its own: sharding changes nothing
                            continue
                        tag, canvas, _ = resolve(man, name, fold, t, st); others = [c for c in Pk if c != canvas]
                        donor = others[int(np.random.default_rng(P.method_seed(s, f"{fold}/{name}/{t}", i, "cutmix-donor")).integers(len(others)))]
                        x, M = P._load_step(pdir(s, tag, t, i))
                        ys, xs = np.nonzero(M); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                        Mb = M[y0:y1, x0:x1].astype(np.float32); fd = fg(donor).astype(np.float32)
                        cnt = fftconvolve(fd, Mb[::-1, ::-1], mode="valid"); valid = np.argwhere(cnt >= Mb.sum() - 0.5)
                        dy, dx = np.argwhere(fd > 0.5).mean(0) - np.argwhere(fg(canvas)).mean(0); ty, tx = y0 + dy, x0 + dx
                        Y, X = valid[int(np.argmin((valid[:, 0] - ty) ** 2 + (valid[:, 1] - tx) ** 2))] if len(valid) \
                            else np.unravel_index(int(cnt.argmax()), cnt.shape)
                        xd = np.array(Image.open(P.NORMAL_DIR / f"{donor}.JPG").convert("RGB").resize((512, 512), Image.LANCZOS)).astype(np.float32)
                        y = x.copy(); mb = Mb[..., None]; y[y0:y1, x0:x1] = (1 - mb) * y[y0:y1, x0:x1] + mb * xd[Y:Y + (y1 - y0), X:X + (x1 - x0)]
                        dst.parent.mkdir(parents=True, exist_ok=True); md = cmpath(s, fold, name, t, i, "masks"); md.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(np.clip(y, 0, 255).round().astype(np.uint8)).save(dst); Image.fromarray(M.astype(np.uint8) * 255).save(md)
            print(f"cutmix seed {s} fold {fold}: done", flush=True)


# ---------------------------------------------------------------- per-run lists
def train_normals(s: int, man: dict) -> list[str]:
    """100 train normals of the seed: drawn from the non-test normals that are not one of the seed's 54 canvases."""
    sp = json.loads(P.SPLITS.read_text())["normals"]
    allN = sorted(p.name for p in P.NORMAL_DIR.glob("*.JPG")); banned = set(sp["test"]) | {f"{c}.JPG" for c in man["pool"]}
    return sorted(str(x) for x in np.random.default_rng([s, 77]).choice([n for n in allN if n not in banned], 100, replace=False))


def fold_epochs(man: dict, s: int, fold: str, method: str, step: int) -> list[list[dict]]:
    """The 20 epoch pools (45 synthetics each) of one run."""
    types = [t for t in CLASSES if t != fold]; name = SET_OF_STEP[step]; Pk = sorted(c for t in types for c in man["groups"][t])
    assert (method, step) in ARMS, (method, step)
    epochs = []
    for e in range(EPOCH_LIMIT):
        ep, canv = [], []
        for t in types:
            e0 = 0 if step == 1 else e
            for st in man["sets"][name][t][e0 * K:(e0 + 1) * K]:
                tag, c, setn = resolve(man, name, fold, t, st); i = st["i"]
                if method == "cutmix":
                    img, msk = cmpath(s, fold, setn, t, i), cmpath(s, fold, setn, t, i, "masks")
                else:
                    itag = tag.replace("s4", "s5", 1) if step == 5 else tag
                    img, msk = ipath(method, s, itag, t, i), ipath(method, s, itag, t, i, "masks")
                ep.append({"image": str(img), "mask": str(msk), "canvas": str(pdir(s, tag, t, i) / "canvas.png")}); canv.append(c)
                assert step >= 4 or st["ref"] in man["budget"][t], (t, st["ref"])
        assert sorted(canv) == Pk, f"seed {s} fold {fold} stage {step} epoch {e}: canvases are not exactly the fold's 45"
        epochs.append(ep)
    return epochs


def run_lists(s: int, fold: str, method: str, step: int) -> dict:
    man = manifest(s); d = OUT / "lists" / f"seed_{s}" / f"fold_{fold}"; d.mkdir(parents=True, exist_ok=True)
    sp = json.loads(P.SPLITS.read_text())["normals"]; tn = train_normals(s, man)
    Pk = sorted(f"{c}.JPG" for t in CLASSES if t != fold for c in man["groups"][t])
    assert not (set(tn) | set(Pk)) & set(sp["test"]) and not set(tn) & set(Pk) and len(Pk) == 45
    test = [{"id": r["id"], "raw": r["raw"], "image": r["ind_img"], "gt_mask": r["gt_mask"]} for r in load_refs()[fold]]
    (d / "train_normals.txt").write_text("\n".join(tn) + "\n"); (d / "m1_extra.txt").write_text("\n".join(Pk) + "\n")
    json.dump(test, open(d / "test_anomalies.json", "w"), indent=1)
    out = {"dir": d, "train_normals": d / "train_normals.txt", "m1_extra": d / "m1_extra.txt", "test": d / "test_anomalies.json", "n_test": len(test)}
    if method != "m1":
        ep = fold_epochs(man, s, fold, method, step); out["epochs"] = d / f"epochs_{method}_s{step}.json"
        out["missing"] = sum(not Path(t[k]).exists() for e in ep for t in e for k in ("image", "mask", "canvas"))
        json.dump({"epochs": ep}, open(out["epochs"], "w"))
    return out


def run_name(method: str, step: int) -> str:
    return "m1" if method == "m1" else f"{method}_s{step}"


def command(s: int, fold: str, method: str, step: int, extra: list[str]) -> list[str]:
    r = run_lists(s, fold, method, step)
    cmd = [sys.executable, "-u", str(ROOT / "scripts/train_uninet.py"), "--hero", "m1" if method == "m1" else HERO, "--no-photometric",
           "--output-dir", str(OUT / RUNS_SUB / run_name(method, step) / f"seed_{s}" / f"fold_{fold}"), "--seed", str(s), "--",
           "--train-normal-list", str(r["train_normals"]), "--test-anomaly-list", str(r["test"]), "--fg-anomaly-dir", str(CASHEW / "birefnet_masks_anomaly")]
    cmd += ["--baseline-extra-list", str(r["m1_extra"])] if method == "m1" else ["--synthetic-epochs", str(r["epochs"])]
    if EPOCH_LIMIT < E:
        cmd += ["--epochs", str(EPOCH_LIMIT)]
    elif TRAIN_EPOCHS != E:
        cmd += ["--epochs", str(TRAIN_EPOCHS)]
    return cmd + extra


def stage_check(seeds: list[int], folds: list[str] | None) -> None:
    train, test, pool = P.normal_pools(); refs = load_refs(); bad = 0; R = K // BUDGET
    for s in seeds:
        man = manifest(s); canv = man["pool"]; want = lambda t: Counter({a: R for a in man["budget"][t]})
        ok = len(set(canv)) == 54 and not set(canv) & (set(train) | set(test)) and sorted(c for g in man["groups"].values() for c in g) == sorted(canv)
        ok &= all(len(man["budget"][t]) == BUDGET and set(man["budget"][t]) <= {r["id"] for r in refs[t]} for t in CLASSES)
        for t in CLASSES:
            s2, s3, s4, d5 = man["sets"]["s2"][t], man["sets"]["s3"][t], man["sets"]["s4"][t], man["s5_donor"][t]; n = len(refs[t])
            ok &= all(Counter(st["ref"] for st in blk[e * K:(e + 1) * K]) == want(t) for blk in (s2, s3) for e in range(E))
            ok &= all(sorted(st["canvas"] for st in s2[e * K:(e + 1) * K]) == man["groups"][t] for e in range(E))
            u4 = Counter(st["ref"] for st in s4); ok &= len(u4) == n and set(u4.values()) == {E * K // n}
            ok &= [(st["ref"], st["canvas"]) for st in s3[:K]] == [(st["ref"], st["canvas"]) for st in s2[:K]]
            u5 = Counter(phys(d) for d in d5); ok &= len(d5) == E * K and max(u5.values()) - min(u5.values()) <= 1 or len(u5) < POOL_SIZE[t]
            ok &= all(len({phys(d) for d in d5[e * K:(e + 1) * K]}) == K for e in range(E))
        print(f"seed {s}: manifest structure {'OK' if ok else 'PROBLEM'} | stage-5 distinct donor defects per type: "
              + str({t: len({phys(d) for d in man['s5_donor'][t]}) for t in CLASSES})); bad += not ok
        miss = 0
        for fold in (folds or CLASSES):
            for method, step in ARMS:
                miss += run_lists(s, fold, method, step).get("missing", 0)
        print(f"seed {s}: {sum(1 for _ in entries(man, folds))} distinct placements; per-fold canvas / budget checks passed for "
              f"folds {folds or 'all'}; files still missing over all arms: {miss}"); bad += miss > 0
    print(f"check: {bad} problem(s)")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["plan", "generate", "refine", "dtd", "cutmix", "check", "command", "arms"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42]); ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1]); ap.add_argument("--seed", type=int); ap.add_argument("--fold")
    ap.add_argument("--method"); ap.add_argument("--step", type=int, default=0)
    a, extra = ap.parse_known_args(); sh = (a.shard[0], a.shard[1])
    if a.stage == "plan":
        stage_plan(a.seeds, a.folds)
    elif a.stage == "generate":
        stage_generate(a.seeds, a.folds, sh)
    elif a.stage == "refine":
        stage_refine(a.seeds, a.folds, sh)
    elif a.stage == "dtd":
        stage_dtd(a.seeds, a.folds, sh)
    elif a.stage == "cutmix":
        stage_cutmix(a.seeds, a.folds, sh)
    elif a.stage == "check":
        stage_check(a.seeds, a.folds)
    elif a.stage == "arms":
        print("\n".join(f"{m} {st}" for m, st in ARMS))
    else:
        print(" ".join(f'"{c}"' if " " in c else c for c in command(a.seed, a.fold, a.method, a.step, [x for x in extra if x != "--"])))


if __name__ == "__main__":
    main()

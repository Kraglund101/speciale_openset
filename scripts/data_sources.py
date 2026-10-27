"""Dataset locations, the generator checkpoint and the small mask / donor-listing helpers the open-set pipeline uses.

Collected verbatim from the scripts they were written in (hole_donor_probe.py, class_calibration.py,
masked_patch_pairing_final.py, donor_candidates_sheet.py), which belong to earlier experiments and are not part of
this repository. Behaviour is unchanged: the 10 seed manifests rebuild identically.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CASHEW_ROOT = ROOT / "anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew"
CASHEW_MASKS = CASHEW_ROOT / "Data" / "Masks" / "Anomaly"
REST = ROOT / "anomverse_extension/datasets/VisA_validation_dataset/restVisA/VisA_20220922"   # VisA donor objects
DATA = D = ROOT / "anomverse_extension/datasets/full_training_dataset"                         # Real-IAD + captions
REAL_ANOMALIES = ROOT / "data/real_anomalies.json"     # the 84 single-label real cashew anomalies: id, VisA label, type
CKPTS = {"clean-20k": ROOT / "results/generator_nocashew_20k_constlr/checkpoint_20000"}       # trained generator
OUT = ROOT / "results/open_set_data"                    # set by the caller; normalised masks go to OUT/ref_masks


# ---------------------------------------------------------------- masks
def binary(path: Path) -> np.ndarray:
    """Any non-zero pixel is mask: VisA GT masks store 1/2, our placed masks store 255."""
    return (np.array(Image.open(path).convert("L")) > 0).astype(np.float32)


def normalise_mask(src: Path, name: str) -> Path:
    """Write a 0/255 copy of a donor mask (VisA stores 1/2/5; downstream loaders threshold at 0.5).

    `name` MUST be unique per donor IMAGE (Real-IAD photographs one sample from several cameras, each a separate
    image + mask). Refuses to overwrite an existing file with different content.
    """
    d = OUT / "ref_masks"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{name}.png"
    arr = (np.array(Image.open(src).convert("L")) > 0).astype(np.uint8) * 255
    if arr.max() == 0:
        raise ValueError(f"donor mask {src} is empty")
    if out.exists():                                 # never rewrite: parallel processes read these files
        if not np.array_equal(np.array(Image.open(out).convert("L")), arr):
            raise RuntimeError(f"{out} already holds a DIFFERENT mask - two donors share the name '{name}'")
        return out
    tmp = d / f"{name}.{os.getpid()}.tmp.png"; Image.fromarray(arr).save(tmp); os.replace(tmp, out)
    return out


def resize_mask(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to (h, w): MAX-POOL when shrinking (never lose a small component), NEAREST when
    enlarging (nothing can be lost)."""
    import torch
    h, w = shape
    if h <= m.shape[0] and w <= m.shape[1]:
        t = torch.from_numpy(m.astype(np.float32))[None, None]
        return torch.nn.functional.adaptive_max_pool2d(t, (h, w))[0, 0].numpy() > 0.5
    return np.array(Image.fromarray((m.astype(np.uint8) * 255)).resize((w, h), Image.NEAREST)) > 127


# ---------------------------------------------------------------- donor listings (configurations 1-4)
CANON = {
    "small holes": "holes",
    "corner or edge breakage": "breakage", "corner and edge breakage": "breakage",
    "middle breakage": "breakage", "small cracks": "breakage",
    "small scratches": "scratches", "burnt": "burnt",
    "same colour spot": "colour_same", "similar colour spot": "colour_same",
    "different colour spot": "colour_diff",
    "stuck together": "stuck", "fryum stuck together": "stuck", "other": "other",
}
DROP = {"stuck", "other"}


def canon(l): return {CANON.get(a.strip().lower(), "?") for a in l.split(",") if a.strip()}
def n_raw(l): return len([a for a in l.split(",") if a.strip()])


def anno(p):
    m = {}
    with open(p) as fh:
        for r in csv.reader(fh):
            if len(r) > 1 and "Anomaly" in r[0]: m[Path(r[0]).stem] = r[1]
    return m


def visa_items(obj, img_dirs, mask_dir, annop, exclude=()):
    """Single-label anomalies of a VisA object (labels mapped by CANON)."""
    out = []
    for i, raw in sorted(anno(annop).items()):
        c = canon(raw)
        if (c & DROP) or not c or n_raw(raw) > 1: continue
        cls = sorted(c)[0]
        if cls in exclude: continue
        img = next((dd / f"{i}{e}" for dd in img_dirs for e in (".JPG", ".jpg", ".png")
                    if (dd / f"{i}{e}").exists()), None)
        mask = next((mask_dir / f"{i}{e}" for e in (".png", ".PNG")
                     if (mask_dir / f"{i}{e}").exists()), None)
        if img and mask:
            out.append({"src": obj, "id": i, "img": img, "mask": mask, "raw": raw, "cls": cls})
    return out


def train_items(src, cls, pred):
    """Entries of the generator training set (master_training.json) matching pred; caller sets _master and _caps."""
    caps = train_items._caps
    out = []
    for m in train_items._master:
        if not pred(m): continue
        mp = m.get("mask_path")
        if not mp or not (D / mp).exists(): continue
        out.append({"src": src, "id": Path(m["image_path"]).stem, "img": D / m["image_path"],
                    "mask": D / mp, "raw": f"{src}/{m.get('defect_type')}", "cls": cls,
                    "cap": caps.get(m["image_path"], "")})
    return out


# ---------------------------------------------------------------- donor listings (configuration 5)
ROWS_CANON = {"small holes": "holes", "corner or edge breakage": "breakage", "corner and edge breakage": "breakage", "middle breakage": "breakage",
              "small cracks": "breakage", "small scratches": "scratches", "burnt": "burnt", "same colour spot": "colour_same",
              "similar colour spot": "colour_same", "different colour spot": "colour_diff"}
# labels outside ROWS_CANON that were accepted for the unrestricted pool (marked with *)
EXTRA = {"color spot similar to the object": "colour_same", "different color spot": "colour_diff", "small chip around edge": "breakage",
         "chip around edge and corner": "breakage", "chunk of gum missing": "breakage", "corner missing": "breakage", "scratches": "scratches",
         "breakage down the middle": "breakage"}


def visa_rows(obj: str) -> dict[str, list[tuple[Path, Path, str]]]:
    """type -> [(image, mask, label)] of the single-label anomalies of a VisA object."""
    out: dict[str, list] = {}
    for r in csv.reader(open(REST / obj / "image_anno.csv")):
        if len(r) < 2 or "Anomaly" not in r[0]:
            continue
        labs = [a.strip().lower() for a in r[1].split(",") if a.strip()]
        if len(labs) != 1:
            continue
        cls = ROWS_CANON.get(labs[0]) or EXTRA.get(labs[0])
        stem = Path(r[0]).stem; img = REST / obj / "Data/Images/Anomaly" / f"{stem}.JPG"; msk = REST / obj / "Data/Masks/Anomaly" / f"{stem}.png"
        if cls and img.exists() and msk.exists():
            out.setdefault(cls, []).append((img, msk, labs[0] + ("" if labs[0] in ROWS_CANON else "*")))
    return out

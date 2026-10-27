"""Mask placement utilities for validation data preparation.

Extracted from viz_mask_placement.py (easy) and redo_hard_placement.py (hard).
Grouping runs on the RAW anomaly mask (no roundtrip): placement is pre-
generation, so no band dilation exists yet, and grouping raw keeps components
maximally split → more placement diversity.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import label as ndimage_label



def _rotate_scale_mask(crop: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Rotate + scale a binary footprint using ONLY the operations of the CLIP/training pipeline
    (src/data/anomaly_dataset.py _clip_augmentation_transforms + crop_utils): rotation in exact 90-degree steps
    (np.rot90, no resampling) and MAX-POOL when shrinking. Enlarging uses NEAREST, which only duplicates pixels.
    The footprint therefore has exactly the shape CLIP was shown, never a resampled one: arbitrary-angle NEAREST
    rotation + NEAREST downscale split thin holes into pieces (cashew 062: 2 pieces -> 3; 135 of 1000 transforms).
    `angle` must be a multiple of 90. Returns float32 {0, 1}.
    """
    import torch
    import torch.nn.functional as F
    if angle % 90 != 0:
        raise ValueError(f"footprint rotation must be a multiple of 90 degrees (CLIP pipeline), got {angle}")
    rot = np.rot90(crop > 0.5, k=int(angle // 90) % 4)
    h, w = rot.shape
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    if new_h < h or new_w < w:
        t = torch.from_numpy(np.ascontiguousarray(rot).astype(np.float32))[None, None]
        out = F.adaptive_max_pool2d(t, (new_h, new_w))[0, 0].numpy() > 0.5
    else:
        out = np.array(Image.fromarray(rot.astype(np.uint8) * 255).resize((new_w, new_h), Image.NEAREST)) > 127
    return out.astype(np.float32)


def load_binary_mask(path: Path) -> np.ndarray:
    """Load grayscale mask, binarize at 0.5. Returns float32 {0.0, 1.0}."""
    mask = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return (mask > 0.5).astype(np.float32)


def _place_single_component(
    crop: np.ndarray,
    foreground_mask: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    max_attempts: int = 500,
    scale_range: Tuple[float, float] = (0.5, 1.0),
    record: Optional[list] = None,
) -> Optional[np.ndarray]:
    """Place a single bbox-cropped component inside FG, avoiding occupied pixels.
    record: if given, the transform of the successful placement is appended (flip_v, flip_h, angle, scale, top, left).

    Returns placed mask [H, W] float32 {0, 1}, or None.
    """
    H, W = foreground_mask.shape

    # Available FG = foreground minus already-occupied pixels
    available = foreground_mask * (1.0 - occupied)
    fg_ys, fg_xs = np.where(available > 0.5)
    if len(fg_ys) == 0:
        return None

    # Random flips
    flip_v = rng.random() < 0.5
    if flip_v:
        crop = crop[::-1, :].copy()
    flip_h = rng.random() < 0.5
    if flip_h:
        crop = crop[:, ::-1].copy()

    # Rotation in 90-degree steps only (CLIP pipeline). Same single uniform draw as before -> quadrant, so the RNG
    # stream (scale, position) is unchanged.
    angle = 90 * int(rng.uniform(0, 360) // 90)
    # Uniform scale (drawn in the same order as before, so the RNG stream is unchanged)
    scale = rng.uniform(*scale_range)
    scaled_arr = _rotate_scale_mask(crop, angle, scale)
    if scaled_arr.sum() == 0:
        return None

    sh, sw = scaled_arr.shape
    for _ in range(max_attempts):
        idx = rng.randint(0, len(fg_ys) - 1)
        cy, cx = fg_ys[idx], fg_xs[idx]
        top, left = cy - sh // 2, cx - sw // 2
        if top < 0 or left < 0 or top + sh > H or left + sw > W:
            continue
        placed = np.zeros((H, W), dtype=np.float32)
        placed[top : top + sh, left : left + sw] = scaled_arr
        # ALL anomaly pixels must be within foreground
        if placed.sum() != (placed * foreground_mask).sum():
            continue
        # No overlap with already-placed components
        if (placed * occupied).sum() > 0:
            continue
        if record is not None:
            record.append({"flip_v": flip_v, "flip_h": flip_h, "angle": angle, "scale": scale, "top": int(top),
                           "left": int(left), "shape": (int(sh), int(sw))})
        return placed
    return None


def _get_connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """Return list of bbox-cropped binary arrays, one per connected component."""
    labeled, n_components = ndimage_label(mask > 0.5)
    components = []
    for i in range(1, n_components + 1):
        comp = (labeled == i).astype(np.float32)
        if comp.sum() < 5:  # skip single-pixel noise
            continue
        ys, xs = np.where(comp > 0.5)
        crop = comp[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        components.append(crop)
    # Sort largest first (most constrained → place first)
    components.sort(key=lambda c: c.sum(), reverse=True)
    return components


# ── Spatial grouping helpers (mirrors crop_utils 3-pass algorithm, no torch) ─


def _merge_bboxes(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """Union of two (y0, y1, x0, x1) bboxes."""
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


def _pad_bbox(
    bbox: Tuple[int, int, int, int], frac: float, H: int, W: int,
) -> Tuple[int, int, int, int]:
    y0, y1, x0, x1 = bbox
    s = max(y1 - y0 + 1, x1 - x0 + 1)
    p = max(1, int(s * frac))
    return (max(0, y0 - p), min(H - 1, y1 + p), max(0, x0 - p), min(W - 1, x1 + p))


def _bbox_max_side(bbox: Tuple[int, int, int, int]) -> int:
    return max(bbox[1] - bbox[0] + 1, bbox[3] - bbox[2] + 1)


def _crop_region_bbox(
    bbox: Tuple[int, int, int, int],
    pad_frac: float,
    max_size: int,
    H: int,
    W: int,
) -> Tuple[int, int, int, int]:
    """Square crop region of at least max_size, centered on padded bbox."""
    padded = _pad_bbox(bbox, pad_frac, H, W)
    crop_s = max(_bbox_max_side(padded), max_size)
    cy = (padded[0] + padded[1]) / 2.0
    cx = (padded[2] + padded[3]) / 2.0
    ry0 = int(round(cy - (crop_s - 1) / 2.0))
    ry1 = ry0 + crop_s - 1
    rx0 = int(round(cx - (crop_s - 1) / 2.0))
    rx1 = rx0 + crop_s - 1
    if ry0 < 0:
        ry1 -= ry0; ry0 = 0
    if ry1 >= H:
        ry0 -= (ry1 - H + 1); ry1 = H - 1
    if rx0 < 0:
        rx1 -= rx0; rx0 = 0
    if rx1 >= W:
        rx0 -= (rx1 - W + 1); rx1 = W - 1
    return (max(0, ry0), ry1, max(0, rx0), rx1)


def _group_components(
    components: List[Dict],
    H: int,
    W: int,
    max_group_size: int = 224,
    pad_frac: float = 0.0,
    overlap_merge: bool = True,
) -> List[List[int]]:
    """3-pass spatial grouping (same algorithm as crop_utils.get_component_groups).

    Args:
        overlap_merge: If True (default), Pass 3 merges groups whose crop
            regions share at least one pixel. If False, Pass 3 is skipped —
            groups only merge if their bboxes unite within max_group_size
            (passes 1/1b) or one's bbox fits inside another's crop region
            (pass 2). Used for placement to keep more components separate.

    Returns list of lists of component indices.
    """
    # Pass 1: greedy merge by y-center
    sorted_idxs = sorted(
        range(len(components)),
        key=lambda i: (components[i]['bbox'][0] + components[i]['bbox'][1]) / 2,
    )

    groups: List[Dict] = []
    for idx in sorted_idxs:
        bbox = components[idx]['bbox']
        placed = False
        for g in groups:
            candidate = _merge_bboxes(g['bb'], bbox)
            if _bbox_max_side(_pad_bbox(candidate, pad_frac, H, W)) <= max_group_size:
                g['idxs'].append(idx)
                g['bb'] = candidate
                placed = True
                break
        if not placed:
            groups.append({'idxs': [idx], 'bb': bbox})

    # Pass 1b: exhaustive pairwise merge
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(len(groups) - 1, i, -1):
                candidate = _merge_bboxes(groups[i]['bb'], groups[j]['bb'])
                if _bbox_max_side(_pad_bbox(candidate, pad_frac, H, W)) <= max_group_size:
                    groups[i]['idxs'].extend(groups[j]['idxs'])
                    groups[i]['bb'] = candidate
                    groups.pop(j)
                    changed = True
            if changed:
                break

    # Pass 2: containment — absorb if bbox fits inside other's crop region
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            ri = _crop_region_bbox(groups[i]['bb'], pad_frac, max_group_size, H, W)
            for j in range(i + 1, len(groups)):
                rj = _crop_region_bbox(groups[j]['bb'], pad_frac, max_group_size, H, W)
                bb_i, bb_j = groups[i]['bb'], groups[j]['bb']
                # j inside i's crop region?
                if bb_j[0] >= ri[0] and bb_j[1] <= ri[1] and bb_j[2] >= ri[2] and bb_j[3] <= ri[3]:
                    groups[i]['idxs'].extend(groups[j]['idxs'])
                    groups[i]['bb'] = _merge_bboxes(bb_i, bb_j)
                    groups.pop(j)
                    changed = True
                    break
                # i inside j's crop region?
                if bb_i[0] >= rj[0] and bb_i[1] <= rj[1] and bb_i[2] >= rj[2] and bb_i[3] <= rj[3]:
                    groups[j]['idxs'].extend(groups[i]['idxs'])
                    groups[j]['bb'] = _merge_bboxes(bb_i, bb_j)
                    groups.pop(i)
                    changed = True
                    break
            if changed:
                break

    # Pass 3: overlap — merge if crop regions share any pixel
    if overlap_merge:
        changed = True
        while changed:
            changed = False
            for i in range(len(groups)):
                ri = _crop_region_bbox(groups[i]['bb'], pad_frac, max_group_size, H, W)
                for j in range(i + 1, len(groups)):
                    rj = _crop_region_bbox(groups[j]['bb'], pad_frac, max_group_size, H, W)
                    if ri[0] <= rj[1] and ri[1] >= rj[0] and ri[2] <= rj[3] and ri[3] >= rj[2]:
                        groups[i]['idxs'].extend(groups[j]['idxs'])
                        groups[i]['bb'] = _merge_bboxes(groups[i]['bb'], groups[j]['bb'])
                        groups.pop(j)
                        changed = True
                        break
                if changed:
                    break

    return [g['idxs'] for g in groups]


def _get_grouped_crops(
    mask: np.ndarray,
    max_group_size: int = 224,
    pad_frac: float = 0.0,
    overlap_merge: bool = True,
    return_bboxes: bool = False,
):
    """Get bbox-cropped masks for spatially grouped connected components.
    return_bboxes=True returns (crops, bboxes) with bbox = (y0, y1, x0, x1) inclusive in source coordinates.

    Nearby components are merged into single crops preserving their relative
    spatial positions.  Uses the same 3-pass algorithm as
    ``crop_utils.get_component_groups``.

    Args:
        overlap_merge: Toggle Pass 3 (crop-region 1-pixel overlap merge).
            See ``_group_components`` for details.

    Returns list of bbox-cropped binary arrays, sorted largest-area first.
    """
    H, W = mask.shape
    labeled, n_components = ndimage_label(mask > 0.5)

    components: List[Dict] = []
    for c in range(1, n_components + 1):
        comp_mask = (labeled == c).astype(np.float32)
        if comp_mask.sum() < 5:
            continue
        ys, xs = np.where(comp_mask > 0.5)
        bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        components.append({'mask': comp_mask, 'bbox': bbox})

    if not components:
        return ([], []) if return_bboxes else []

    if len(components) == 1:
        m = components[0]['mask']
        ys, xs = np.where(m > 0.5)
        crop = [m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]]
        return (crop, [(int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))]) if return_bboxes else crop

    group_indices = _group_components(
        components, H, W, max_group_size, pad_frac, overlap_merge=overlap_merge,
    )

    crops: List[np.ndarray] = []
    bboxes: List[Tuple[int, int, int, int]] = []
    for idxs in group_indices:
        combined = np.zeros((H, W), dtype=np.float32)
        for i in idxs:
            combined = np.maximum(combined, components[i]['mask'])
        ys, xs = np.where(combined > 0.5)
        crop = combined[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        crops.append(crop)
        bboxes.append((int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())))

    order = sorted(range(len(crops)), key=lambda k: crops[k].sum(), reverse=True)   # stable, largest area first
    crops = [crops[k] for k in order]
    if return_bboxes:
        return crops, [bboxes[k] for k in order]
    return crops


def place_aligned_mask(
    anomaly_mask: np.ndarray,
    foreground_mask: np.ndarray,
    max_search_radius: int = 200,
    min_inside_frac: float = 1.0,
) -> Optional[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
    """Place anomaly mask at the same position as the original, preserving spatial layout.

    All components are placed as a single unit (no random transforms). The function
    tries to match the original centroid position on the target canvas, searching
    outward if the exact position is invalid.

    Args:
        anomaly_mask: [H, W] float32 binary mask at its ORIGINAL position.
        foreground_mask: [H, W] float32 binary mask of the target canvas foreground.
        max_search_radius: Max pixel distance to search from the original centroid.
        min_inside_frac: Minimum fraction of anomaly pixels that must be inside
            foreground (default 1.0 = 100%). Lower values (e.g. 0.9) allow
            edge-hugging anomalies that can never be fully contained.

    Returns:
        (placed_mask, target_centroid, actual_centroid) or None.
        placed_mask: [H, W] float32 {0, 1}.
        target_centroid: (cy, cx) — original centroid in the anomaly mask.
        actual_centroid: (cy, cx) — centroid of the placed mask on the canvas.
    """
    H, W = foreground_mask.shape
    ys, xs = np.where(anomaly_mask > 0.5)
    if len(ys) == 0:
        return None

    # Extract the anomaly block (bounding box crop, preserving relative positions)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    block = anomaly_mask[y0:y1, x0:x1]
    bh, bw = block.shape

    # Original centroid
    cy_orig = int(round(ys.mean()))
    cx_orig = int(round(xs.mean()))
    # Offset from block top-left to centroid
    off_y = cy_orig - y0
    off_x = cx_orig - x0

    def _try_place(cy: int, cx: int) -> Optional[np.ndarray]:
        """Try placing the block so its centroid lands at (cy, cx)."""
        top = cy - off_y
        left = cx - off_x
        if top < 0 or left < 0 or top + bh > H or left + bw > W:
            return None
        placed = np.zeros((H, W), dtype=np.float32)
        placed[top:top + bh, left:left + bw] = block
        inside = (placed * foreground_mask).sum()
        if inside < min_inside_frac * placed.sum():
            return None
        return placed

    # Try exact original position first
    result = _try_place(cy_orig, cx_orig)
    if result is not None:
        return result, (cy_orig, cx_orig), (cy_orig, cx_orig)

    # Spiral outward: search FG pixels by increasing distance from original centroid
    fg_ys, fg_xs = np.where(foreground_mask > 0.5)
    if len(fg_ys) == 0:
        return None

    # Compute distances from original centroid to all FG pixels
    dists = np.sqrt((fg_ys - cy_orig) ** 2 + (fg_xs - cx_orig) ** 2)
    order = np.argsort(dists)

    for i in order:
        if dists[i] > max_search_radius:
            break
        cy_try, cx_try = int(fg_ys[i]), int(fg_xs[i])
        result = _try_place(cy_try, cx_try)
        if result is not None:
            # Compute actual centroid of the placed mask
            p_ys, p_xs = np.where(result > 0.5)
            actual_cy = int(round(p_ys.mean()))
            actual_cx = int(round(p_xs.mean()))
            return result, (cy_orig, cx_orig), (actual_cy, actual_cx)

    return None


def place_easy_mask(
    anomaly_mask: np.ndarray,
    foreground_mask: np.ndarray,
    max_attempts: int = 500,
    scale_range: Tuple[float, float] = (0.5, 1.0),
    seed: int | None = None,
    max_group_size: int = 224,
    overlap_merge: bool = False,
    record: Optional[list] = None,
) -> Optional[np.ndarray]:
    """Place anomaly mask so ALL anomaly pixels are inside foreground.
    record: optional list; each placed group appends its source bbox + transform (to paste the real pixels identically).

    Spatially nearby connected components are grouped (same 3-pass algorithm
    as CLIP crop grouping) and transformed as a single unit, preserving their
    relative positions.  Distant groups are placed independently.

    Steps per group: bbox-crop → random flips → random rotation →
    random scale → pick a random FG pixel as centre, check 100% coverage
    + no overlap with other groups.

    Args:
        anomaly_mask: [H, W] float32 binary mask of the anomaly.
        foreground_mask: [H, W] float32 binary mask of the valid foreground.
        max_attempts: Max random placement attempts per group.
        scale_range: (lo, hi) uniform scale factor.
        seed: Optional seed for local RNG (deterministic placement).
        max_group_size: Max padded bbox side for merging nearby components
            into a single group (default 224, matching CLIP crop grouping).
        overlap_merge: If True, merges groups whose 224×224 crop regions
            share even one pixel (Pass 3 of the 3-pass algorithm). Default
            False for placement — keeps more components separate for
            greater placement diversity.

    Returns:
        Placed mask [H, W] float32 {0, 1}, or None if no valid position found.
    """
    rng = random.Random(seed)
    H, W = foreground_mask.shape

    ys, xs = np.where(anomaly_mask > 0.5)
    if len(ys) == 0:
        return None

    fg_ys, fg_xs = np.where(foreground_mask > 0.5)
    if len(fg_ys) == 0:
        return None

    group_crops, group_bboxes = _get_grouped_crops(
        anomaly_mask, max_group_size=max_group_size, overlap_merge=overlap_merge, return_bboxes=True,
    )
    rec = [] if record is not None else None

    def _done(result):
        if record is not None and result is not None:
            for g, b in zip(rec, group_bboxes):
                g["src_bbox"] = b
            record.extend(rec)
        return result

    if len(group_crops) == 1:
        # Single group: original fast path (no overhead)
        crop = group_crops[0]
        return _done(_place_single_component(
            crop, foreground_mask,
            occupied=np.zeros((H, W), dtype=np.float32),
            rng=rng, max_attempts=max_attempts, scale_range=scale_range, record=rec,
        ))

    # Multiple groups: place each independently
    occupied = np.zeros((H, W), dtype=np.float32)
    combined = np.zeros((H, W), dtype=np.float32)

    for group_crop in group_crops:
        placed = _place_single_component(
            group_crop, foreground_mask, occupied,
            rng=rng, max_attempts=max_attempts, scale_range=scale_range, record=rec,
        )
        if placed is None:
            return None  # failed to place this group
        combined = np.maximum(combined, placed)
        occupied = np.maximum(occupied, placed)

    return _done(combined)


def place_hard_mask(
    ref_mask: np.ndarray,
    canvas_fg_dict: Dict[str, np.ndarray],
    scale_range: Tuple[float, float] = (0.8, 1.2),
    used: Set[str] | None = None,
    seed: int | None = None,
) -> Optional[Tuple[str, np.ndarray, float, float, float]]:
    """Place mask so entire FG is inside the placed mask (100% coverage).

    Exhaustive search: 4 flip combos × 4 right-angle rotations (CLIP pipeline) × progressive scales.

    Args:
        ref_mask: [H, W] float32 binary mask to place.
        canvas_fg_dict: {canvas_id: fg_mask} dict of candidate canvases.
        scale_range: (lo, hi) initial scale range, widens progressively.
        used: Canvas IDs already used (skip them).
        seed: Optional seed for deterministic ordering.

    Returns:
        (canvas_id, placed_mask, angle, scale, coverage%) or None.
    """
    rng = random.Random(seed)
    if used is None:
        used = set()

    ys, xs = np.where(ref_mask > 0.5)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop_raw = ref_mask[y0:y1, x0:x1]

    # Precompute canvas FG centres (skip used)
    fg_info: dict = {}
    for cid, fg in canvas_fg_dict.items():
        if cid in used:
            continue
        fg_ys, fg_xs = np.where(fg > 0.5)
        if len(fg_ys) == 0:
            continue
        fg_info[cid] = (fg, fg.shape, fg_ys.mean(), fg_xs.mean(), fg.sum())

    angles = [0, 90, 180, 270]          # 90-degree steps only (CLIP pipeline); was 72 continuous angles
    flips = [(False, False), (True, False), (False, True), (True, True)]

    max_scale_bump = 5  # up to +0.5 above original max
    for bump in range(max_scale_bump + 1):
        lo = scale_range[0]
        hi = scale_range[1] + bump * 0.1
        scales = list(np.linspace(lo, hi, max(5, int((hi - lo) / 0.05) + 1)))
        rng.shuffle(angles)
        rng.shuffle(scales)

        for fv, fh in flips:
            crop = crop_raw.copy()
            if fv:
                crop = crop[::-1, :].copy()
            if fh:
                crop = crop[:, ::-1].copy()

            for angle in angles:
                for scale_try in scales:
                    scaled_arr = _rotate_scale_mask(crop, angle, scale_try)
                    if scaled_arr.sum() == 0:
                        continue
                    sh, sw = scaled_arr.shape
                    m_ys, m_xs = np.where(scaled_arr > 0.5)
                    m_cy, m_cx = m_ys.mean(), m_xs.mean()

                    for cid, (fg, (H, W), fg_cy, fg_cx, fg_sum) in fg_info.items():
                        if sh > H or sw > W:
                            continue
                        top = int(fg_cy - m_cy)
                        left = int(fg_cx - m_cx)
                        if top < 0 or left < 0 or top + sh > H or left + sw > W:
                            continue

                        placed = np.zeros((H, W), dtype=np.float32)
                        placed[top : top + sh, left : left + sw] = scaled_arr

                        fg_inside = (fg * placed).sum()
                        if fg_inside >= fg_sum:
                            cov = placed.sum() / (H * W) * 100
                            return (cid, placed, angle, scale_try, cov)

    return None

"""
CLIP crop utilities: component grouping and group-based cropping.

Pipeline:
1. Connected components via scipy.ndimage.label
2. Rectangular bounding box per component (+ 10% padding)
3. Greedy spatial grouping: merge nearby components if padded bbox ≤ 224
4. Containment pass: if group B's bbox is inside group A's crop region, absorb B
   (crop region = square of max(padded_size, 224), centered on padded bbox)
5. Squarify only at crop time
6. Crop:
   - P ≤ 224: random 224×224 position containing full padded bbox
   - P > 224: crop P×P (the square padded bbox), resize to 224×224
"""
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


def lanczos_resize_tensor(t: torch.Tensor, size: int, value_range: str = "01") -> torch.Tensor:
    """Resize a [C, H, W] image tensor to (size, size) via PIL LANCZOS-3.

    F.interpolate has no LANCZOS path, so we round-trip through PIL. Precision
    loss is bounded at 1/255 (~0.4%) which is negligible for image content.

    Args:
        t: [C, H, W] float tensor (C ∈ {1, 3}). Expected range matches `value_range`.
        size: Target H = W.
        value_range: "01" for [0, 1] tensors, "-11" for [-1, 1] tensors.
    """
    if value_range == "-11":
        x = (t + 1.0) / 2.0
    else:
        x = t
    arr = (x.detach().clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().contiguous().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        pil = Image.fromarray(arr[..., 0], mode="L")
    else:
        pil = Image.fromarray(arr, mode="RGB")
    pil = pil.resize((size, size), Image.LANCZOS)
    out_np = np.array(pil)  # copy → writable
    if out_np.ndim == 2:
        out = torch.from_numpy(out_np).unsqueeze(0)
    else:
        out = torch.from_numpy(out_np).permute(2, 0, 1).contiguous()
    out = out.to(t.dtype) / 255.0
    if value_range == "-11":
        out = out * 2.0 - 1.0
    return out


def square_bbox(
    bbox: Tuple[int, int, int, int],
    img_h: int,
    img_w: int,
) -> Tuple[int, int, int, int]:
    """Expand rectangular bbox to square, centered, clipped to image bounds.

    Args:
        bbox: (y0, y1, x0, x1) inclusive coordinates
        img_h: Image height
        img_w: Image width

    Returns:
        Square bbox (y0, y1, x0, x1)
    """
    y0, y1, x0, x1 = bbox
    bh = y1 - y0 + 1
    bw = x1 - x0 + 1
    s = max(bh, bw)
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    ny0 = int(round(cy - (s - 1) / 2.0))
    ny1 = ny0 + s - 1
    nx0 = int(round(cx - (s - 1) / 2.0))
    nx1 = nx0 + s - 1
    # Shift to stay within image bounds
    if ny0 < 0:
        ny1 -= ny0
        ny0 = 0
    if ny1 >= img_h:
        ny0 -= (ny1 - img_h + 1)
        ny1 = img_h - 1
    if nx0 < 0:
        nx1 -= nx0
        nx0 = 0
    if nx1 >= img_w:
        nx0 -= (nx1 - img_w + 1)
        nx1 = img_w - 1
    ny0 = max(0, ny0)
    nx0 = max(0, nx0)
    return (ny0, ny1, nx0, nx1)


def merge_bboxes(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """Merge two bboxes into their union."""
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


def pad_bbox(
    bbox: Tuple[int, int, int, int],
    frac: float,
    img_h: int,
    img_w: int,
) -> Tuple[int, int, int, int]:
    """Pad bbox by fraction of max side, clipped to image bounds.

    Args:
        bbox: (y0, y1, x0, x1) inclusive
        frac: Padding fraction (e.g., 0.1 for 10%)
        img_h: Image height
        img_w: Image width

    Returns:
        Padded bbox (y0, y1, x0, x1)
    """
    y0, y1, x0, x1 = bbox
    s = max(y1 - y0 + 1, x1 - x0 + 1)
    p = max(1, int(s * frac))
    return (max(0, y0 - p), min(img_h - 1, y1 + p), max(0, x0 - p), min(img_w - 1, x1 + p))


def bbox_size(bbox: Tuple[int, int, int, int]) -> int:
    """Max side length of a bbox."""
    return max(bbox[1] - bbox[0] + 1, bbox[3] - bbox[2] + 1)


def _crop_region(
    bbox: Tuple[int, int, int, int],
    pad_frac: float,
    max_crop: int,
    img_h: int,
    img_w: int,
) -> Tuple[int, int, int, int]:
    """Compute the actual square crop region for a group's merged bbox.

    The crop is always at least max_crop×max_crop (since CLIP needs 224×224).
    For larger groups, the crop is the squarified padded bbox.
    Returns (y0, y1, x0, x1) clipped to image bounds.
    """
    padded = pad_bbox(bbox, pad_frac, img_h, img_w)
    crop_s = max(bbox_size(padded), max_crop)
    cy = (padded[0] + padded[1]) / 2.0
    cx = (padded[2] + padded[3]) / 2.0
    ry0 = int(round(cy - (crop_s - 1) / 2.0))
    ry1 = ry0 + crop_s - 1
    rx0 = int(round(cx - (crop_s - 1) / 2.0))
    rx1 = rx0 + crop_s - 1
    # Shift to stay within image bounds
    if ry0 < 0:
        ry1 -= ry0; ry0 = 0
    if ry1 >= img_h:
        ry0 -= (ry1 - img_h + 1); ry1 = img_h - 1
    if rx0 < 0:
        rx1 -= rx0; rx0 = 0
    if rx1 >= img_w:
        rx0 -= (rx1 - img_w + 1); rx1 = img_w - 1
    return (max(0, ry0), ry1, max(0, rx0), rx1)


def get_component_groups(
    mask: np.ndarray,
    img_h: int,
    img_w: int,
    max_crop: int = 224,
    pad_frac: float = 0.0,
) -> List[Dict]:
    """Find connected components and group spatially overlapping ones.

    Bboxes stay RECTANGULAR during grouping. Squarification only at crop time.

    Pipeline:
    1. Connected components → raw rectangular bboxes
    2. Greedy merge: combine if padded union max-side ≤ max_crop
    3. Containment: if group B's bbox is inside group A's crop region → absorb
       (crop region = square of max(padded_size, max_crop))
    4. Final: squarify + pad for actual cropping

    Args:
        mask: Binary mask (H, W), values in {0, 1} or {0, 255}
        img_h: Image height
        img_w: Image width
        max_crop: Maximum crop size for a group (default: 224 for CLIP)
        pad_frac: Padding fraction for bounding boxes

    Returns:
        List of group dicts, each with:
          - 'component_ids': list of component label ints
          - 'mask': combined binary mask for the group (H, W)
          - 'bbox': square padded bbox (y0, y1, x0, x1) ready for cropping
          - 'bbox_size': max side of the padded bbox
    """
    mask_bin = (mask > 0.5).astype(np.uint8) if mask.dtype != np.uint8 else mask
    if mask_bin.max() > 1:
        mask_bin = (mask_bin > 127).astype(np.uint8)

    labeled, n_components = ndimage.label(mask_bin)

    if n_components == 0:
        return []

    # Raw RECTANGULAR bbox per component — NO squarification
    components = []
    for c in range(1, n_components + 1):
        comp_mask = (labeled == c)
        ys, xs = np.where(comp_mask)
        if len(ys) == 0:
            continue
        raw_bb = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        components.append({
            'id': c,
            'mask': comp_mask.astype(np.float32),
            'bbox': raw_bb,
            'area': int(comp_mask.sum()),
        })

    if not components:
        return []

    # Greedy spatial grouping: sort by y-center, merge if padded union ≤ max_crop
    sorted_idxs = sorted(
        range(len(components)),
        key=lambda i: (components[i]['bbox'][0] + components[i]['bbox'][1]) / 2,
    )

    groups: List[Dict] = []
    for idx in sorted_idxs:
        comp = components[idx]
        placed = False
        for g in groups:
            candidate = merge_bboxes(g['_merged_bb'], comp['bbox'])
            padded = pad_bbox(candidate, pad_frac, img_h, img_w)
            if bbox_size(padded) <= max_crop:
                g['_comp_idxs'].append(idx)
                g['_merged_bb'] = candidate
                placed = True
                break
        if not placed:
            groups.append({
                '_comp_idxs': [idx],
                '_merged_bb': comp['bbox'],
            })

    # Pass 1: merge groups whose combined padded bbox fits in max_crop
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(len(groups) - 1, i, -1):
                candidate = merge_bboxes(
                    groups[i]['_merged_bb'], groups[j]['_merged_bb'],
                )
                padded = pad_bbox(candidate, pad_frac, img_h, img_w)
                if bbox_size(padded) <= max_crop:
                    groups[i]['_comp_idxs'].extend(groups[j]['_comp_idxs'])
                    groups[i]['_merged_bb'] = candidate
                    groups.pop(j)
                    changed = True
            if changed:
                break

    # Pass 2: containment — if group B's bbox is inside group A's crop
    # region, absorb B into A (or vice versa). Crop region is a square of
    # max(padded_size, max_crop) — the actual area that gets cropped.
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            ri = _crop_region(
                groups[i]['_merged_bb'], pad_frac, max_crop, img_h, img_w,
            )
            for j in range(i + 1, len(groups)):
                rj = _crop_region(
                    groups[j]['_merged_bb'], pad_frac, max_crop, img_h, img_w,
                )
                bb_i = groups[i]['_merged_bb']
                bb_j = groups[j]['_merged_bb']
                # j inside i's crop region?
                if (bb_j[0] >= ri[0] and bb_j[1] <= ri[1]
                        and bb_j[2] >= ri[2] and bb_j[3] <= ri[3]):
                    groups[i]['_comp_idxs'].extend(groups[j]['_comp_idxs'])
                    groups[i]['_merged_bb'] = merge_bboxes(bb_i, bb_j)
                    groups.pop(j)
                    changed = True
                    break
                # i inside j's crop region?
                elif (bb_i[0] >= rj[0] and bb_i[1] <= rj[1]
                        and bb_i[2] >= rj[2] and bb_i[3] <= rj[3]):
                    groups[j]['_comp_idxs'].extend(groups[i]['_comp_idxs'])
                    groups[j]['_merged_bb'] = merge_bboxes(bb_i, bb_j)
                    groups.pop(i)
                    changed = True
                    break
            if changed:
                break

    # Pass 3: overlap — if two groups' crop regions share any pixel, merge them.
    # Handles cases where groups partially overlap but neither fully contains
    # the other (e.g. 5 groups piled in the same area of a toothbrush).
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            ri = _crop_region(
                groups[i]['_merged_bb'], pad_frac, max_crop, img_h, img_w,
            )
            for j in range(i + 1, len(groups)):
                rj = _crop_region(
                    groups[j]['_merged_bb'], pad_frac, max_crop, img_h, img_w,
                )
                # Rectangles overlap if they share any pixel
                if (ri[0] <= rj[1] and ri[1] >= rj[0]
                        and ri[2] <= rj[3] and ri[3] >= rj[2]):
                    groups[i]['_comp_idxs'].extend(groups[j]['_comp_idxs'])
                    groups[i]['_merged_bb'] = merge_bboxes(
                        groups[i]['_merged_bb'], groups[j]['_merged_bb'])
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break

    # Build final group info — squarify NOW for actual cropping
    result = []
    for g in groups:
        group_mask = np.zeros_like(mask_bin, dtype=np.float32)
        for idx in g['_comp_idxs']:
            group_mask = np.maximum(group_mask, components[idx]['mask'])

        merged_bb = g['_merged_bb']
        for idx in g['_comp_idxs']:
            merged_bb = merge_bboxes(merged_bb, components[idx]['bbox'])
        # Squarify + pad for cropping
        sq_bb = square_bbox(merged_bb, img_h, img_w)
        padded_bb = pad_bbox(sq_bb, pad_frac, img_h, img_w)
        padded_bb = square_bbox(padded_bb, img_h, img_w)

        result.append({
            'component_ids': [components[idx]['id'] for idx in g['_comp_idxs']],
            'mask': group_mask,
            'bbox': padded_bb,
            'bbox_size': bbox_size(padded_bb),
        })

    return result


def clip_crop_from_groups(
    image: torch.Tensor,
    mask: torch.Tensor,
    groups: List[Dict],
    crop_size: int = 224,
    clip_masks: Optional[List[torch.Tensor]] = None,
    return_crop_meta: bool = False,
    position_constraint_mask: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
    """Pick a random group and produce a CLIP crop.

    Two modes:
    - Group bbox ≤ crop_size: random position 224×224 crop containing the bbox
    - Group bbox > crop_size: crop the bbox, resize to 224×224

    Args:
        image: Image tensor [C, H, W] (any range, e.g. [0,1] or [-1,1])
        mask: Full binary mask [1, H, W] in {0, 1}
        groups: Output of get_component_groups()
        crop_size: Target crop size (default: 224)
        clip_masks: Optional list of extra [1, H, W] masks to crop through the
            same window (e.g. roundtripped dilated/core masks for CLIP alignment).
        return_crop_meta: If True, append a crop_meta dict to the return tuple.
        position_constraint_mask: Optional [1, H, W] mask. When provided (e.g.
            dilated mask), the direct-crop branch picks a position that also
            contains this mask's per-group bbox. If no valid position exists,
            falls back to the resize branch using the constraint bbox.

    Returns:
        Base return (depends on clip_masks):
            If clip_masks is None: (cropped_image, cropped_mask)
            If clip_masks is provided: (cropped_image, cropped_mask, list_of_cropped_extra_masks)
        If return_crop_meta is True, an additional dict is appended:
            {"crop_top": int, "crop_left": int, "crop_h": int, "crop_w": int, "resized": bool}
            Coordinates are in original-resolution space.
    """
    def _crop_extra_masks(extra_masks, top, left, size_h, size_w, out_size):
        """Slice and resize extra masks through the same crop window."""
        results = []
        for em in extra_masks:
            em_crop = em[:, top:top + size_h, left:left + size_w]
            if size_h != out_size or size_w != out_size:
                em_out = F.adaptive_max_pool2d(
                    em_crop.unsqueeze(0).float(), (out_size, out_size),
                ).squeeze(0)
            else:
                em_out = em_crop
            results.append(em_out)
        return results

    def _pack(img_c, mask_c, extras, meta=None):
        result = (img_c, mask_c, extras) if clip_masks is not None else (img_c, mask_c)
        if return_crop_meta:
            result = result + (meta,)
        return result

    if not groups:
        # Fallback: center crop
        _, h, w = image.shape
        top = max(0, (h - crop_size) // 2)
        left = max(0, (w - crop_size) // 2)
        img_crop = image[:, top:top + crop_size, left:left + crop_size]
        mask_crop = mask[:, top:top + crop_size, left:left + crop_size]
        img_out = lanczos_resize_tensor(img_crop, crop_size, value_range="01")
        mask_out = F.adaptive_max_pool2d(mask_crop.unsqueeze(0).float(),
                                 (crop_size, crop_size)).squeeze(0)
        extras = _crop_extra_masks(clip_masks, top, left, crop_size, crop_size, crop_size) if clip_masks else []
        meta = {"crop_top": top, "crop_left": left, "crop_h": crop_size, "crop_w": crop_size, "resized": False}
        return _pack(img_out, mask_out, extras, meta)

    # Pick random group weighted by mask area (larger anomaly regions more likely)
    weights = [float(g['mask'].sum()) for g in groups]
    group = random.choices(groups, weights=weights, k=1)[0]
    y0, y1, x0, x1 = group['bbox']
    P = group['bbox_size']
    _, img_h, img_w = image.shape

    # Compute per-group constraint bbox (dilated region belonging to this group).
    # Expand the group's raw mask by half the crop size and intersect with
    # the constraint mask, then take the bbox of the intersection.
    cy0, cy1, cx0, cx1 = y0, y1, x0, x1
    if position_constraint_mask is not None:
        constraint_np = position_constraint_mask.squeeze(0).cpu().numpy() > 0.5
        group_mask = group['mask'] > 0.5
        expanded = ndimage.binary_dilation(group_mask, iterations=crop_size // 2)
        this_group_constraint = constraint_np & expanded
        ys_c, xs_c = np.where(this_group_constraint)
        if len(ys_c) > 0:
            cy0, cy1 = int(ys_c.min()), int(ys_c.max())
            cx0, cx1 = int(xs_c.min()), int(xs_c.max())

    constraint_h = cy1 - cy0 + 1
    constraint_w = cx1 - cx0 + 1
    constraint_max_side = max(constraint_h, constraint_w)

    if P <= crop_size and constraint_max_side <= crop_size:
        # Direct-crop branch: random 224×224 containing BOTH raw bbox and constraint bbox.
        # top must satisfy: top ≤ cy0 and top + crop_size - 1 ≥ cy1
        top_min = max(0, cy1 - crop_size + 1)
        top_max = min(cy0, img_h - crop_size)
        left_min = max(0, cx1 - crop_size + 1)
        left_max = min(cx0, img_w - crop_size)

        if top_min > top_max or left_min > left_max:
            # Near image edge — center on constraint bbox
            ccy = (cy0 + cy1) // 2
            ccx = (cx0 + cx1) // 2
            top = max(0, min(ccy - crop_size // 2, img_h - crop_size))
            left = max(0, min(ccx - crop_size // 2, img_w - crop_size))
        else:
            top = random.randint(top_min, top_max)
            left = random.randint(left_min, left_max)

        img_crop = image[:, top:top + crop_size, left:left + crop_size]
        mask_crop = mask[:, top:top + crop_size, left:left + crop_size]
        extras = _crop_extra_masks(clip_masks, top, left, crop_size, crop_size, crop_size) if clip_masks else []
        meta = {"crop_top": top, "crop_left": left, "crop_h": crop_size, "crop_w": crop_size, "resized": False}
        return _pack(img_crop, mask_crop, extras, meta)
    else:
        # Resize branch: use the LARGER of raw bbox and constraint bbox
        if constraint_max_side > P:
            # Constraint wins — square it up and use as crop region
            sq = square_bbox((cy0, cy1, cx0, cx1), img_h, img_w)
            ry0, ry1, rx0, rx1 = sq
        else:
            ry0, ry1, rx0, rx1 = y0, y1, x0, x1
        crop_h = ry1 + 1 - ry0
        crop_w = rx1 + 1 - rx0
        img_crop = image[:, ry0:ry1 + 1, rx0:rx1 + 1]
        mask_crop = mask[:, ry0:ry1 + 1, rx0:rx1 + 1]
        img_out = lanczos_resize_tensor(img_crop, crop_size, value_range="01")
        mask_out = F.adaptive_max_pool2d(
            mask_crop.unsqueeze(0).float(), (crop_size, crop_size),
        ).squeeze(0)
        extras = _crop_extra_masks(clip_masks, ry0, rx0, crop_h, crop_w, crop_size) if clip_masks else []
        meta = {"crop_top": ry0, "crop_left": rx0, "crop_h": crop_h, "crop_w": crop_w, "resized": True}
        return _pack(img_out, mask_out, extras, meta)


def clip_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: int = 224,
    pad_frac: float = 0.0,
    clip_masks: Optional[List[torch.Tensor]] = None,
    return_crop_meta: bool = False,
    position_constraint_mask: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
    """Full CLIP crop pipeline: group components and crop a random group.

    Convenience function combining get_component_groups + clip_crop_from_groups.

    Args:
        image: Image tensor [C, H, W]
        mask: Binary mask tensor [1, H, W] in {0, 1}
        crop_size: Target size (default: 224)
        pad_frac: Bbox padding fraction (default: 0.10)
        clip_masks: Optional list of extra [1, H, W] masks to crop.
        return_crop_meta: If True, append crop coordinate dict to return tuple.

    Returns:
        If clip_masks is None:
            (cropped_image, cropped_mask)
        If clip_masks is provided:
            (cropped_image, cropped_mask, list_of_cropped_extra_masks)
        If return_crop_meta is True, an additional dict is appended.
    """
    _, img_h, img_w = image.shape
    mask_np = mask.squeeze(0).cpu().numpy()
    groups = get_component_groups(mask_np, img_h, img_w, max_crop=crop_size, pad_frac=pad_frac)
    return clip_crop_from_groups(
        image, mask, groups, crop_size=crop_size,
        clip_masks=clip_masks, return_crop_meta=return_crop_meta,
        position_constraint_mask=position_constraint_mask,
    )


def clip_crop_multi(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: int = 224,
    pad_frac: float = 0.0,
    n_groups: int = 2,
    clip_masks: Optional[List[torch.Tensor]] = None,
    return_crop_meta: bool = False,
    position_constraint_mask: Optional[torch.Tensor] = None,
) -> Union[
    Tuple[List[torch.Tensor], List[torch.Tensor], List[bool]],
    Tuple[List[torch.Tensor], List[torch.Tensor], List[bool], List[List[torch.Tensor]]],
]:
    """Return n_groups CLIP crops from different connected-component groups.

    Uses component grouping to split disconnected anomaly regions into separate
    crops for independent CLIP encoding (multi-crop mode).

    Args:
        image: Image tensor [C, H, W]
        mask: Binary mask tensor [1, H, W] in {0, 1}
        crop_size: Target crop size (default: 224)
        pad_frac: Bbox padding fraction (default: 0.10)
        n_groups: Number of groups to return (default: 2)
        clip_masks: Optional list of extra [1, H, W] masks to crop through the
            same window as each group (e.g. roundtripped dilated/core masks).
        return_crop_meta: If True, include crop coordinate dicts in the return.

    Returns:
        If clip_masks is None:
            (crops, crop_masks, valid)
        If clip_masks is provided:
            (crops, crop_masks, valid, extra_crops)
            where extra_crops[i] is a list of cropped extra masks for group i.
        If return_crop_meta is True, an additional list of crop_meta dicts is appended.
    """
    _, img_h, img_w = image.shape
    mask_np = mask.squeeze(0).cpu().numpy()
    groups = get_component_groups(mask_np, img_h, img_w, max_crop=crop_size, pad_frac=pad_frac)

    # Weighted sampling without replacement (proportional to anomaly pixel count)
    if len(groups) >= n_groups:
        pool = list(groups)
        area_weights = [float(g['mask'].sum()) for g in pool]
        selected = []
        for _ in range(n_groups):
            chosen = random.choices(pool, weights=area_weights, k=1)[0]
            idx = pool.index(chosen)
            selected.append(chosen)
            pool.pop(idx)
            area_weights.pop(idx)
    else:
        selected = list(groups)

    crops: List[torch.Tensor] = []
    crop_masks: List[torch.Tensor] = []
    valid: List[bool] = []
    all_extras: List[List[torch.Tensor]] = []
    all_metas: List[Dict] = []

    for i in range(n_groups):
        if i < len(selected):
            result = clip_crop_from_groups(
                image, mask, [selected[i]], crop_size=crop_size,
                clip_masks=clip_masks, return_crop_meta=return_crop_meta,
                position_constraint_mask=position_constraint_mask,
            )
            if return_crop_meta:
                if clip_masks is not None:
                    crop_img, crop_mask, extras, meta = result
                    all_extras.append(extras)
                else:
                    crop_img, crop_mask, meta = result
                all_metas.append(meta)
            else:
                if clip_masks is not None:
                    crop_img, crop_mask, extras = result
                    all_extras.append(extras)
                else:
                    crop_img, crop_mask = result
            crops.append(crop_img)
            crop_masks.append(crop_mask)
            valid.append(True)
        else:
            # No group for this slot — zero tensors
            crops.append(torch.zeros(
                image.shape[0], crop_size, crop_size,
                device=image.device, dtype=image.dtype,
            ))
            crop_masks.append(torch.zeros(
                1, crop_size, crop_size,
                device=mask.device, dtype=mask.dtype,
            ))
            valid.append(False)
            if clip_masks is not None:
                all_extras.append([
                    torch.zeros(1, crop_size, crop_size, device=mask.device, dtype=mask.dtype)
                    for _ in clip_masks
                ])
            if return_crop_meta:
                all_metas.append({"crop_top": 0, "crop_left": 0, "crop_h": 0, "crop_w": 0, "resized": False})

    result_tuple: tuple = ()
    if clip_masks is not None:
        result_tuple = (crops, crop_masks, valid, all_extras)
    else:
        result_tuple = (crops, crop_masks, valid)
    if return_crop_meta:
        result_tuple = result_tuple + (all_metas,)
    return result_tuple

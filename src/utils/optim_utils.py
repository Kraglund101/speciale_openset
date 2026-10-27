"""Optimizer utilities: parameter group splitting + L2-SP regularization."""
from __future__ import annotations

from typing import Dict, List, Set, Tuple, Union

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────
# Module flattening
# ──────────────────────────────────────────────────────────────────────

def flatten_modules(
    obj: Union[nn.Module, list, tuple, dict, nn.ModuleList, nn.ModuleDict],
) -> List[nn.Module]:
    """Normalize a container of modules into a flat list.

    Accepts nn.Module, list/tuple, dict, ModuleList, ModuleDict.
    Does NOT recurse into sub-modules — just unwraps the top-level container.
    """
    if isinstance(obj, nn.ModuleDict):
        return list(obj.values())
    if isinstance(obj, (nn.ModuleList, list, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        return list(obj.values())
    if isinstance(obj, nn.Module):
        return [obj]
    raise TypeError(f"Cannot flatten object of type {type(obj)}")


# ──────────────────────────────────────────────────────────────────────
# Norm parameter detection
# ──────────────────────────────────────────────────────────────────────

_NORM_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def build_norm_param_id_set(modules: List[nn.Module]) -> Set[int]:
    """Collect ``id(p)`` for every parameter belonging to a normalisation layer.

    Two heuristics:
    1. ``isinstance(submodule, <norm_type>)`` for standard PyTorch norms.
    2. Fallback: if ``submodule.__class__.__name__`` contains "norm" (case-
       insensitive), include its params too.
    """
    ids: Set[int] = set()
    for mod in modules:
        for name, submod in mod.named_modules():
            is_norm = isinstance(submod, _NORM_TYPES)
            if not is_norm and "norm" in submod.__class__.__name__.lower():
                is_norm = True
            if is_norm:
                for p in submod.parameters():
                    ids.add(id(p))
    return ids


# ──────────────────────────────────────────────────────────────────────
# Decay / no-decay split
# ──────────────────────────────────────────────────────────────────────

def split_decay_no_decay(
    modules: List[nn.Module],
    norm_param_ids: Set[int],
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split parameters of *modules* into (decay, no_decay) lists.

    A parameter is **no_decay** if ANY of:
    - ``id(p)`` is in *norm_param_ids*
    - its name ends with ``".bias"``
    - ``"norm"`` appears (case-insensitive) anywhere in its name

    Otherwise it goes into **decay**.  Parameters that do not require grad
    are silently skipped.  Deduplication is done via ``id(p)``.
    """
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    seen: Set[int] = set()

    for mod in modules:
        for name, p in mod.named_parameters(recurse=True):
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)

            is_no_decay = (
                pid in norm_param_ids
                or name.endswith(".bias")
                or "norm" in name.lower()
            )
            if is_no_decay:
                no_decay.append(p)
            else:
                decay.append(p)

    return decay, no_decay


# ──────────────────────────────────────────────────────────────────────
# L2-SP Regularizer
# ──────────────────────────────────────────────────────────────────────

class L2SPRegularizer:
    """L2-SP: regularise towards pretrained weights instead of zero.

    ``L_sp = (lambda_sp / 2) * sum_i (theta_i - theta0_i)^2``

    The 1/2 is the standard convention so that the gradient is simply
    ``lambda_sp * (theta - theta0)`` with no factor of 2.

    *theta0* is captured once at construction (pretrained anchor).
    The regularisation term is computed in fp32 and returns a scalar
    tensor with gradient attached to the live parameters.
    """

    def __init__(self, params: List[nn.Parameter], lambda_sp: float = 1e-4) -> None:
        self.lambda_sp = lambda_sp
        self.params = list(params)
        # Capture pretrained weights as fixed anchors (independent clones)
        self.theta0 = [p.data.clone().detach() for p in self.params]
        self.total_elements = sum(p.numel() for p in self.params)

    def compute(self) -> torch.Tensor:
        """Return the L2-SP penalty (scalar with grad)."""
        loss = torch.tensor(0.0, device=self.params[0].device, dtype=torch.float32)
        for p, p0 in zip(self.params, self.theta0):
            loss = loss + ((p.float() - p0) ** 2).sum()
        return (self.lambda_sp / 2.0) * loss

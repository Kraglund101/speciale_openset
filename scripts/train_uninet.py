"""Portable launcher for UniNet / CAAO training.

Wraps the (very large) run_uninet_cashew.py with the thesis hero configurations,
resolving every path from the repo root.

Heroes (from thesis Table 15/16):
  m1    Mode 1, normal-only baseline      WRN-50-2, local KD, no two-sided, no SG
  caao  Mode 4, canvas-aligned anomaly-only  ConvNeXt-B, local KD, w_canvas=1,
        projection consistency lambda=0.2 (thesis Table 16 hero)

Examples:
  # normal-only hero, one seed, photometric augmentation OFF
  python scripts/train_uninet.py --hero m1 --no-photometric \
      --cashew-100-dir results/<run>/cashew_100_rp

  # CAAO hero
  python scripts/train_uninet.py --hero caao --no-photometric \
      --cashew-100-dir results/<run>/cashew_100_rp

  # anything else passes straight through to the underlying script
  python scripts/train_uninet.py --hero caao -- --epochs 20 --w-preserve 0.25
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "uninet" / "run_uninet_cashew.py"

# Flag sets that define each hero. Everything else is shared.
HEROES = {
    "m1": {
        "desc": "Mode 1 — normal-only baseline (WRN-50-2, local KD)",
        "flags": ["--modes", "1", "--lkd-mode", "local", "--backbone", "wrn50"],
    },
    # Mode 3 = original UniNet supervised loss on the synthetics (masks + BCE head). Backbone ConvNeXt-B as caao: the
    # runner builds WRN-50-2 for modes 1-3 and then swaps to ConvNeXt for EVERY mode when --backbone convnext_b (L5208).
    # Open-set repeat of every synthetic arm (2026-10-15, OPEN_SET_HERO=mode3).
    "mode3": {
        "desc": "Mode 3 — supervised UniNet on synthetic anomalies (ConvNeXt-B, local KD, BCE head)",
        "flags": ["--modes", "3", "--lkd-mode", "local", "--backbone", "convnext_b"],
    },
    # CAAO hero = thesis Table 16 (final_thesis.pdf p.141) and Table 13 / §6.7: lambda_cons 0.20, NO SG, local, w_canvas 1.
    # Thesis SG modes (§6.7 p.136): partial = SG only on the canvas target + the consistency target = the runner's
    # DEFAULT; full = --stop-gradient; no = --no-detach-any. The Table 16 CAAO row is the server run
    # mode4_caao_cos_nodetach_lambda020 (results/thesis_server/caao_hero_l020_nosg) -> --no-detach-any.
    # FIXED 2026-10-01: until then this hero lacked --no-detach-any, i.e. every local "caao" run was PARTIAL SG.
    # (2026-09-13..15 runs used lambda 0.10 by mistake, copied from run_d1_5seed_expand.sh -> hero "caao_l010".)
    "caao": {
        "desc": "Mode 4 — CAAO hero (ConvNeXt-B, local KD, w_canvas=1, projcons 0.2, no SG) — thesis Table 16",
        "flags": ["--modes", "4", "--lkd-mode", "local", "--backbone", "convnext_b",
                  "--canvas-align-anomaly-only", "--w-canvas", "1",
                  "--proj-consistency", "--lambda-proj", "0.2", "--no-detach-any"],
    },
    # The pre-2026-10-01 local "caao" (partial SG), kept so those runs stay reproducible.
    "caao_partial_sg": {
        "desc": "Mode 4 — CAAO, projcons 0.2, PARTIAL SG (runner default; what local 'caao' was until 2026-10-01)",
        "flags": ["--modes", "4", "--lkd-mode", "local", "--backbone", "convnext_b",
                  "--canvas-align-anomaly-only", "--w-canvas", "1",
                  "--proj-consistency", "--lambda-proj", "0.2"],
    },
    # Full-image canvas alignment (Term 2 over ALL positions, not only the anomaly mask) - background test 2x2.
    "caao_l010_uniform": {
        "desc": "Mode 4 — CAAO, lambda_proj 0.10, canvas alignment UNIFORM over all positions (no anomaly-only)",
        "flags": ["--modes", "4", "--lkd-mode", "local", "--backbone", "convnext_b",
                  "--canvas-align-mode", "uniform", "--w-canvas", "1",
                  "--proj-consistency", "--lambda-proj", "0.1"],
    },
    "caao_l010": {
        "desc": "Mode 4 — CAAO, lambda_proj 0.10, PARTIAL SG (D1 5-seed script variant; NOT the thesis hero)",
        "flags": ["--modes", "4", "--lkd-mode", "local", "--backbone", "convnext_b",
                  "--canvas-align-anomaly-only", "--w-canvas", "1",
                  "--proj-consistency", "--lambda-proj", "0.1"],
    },
}

# Photometric augmentation strengths used in the thesis runs.
PHOTOMETRIC_ON = ["--train-cj", "0.1", "--train-blur", "0.5", "--train-noise", "0.005"]
PHOTOMETRIC_OFF = ["--train-cj", "0", "--train-blur", "0", "--train-noise", "0"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hero", required=True, choices=sorted(HEROES),
                    help="which thesis hero configuration to train")
    ap.add_argument("--cashew-100-dir", type=Path, default=None,
                    help="generated cashews (…/cashew_100_rp). Required unless --dry-run.")
    ap.add_argument("--prep-dir", type=Path, default=None,
                    help="prep dir (default: <cashew-100-dir>/prep)")
    ap.add_argument("--mask-dir", type=Path, default=None,
                    help="refined masks (default: <cashew-100-dir>/refined_masks_cfg7_vis)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: results/uninet/<hero>/seed_<seed>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index; default: keep the inherited CUDA_VISIBLE_DEVICES (launchers pin MIG slices), else 0")
    ap.add_argument("--placement", default="rp", choices=["rp", "mo"])
    ap.add_argument("--cfg-variant", default="cfg7_vis")
    ap.add_argument("--no-photometric", action="store_true",
                    help="zero the colour-jitter / blur / noise augmentation")
    ap.add_argument("--no-augment", action="store_true",
                    help="drop --augment entirely (geometric augmentation too)")
    ap.add_argument("--no-save-best", action="store_true",
                    help="Do not keep the best-epoch checkpoint. By default every run also writes <name>_best.pt "
                         "(the epoch with the best composite image AUROC; chosen on the test set, like the tables' "
                         "'max' reduction) next to the final checkpoint.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved command without running it")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="everything after -- is forwarded to run_uninet_cashew.py")
    args = ap.parse_args()

    if not RUNNER.exists():
        print(f"ERROR: runner not found at\n  {RUNNER}", file=sys.stderr)
        return 2

    hero = HEROES[args.hero]
    cashew = args.cashew_100_dir
    prep = args.prep_dir or (cashew / "prep" if cashew else None)
    masks = args.mask_dir or (cashew / "refined_masks_cfg7_vis" if cashew else None)
    out = args.output_dir or (ROOT / "results" / "uninet" / args.hero / f"seed_{args.seed}")

    cmd = [sys.executable, str(RUNNER),
           "--placement", args.placement,
           "--cfg-variant", args.cfg_variant,
           "--output-dir", str(out),
           "--seed", str(args.seed),
           "--all-shifts"]
    if not args.no_augment:
        cmd.append("--augment")
    cmd += PHOTOMETRIC_OFF if args.no_photometric else PHOTOMETRIC_ON
    if not args.no_save_best:
        cmd.append("--save-best")
    if cashew:
        cmd += ["--cashew-100-dir", str(cashew)]
    if prep:
        cmd += ["--prep-dir", str(prep)]
    if masks:
        cmd += ["--mask-dir", str(masks)]
    cmd += hero["flags"]
    extra = [a for a in args.extra if a != "--"]
    cmd += extra

    print(f"hero      : {args.hero}  —  {hero['desc']}")
    print(f"photometric: {'OFF' if args.no_photometric else 'ON (cj .1 / blur .5 / noise .005)'}")
    print(f"output    : {out}")
    print(f"gpu       : {args.gpu if args.gpu is not None else os.environ.get('CUDA_VISIBLE_DEVICES', '0')}\n")
    print("command:")
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")

    # open-set runs give their synthetics / extra normals and test anomalies explicitly and need no fixed synthetic set
    open_set = "--test-anomaly-list" in extra and ("--synthetic-epochs" in extra or "--baseline-extra-list" in extra)
    missing = [] if open_set else [(n, p) for n, p in
               (("--cashew-100-dir", cashew), ("--prep-dir", prep), ("--mask-dir", masks))
               if p is None or not Path(p).exists()]
    if missing:
        print("MISSING INPUTS:")
        for n, p in missing:
            print(f"  {n:18s} {p if p else '(not given)'}")
        print("\nThese are produced by the generation pipeline, e.g.:")
        print("  python scripts/generate_100_cashews.py --checkpoint <ckpt> --all")
        print("  python scripts/compute_refined_masks.py")
        if not args.dry_run:
            print("\nRefusing to launch without them. Re-run with --dry-run to just see the command.")
            return 1

    if args.dry_run:
        return 0

    gpu = str(args.gpu) if args.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, SPECIALE_ROOT=str(ROOT))
    out.mkdir(parents=True, exist_ok=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())

"""Live loss curve viewer — auto-refreshes every 10s.

Reads:
  - losses.txt (one float per line) for per-step diffusion loss
  - stats.csv (23 cols) for diagnostics

3x3 grid layout:
  Row 0: [Cond Modes] [Core/Band (both cond)] [Band/Core Ratio | x0 | ctx]
  Row 1: [Gates] [Role Embeddings] [Grad Norm]
  Row 2: [IP/Text Output Ratio] [IP Entropy] [IP/Text Key Ratio]

Core/Band and Band/Core use s_core/s_band directly (both-cond filtered).

Usage:
  python live_loss.py <losses_file>
"""
import json
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def parse_losses(filepath: str) -> list:
    """Parse loss values — auto-detects format."""
    losses = []
    tqdm_pattern = re.compile(r"loss=([0-9]+\.[0-9]+)")
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                losses.append(float(line))
                continue
            except ValueError:
                pass
            m = tqdm_pattern.search(line)
            if m:
                losses.append(float(m.group(1)))
    return losses


def parse_stats(filepath: str) -> dict:
    """Parse stats.csv -> dict of arrays (23 columns)."""
    steps, diff_losses = [], []
    attn_gates, ff_gates, l2sp_vals = [], [], []
    core_losses, band_losses = [], []
    x0_ratios, x0_losses, eps_losses = [], [], []
    grad_norms, ctx_drops = [], []
    emb_global, emb_anomaly, emb_normal = [], [], []
    loss_keeps, loss_drop_viss, loss_drop_txts = [], [], []
    ip_entropies, ip_norm_ratios, ip_key_ratios = [], [], []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                steps.append(int(parts[0]))
                diff_losses.append(float(parts[1]))
                attn_gates.append(float(parts[4]))
                ff_gates.append(float(parts[5]))
                l2sp_vals.append(float(parts[6]))
                core_losses.append(float(parts[7]) if len(parts) > 8 else 0.0)
                band_losses.append(float(parts[8]) if len(parts) > 8 else 0.0)
                x0_ratios.append(float(parts[9]) if len(parts) > 9 else 0.0)
                x0_losses.append(float(parts[10]) if len(parts) > 10 else 0.0)
                eps_losses.append(float(parts[11]) if len(parts) > 11 else 0.0)
                grad_norms.append(float(parts[12]) if len(parts) > 12 else 0.0)
                ctx_drops.append(float(parts[13]) if len(parts) > 13 else 0.0)
                emb_global.append(float(parts[14]) if len(parts) > 14 else 0.0)
                emb_anomaly.append(float(parts[15]) if len(parts) > 15 else 0.0)
                emb_normal.append(float(parts[16]) if len(parts) > 16 else 0.0)
                loss_keeps.append(float(parts[17]) if len(parts) > 17 else 0.0)
                loss_drop_viss.append(float(parts[18]) if len(parts) > 18 else 0.0)
                loss_drop_txts.append(float(parts[19]) if len(parts) > 19 else 0.0)
                ip_entropies.append(float(parts[20]) if len(parts) > 20 else 0.0)
                ip_norm_ratios.append(float(parts[21]) if len(parts) > 21 else 0.0)
                ip_key_ratios.append(float(parts[22]) if len(parts) > 22 else 0.0)
    except FileNotFoundError:
        pass
    return {
        "steps": np.array(steps),
        "diff_loss": np.array(diff_losses),
        "attn_gate": np.array(attn_gates),
        "ff_gate": np.array(ff_gates),
        "l2sp": np.array(l2sp_vals),
        "core_loss": np.array(core_losses),
        "band_loss": np.array(band_losses),
        "x0_ratio": np.array(x0_ratios),
        "x0_loss": np.array(x0_losses),
        "eps_loss": np.array(eps_losses),
        "grad_norm": np.array(grad_norms),
        "ctx_drop": np.array(ctx_drops),
        "emb_global": np.array(emb_global),
        "emb_anomaly": np.array(emb_anomaly),
        "emb_normal": np.array(emb_normal),
        "loss_keep": np.array(loss_keeps),
        "loss_drop_vis": np.array(loss_drop_viss),
        "loss_drop_txt": np.array(loss_drop_txts),
        "ip_entropy": np.array(ip_entropies),
        "ip_norm_ratio": np.array(ip_norm_ratios),
        "ip_key_ratio": np.array(ip_key_ratios),
    }


def ema_smooth(values, smoothing=0.6):
    """Exponential moving average."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
    return ema


SMOOTH_FAST = 0.6
SMOOTH_SLOW = 0.99
SM = SMOOTH_SLOW


def _load_run_title(run_dir: str) -> str:
    """Build a descriptive title from run_config.json."""
    config_path = Path(run_dir) / "run_config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        parts = [
            f"VM{cfg.get('visual_mode', '?')}",
            f"LoRA={cfg['lora_rank']}" if cfg.get('lora_rank', 0) > 0 else "no-LoRA",
            f"BS={cfg.get('batch_size', '?')}",
            f"SA={cfg.get('sa_num_layers', '?')}L/{cfg.get('sa_num_heads', '?')}H",
            f"drop={cfg.get('drop_image_prob', 0) + cfg.get('drop_text_prob', 0) + cfg.get('drop_both_prob', 0):.0%}",
            f"gates={'yes' if cfg.get('learnable_gates', True) else 'no'}",
            f"UNet={'QO-' + cfg['unfreeze_qo'] if cfg.get('unfreeze_qo') else 'frozen'}",
            cfg.get('optimizer', 'adamw'),
            f"ts={cfg.get('timestep_sampling', '?')}",
            f"corrupt={cfg['corrupt_context']}" if cfg.get('corrupt_context', 0) > 0 else None,
            f"x0={cfg['x0_start_ratio']}->{cfg['x0_end_ratio']} w={cfg.get('x0_warmup_frac', 0)} ctx={'no' if cfg.get('x0_no_context') else 'yes'}" if cfg.get('x0_objective') else None,
        ]
        parts = [p for p in parts if p is not None]
        return " | ".join(parts)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return "Live Training Loss"


def main():
    if len(sys.argv) < 2:
        print("Usage: python live_loss.py <losses_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    run_dir = str(Path(filepath).parent)
    stats_path = str(Path(run_dir) / "stats.csv")
    run_title = _load_run_title(run_dir)

    fig, axes = plt.subplots(3, 3, figsize=(21, 15))
    fig.canvas.manager.set_window_title(run_title)
    fig.suptitle(run_title, fontsize=11, fontweight="bold")
    all_axes = list(axes.flat)

    def update(frame):
        losses = parse_losses(filepath)
        stats = parse_stats(stats_path)
        if not losses:
            return

        for ax in all_axes:
            ax.clear()

        steps = np.arange(len(losses))
        arr = np.array(losses)
        ema_slow = ema_smooth(arr, SM)

        s_steps = stats["steps"]
        s_core = stats["core_loss"]
        s_band = stats["band_loss"]
        s_x0 = stats["x0_loss"]
        s_eps = stats["eps_loss"]
        s_x0r = stats["x0_ratio"]
        s_gn = stats["grad_norm"]
        s_ctx = stats["ctx_drop"]
        s_loss_keep = stats["loss_keep"]
        s_loss_dv = stats["loss_drop_vis"]
        s_loss_dt = stats["loss_drop_txt"]
        s_ip_ent = stats["ip_entropy"]
        s_ip_nr = stats["ip_norm_ratio"]
        s_ip_kr = stats["ip_key_ratio"]

        has_cb = len(s_steps) > 0 and len(s_core) > 0 and s_core.any()
        has_x0 = len(s_steps) > 0 and s_x0.any()
        has_gn = len(s_steps) > 0 and s_gn.any()
        has_ctx = (len(s_steps) > 0 and s_ctx.any()
                   and (s_ctx.max() - s_ctx.min()) > 0.01)
        has_emb = (len(s_steps) > 0
                   and (stats["emb_global"].any() or stats["emb_anomaly"].any()
                        or stats["emb_normal"].any()))
        has_cond_modes = (len(s_steps) > 0
                          and (s_loss_keep.any() or s_loss_dv.any() or s_loss_dt.any()))
        has_ip_diag = len(s_steps) > 0 and (s_ip_ent.any() or s_ip_nr.any())

        # =====================================================
        # 3x3 grid — row 0 varies, rows 1-2 fixed
        # Row 0: [Cond Modes] [Core/Band (both cond)] [Band/Core | x0 | ctx]
        # Row 1: [Gates] [Role Embeddings] [Grad Norm]
        # Row 2: [IP/Text Output Ratio] [IP Entropy] [IP/Text Key Ratio]
        # =====================================================
        ax_trend, ax_cb = axes[0, 0], axes[0, 1]
        ax_x0eps = None
        ax_ctx = None
        if has_x0:
            ax_x0eps = axes[0, 2]
        elif has_ctx:
            ax_ctx = axes[0, 2]
        else:
            ax_ratio = axes[0, 2]

        # Row 1 fixed
        ax_gates = axes[1, 0]
        ax_emb = axes[1, 1]
        ax_gn = axes[1, 2]
        # Row 2 fixed: IP Output Ratio, IP Key Ratio, IP Entropy
        ax_ip_ratio = axes[2, 0]
        ax_ip_key = axes[2, 1]
        ax_ip_ent = axes[2, 2]

        # === [0,0] Conditioning Mode Losses ===
        if has_cond_modes:
            def _mode_ema_filtered(vals, steps_arr, sm):
                """EMA over nonzero entries only."""
                nz = vals > 0
                if not nz.any():
                    return np.array([]), np.array([])
                return steps_arr[nz], ema_smooth(vals[nz], sm)
            all_ema = ema_smooth(stats["diff_loss"], SM)
            ax_trend.plot(s_steps, all_ema, color="black", linewidth=2,
                          alpha=1.0, label=f"Overall ({all_ema[-1]:.4f})")
            k_st, k_em = _mode_ema_filtered(s_loss_keep, s_steps, SM)
            if len(k_em):
                ax_trend.plot(k_st, k_em, color="blue", linewidth=2,
                              alpha=1.0, label=f"Keep both ({k_em[-1]:.4f})")
            dv_st, dv_em = _mode_ema_filtered(s_loss_dv, s_steps, SM)
            if len(dv_em):
                ax_trend.plot(dv_st, dv_em, color="red", linewidth=1,
                              alpha=0.35, label=f"Keep text ({dv_em[-1]:.4f})")
            dt_st, dt_em = _mode_ema_filtered(s_loss_dt, s_steps, SM)
            if len(dt_em):
                ax_trend.plot(dt_st, dt_em, color="orange", linewidth=1,
                              alpha=0.35, label=f"Keep visual ({dt_em[-1]:.4f})")
            ax_trend.set_title(f"Loss by Conditioning Mode, EMA({SM}) \u2014 step {len(losses)}")
        else:
            ax_trend.plot(steps, ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
            ax_trend.set_title(f"Smooth Trend \u2014 step {len(losses)}, trend: {ema_slow[-1]:.4f}")
        ax_trend.set_xlabel("Step")
        ax_trend.set_ylabel("Loss")
        ax_trend.legend(loc="upper left", fontsize=8)
        ax_trend.grid(True, alpha=0.3)

        # === [0,1] Core vs Band (both cond only — direct from stats) ===
        if has_cb:
            ce = ema_smooth(s_core, SM)
            be = ema_smooth(s_band, SM)
            combined = 0.8 * ce + 0.2 * be
            ax_cb.fill_between(s_steps, 0, ce, color="#2196F3", alpha=0.4)
            ax_cb.fill_between(s_steps, ce, ce + be, color="#FF9800", alpha=0.4)
            ax_cb.plot(s_steps, ce, color="#1565C0", linewidth=1.5, label=f"Core ({ce[-1]:.4f})")
            ax_cb.plot(s_steps, ce + be, color="#E65100", linewidth=1.5, label=f"Band stacked ({be[-1]:.4f})")
            ax_cb.plot(s_steps, combined, color="black", linewidth=2, linestyle="--",
                       label=f"0.8\u00d7core+0.2\u00d7band = {combined[-1]:.4f}")
            ax_cb.legend(loc="upper left", fontsize=8)
            ax_cb.set_title(f"Core vs Band (both cond only), EMA({SM})")
        else:
            ax_cb.set_title("Core vs Band (no data yet)")
        ax_cb.set_xlabel("Step")
        ax_cb.set_ylabel("Per-pixel MSE")
        ax_cb.grid(True, alpha=0.3)

        # === [0,2] Band/Core Ratio (default) | x0 | ctx ===
        if ax_x0eps is not None:
            # x0 vs eps + annealing
            x0_vals_f = s_x0[s_x0 > 0]
            eps_vals_f = s_eps[s_eps > 0]
            x0_steps_f = s_steps[s_x0 > 0]
            eps_steps_f = s_steps[s_eps > 0]
            lines = []
            if len(x0_vals_f) > 1:
                x0_slow = ema_smooth(x0_vals_f, SM)
                l1, = ax_x0eps.plot(x0_steps_f, x0_slow, color="#D32F2F", linewidth=2.5, label="x0 loss")
                lines.append(l1)
            if len(eps_vals_f) > 1:
                eps_slow = ema_smooth(eps_vals_f, SM)
                l2, = ax_x0eps.plot(eps_steps_f, eps_slow, color="#1976D2", linewidth=2.5, label="\u03b5 loss")
                lines.append(l2)
            ax_ann = ax_x0eps.twinx()
            l3, = ax_ann.plot(s_steps, s_x0r, color="#FF9800", linewidth=1.5,
                              linestyle="--", alpha=0.7, label="x0 ratio")
            ax_ann.fill_between(s_steps, 0, s_x0r, color="#FFE0B2", alpha=0.3)
            ax_ann.set_ylim(-0.05, 1.1)
            ax_ann.set_ylabel("x0 ratio", color="#FF9800")
            lines.append(l3)
            ax_x0eps.legend(handles=lines, loc="upper left", fontsize=7)
            title_parts = []
            if len(x0_vals_f) > 1:
                title_parts.append(f"x0={ema_smooth(x0_vals_f, SM)[-1]:.4f}")
            if len(eps_vals_f) > 1:
                title_parts.append(f"\u03b5={ema_smooth(eps_vals_f, SM)[-1]:.4f}")
            ax_x0eps.set_title(f"x0 vs \u03b5 Loss, EMA({SM}) \u2014 {', '.join(title_parts)}")
            ax_x0eps.set_xlabel("Step")
            ax_x0eps.set_ylabel("Loss")
            ax_x0eps.grid(True, alpha=0.3)
        elif ax_ctx is not None:
            ax_ctx.plot(s_steps, s_ctx, color="#D32F2F", linewidth=2.5, label="Ctx dropout rate")
            ax_ctx.fill_between(s_steps, 0, s_ctx, color="#EF9A9A", alpha=0.3)
            ax_ctx.set_ylim(-0.05, 1.1)
            ax_ctx.set_title(f"Context Dropout Schedule \u2014 current: {s_ctx[-1]:.1%}")
            ax_ctx.set_xlabel("Step")
            ax_ctx.set_ylabel("Dropout rate")
            ax_ctx.legend(loc="upper left", fontsize=8)
            ax_ctx.grid(True, alpha=0.3)
        else:
            # Default: Band/Core Ratio (both cond only — direct from stats)
            if has_cb:
                safe_core = np.maximum(s_core, 1e-10)
                cbr = s_band / safe_core
                cbre = ema_smooth(cbr, SM)
                ax_ratio.plot(s_steps, cbr, color="gray", linewidth=0.5, alpha=0.3, label="Raw")
                ax_ratio.plot(s_steps, cbre, color="purple", linewidth=2, label=f"EMA ({SM})")
                ax_ratio.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
                ax_ratio.set_title(f"Band/Core Ratio (both cond only), EMA({SM}) \u2014 {cbre[-1]:.2f}x")
                ax_ratio.legend(loc="upper left", fontsize=8)
            else:
                ax_ratio.set_title("Band/Core Ratio (no data yet)")
            ax_ratio.set_xlabel("Step")
            ax_ratio.set_ylabel("Band / Core")
            ax_ratio.grid(True, alpha=0.3)

        # === [1,0] Gates ===
        if len(s_steps) > 0:
            ax_gates.plot(s_steps, stats["attn_gate"], color="blue",
                          linewidth=1.5, label="Attn gate")
            ax_gates.plot(s_steps, stats["ff_gate"], color="green",
                          linewidth=1.5, label="FF gate")
            ax_gates.set_title(f"Gates \u2014 attn={stats['attn_gate'][-1]:.4f}, "
                               f"ff={stats['ff_gate'][-1]:.4f}")
            ax_gates.legend(loc="upper left", fontsize=8)
        else:
            ax_gates.set_title("Gates (no data yet)")
        ax_gates.set_xlabel("Step")
        ax_gates.set_ylabel("Gate value")
        ax_gates.grid(True, alpha=0.3)

        # === [1,1] Role Embedding Norms ===
        if has_emb:
            s_eg = stats["emb_global"]
            s_ea = stats["emb_anomaly"]
            s_en = stats["emb_normal"]
            ax_emb.plot(s_steps, s_eg, color="#4CAF50", linewidth=2,
                        label=f"Global ({s_eg[-1]:.4f})")
            ax_emb.plot(s_steps, s_ea, color="#F44336", linewidth=2,
                        label=f"Anomaly ({s_ea[-1]:.4f})")
            ax_emb.plot(s_steps, s_en, color="#2196F3", linewidth=2,
                        label=f"Band ({s_en[-1]:.4f})")
            ax_emb.set_title("Role Embedding Norms")
            ax_emb.legend(loc="upper left", fontsize=8)
        else:
            ax_emb.set_title("Role Embedding Norms (no data)")
        ax_emb.set_xlabel("Step")
        ax_emb.set_ylabel("L2 Norm")
        ax_emb.grid(True, alpha=0.3)

        # === [1,2] Grad Norm ===
        if has_gn:
            gn_ema = ema_smooth(s_gn, SM)
            ax_gn.plot(s_steps, s_gn, color="gray", linewidth=0.5, alpha=0.3, label="Raw")
            ax_gn.plot(s_steps, gn_ema, color="#388E3C", linewidth=2.5, label=f"EMA ({SM})")
            ax_gn.axhline(y=1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, label="clip=1.0")
            ax_gn.set_title(f"Grad Norm \u2014 {gn_ema[-1]:.2f}")
            ax_gn.legend(loc="upper left", fontsize=8)
        else:
            ax_gn.set_title("Grad Norm (no data yet)")
        ax_gn.set_xlabel("Step")
        ax_gn.set_ylabel("Gradient L2 norm")
        ax_gn.grid(True, alpha=0.3)

        # === [2,0] IP Output Ratio (||ip_out|| / ||h_pre||) ===
        if has_ip_diag and s_ip_nr.any():
            nz = s_ip_nr > 0
            if nz.any():
                ema_nr = ema_smooth(s_ip_nr[nz], SM)
                ax_ip_ratio.plot(s_steps[nz], ema_nr, color="purple", linewidth=2,
                                 label=f"ratio ({ema_nr[-1]:.4f})")
                ax_ip_ratio.legend(loc="upper left", fontsize=8)
                ax_ip_ratio.set_title(f"IP Output Ratio ||ip_out||/||h_pre|| \u2014 {ema_nr[-1]:.4f}")
            else:
                ax_ip_ratio.set_title("IP Output Ratio ||ip_out||/||h_pre|| (no nonzero data)")
        else:
            ax_ip_ratio.set_title("IP Output Ratio ||ip_out||/||h_pre|| (no data yet)")
        ax_ip_ratio.set_xlabel("Step")
        ax_ip_ratio.set_ylabel("Ratio")
        ax_ip_ratio.grid(True, alpha=0.3)

        # === [2,1] IP Attention Entropy ===
        if has_ip_diag and s_ip_ent.any():
            nz = s_ip_ent > 0
            if nz.any():
                ema_ent = ema_smooth(s_ip_ent[nz], SM)
                ax_ip_ent.plot(s_steps[nz], ema_ent, color="teal", linewidth=2,
                               label=f"entropy ({ema_ent[-1]:.2f})")
                ax_ip_ent.legend(loc="upper left", fontsize=8)
                ax_ip_ent.set_title(f"IP Attention Entropy \u2014 {ema_ent[-1]:.2f}")
            else:
                ax_ip_ent.set_title("IP Attention Entropy (no nonzero data)")
        else:
            ax_ip_ent.set_title("IP Attention Entropy (no data yet)")
        ax_ip_ent.set_xlabel("Step")
        ax_ip_ent.set_ylabel("Entropy")
        ax_ip_ent.grid(True, alpha=0.3)

        # === [2,2] IP Key Ratio (||ip_k|| / ||text_k||) ===
        if has_ip_diag and s_ip_kr.any():
            nz = s_ip_kr > 0
            if nz.any():
                ema_kr = ema_smooth(s_ip_kr[nz], SM)
                ax_ip_key.plot(s_steps[nz], ema_kr, color="#E65100", linewidth=2,
                               label=f"key ratio ({ema_kr[-1]:.4f})")
                ax_ip_key.legend(loc="upper left", fontsize=8)
                ax_ip_key.set_title(f"IP Key Ratio ||ip_k||/||text_k|| \u2014 {ema_kr[-1]:.4f}")
            else:
                ax_ip_key.set_title("IP Key Ratio ||ip_k||/||text_k|| (no nonzero data)")
        else:
            ax_ip_key.set_title("IP Key Ratio ||ip_k||/||text_k|| (no data yet)")
        ax_ip_key.set_xlabel("Step")
        ax_ip_key.set_ylabel("Ratio")
        ax_ip_key.grid(True, alpha=0.3)

        fig.tight_layout()

    ani = FuncAnimation(fig, update, interval=10_000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()

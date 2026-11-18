r"""Generate 300-DPI IEEE/Elsevier Journal-Style Figures (Times New Roman Bold) from Live LLM Traces.

Reads ``results/live_llm_benchmark.json`` produced by ``scripts/run_live_llm_benchmark.py``
and generates two publication-grade figures in ``assets/``:
    1. ``assets/live_llm_pareto_frontier.png``
    2. ``assets/live_llm_verification_harm_audit.png``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossverify.markov_theory import compute_phase_diagram_grid

FAMILY_STYLE = {
    "S1_SinglePass": {"color": "#1F4E79", "marker": "o", "label": "S1: Single-Pass (Real LLM)"},
    "S2_SelfConsistency": {"color": "#C55A11", "marker": "s", "label": "S2: Self-Consistency (SC@K)"},
    "S3_GenVerifyLoop": {"color": "#5B2C6F", "marker": "^", "label": "S3: Generator-Verifier Loop"},
    "S4_AdversarialDebateDAG": {"color": "#A93226", "marker": "D", "label": "S4: 3-Family Debate DAG"},
    "S5_ParetoAdaptiveRouter": {"color": "#0B6E4F", "marker": "P", "label": "S5: ParetoAdaptiveRouter (Ours)"},
}


def _apply_journal_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "font.weight": "bold",
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.2,
            "axes.labelweight": "bold",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#111111",
            "axes.linewidth": 1.35,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.major.size": 4.6,
            "ytick.major.size": 4.6,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "axes.grid": True,
            "grid.color": "#94A3B8",
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
            "grid.linewidth": 0.75,
            "legend.edgecolor": "#111111",
            "legend.fancybox": False,
            "legend.framealpha": 0.96,
            "figure.facecolor": "#FFFFFF",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _polish_journal_ax(ax: plt.Axes, *, bar_chart: bool = False) -> None:
    if bar_chart:
        ax.tick_params(axis="x", top=False, direction="out", length=3.5, width=1.2)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontfamily("Times New Roman")
        lbl.set_fontweight("bold")
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_linewidth(1.1)
        leg.get_frame().set_edgecolor("#111111")
        for txt in leg.get_texts():
            txt.set_fontfamily("Times New Roman")
            txt.set_fontweight("bold")


def plot_live_pareto_frontier(data: Dict[str, Any], out_path: Path) -> None:
    cfgs = data["configurations"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.6, 6.2), gridspec_kw={"width_ratios": [1.26, 1.0]})

    # --- Panel (a): Empirical Token Cost vs Accuracy Pareto Frontier ---
    pareto_pts = sorted([c for c in cfgs if c["is_pareto_2d"]], key=lambda x: x["mean_tokens"])
    px = [p["mean_tokens"] for p in pareto_pts]
    py = [p["accuracy"] * 100.0 for p in pareto_pts]
    ax1.plot(px, py, color="#0B6E4F", lw=2.5, ls="-", zorder=3, label="Empirical Pareto Frontier")
    ax1.fill_between(px, py, min(c["accuracy"] * 100.0 for c in cfgs) - 3.0, color="#0B6E4F", alpha=0.07)

    seen_fams = set()
    for c in cfgs:
        fam = c["topology_family"]
        st = FAMILY_STYLE[fam]
        lbl = st["label"] if fam not in seen_fams else None
        seen_fams.add(fam)
        edge_col = "#B91C1C" if "[Harm]" in c["label"] else "#111111"
        sz = 165 if fam == "S5_ParetoAdaptiveRouter" else 98
        ax1.scatter(
            c["mean_tokens"],
            c["accuracy"] * 100.0,
            c=st["color"],
            marker=st["marker"],
            s=sz,
            edgecolors=edge_col,
            linewidths=1.6 if "[Harm]" in c["label"] else 1.15,
            zorder=6 if fam == "S5_ParetoAdaptiveRouter" else 4,
            label=lbl,
        )

    explicit_labels = {
        "S1_DeepSeekFlash_NoThink": (640, 49.3, "DeepSeek-Flash (NoThink)"),
        "S1_QwenFlash_NoThink": (580, 66.8, "Qwen3.8-Flash (NoThink)"),
        "S2_SC5_DeepSeekFlash": (1840, 55.2, "SC@5 DeepSeek-Flash"),
        "S1_MiniMax_M31": (940, 79.2, "MiniMax-M3.1"),
        "S1_MiMo_V26": (240, 88.8, "MiMo-V2.6-Flash"),
        "S5_ParetoRouter_Eco": (220, 95.8, "ParetoRouter-Eco (Ours)"),
        "S1_GLM53_FlashX": (360, 102.8, "GLM-5.3-FlashX"),
        "S5_ParetoRouter_Strict": (3420, 101.4, "ParetoRouter-Strict (Ours)"),
        "S3_DeepSeek_ToolGLMVer": (2320, 82.8, "DeepSeek + MicroCheck + GLM Ver"),
        "S4_HeteroDebate_3Family": (4220, 98.0, "3-Family Debate DAG"),
        "S3_MiniMax_WeakQwenCritic": (3560, 74.8, "MiniMax + Weak Critic [Harm]"),
        "S3_GLM53_WeakQwenCritic": (5250, 87.8, "GLM-5.3 + Weak Critic [Harm]"),
    }
    cfg_map = {c["config_id"]: c for c in cfgs}
    for cid, (tx, ty, short_txt) in explicit_labels.items():
        if cid in cfg_map:
            pt = cfg_map[cid]
            col = "#991B1B" if "Harm" in short_txt else ("#064E3B" if "Ours" in short_txt else "#111827")
            ax1.annotate(
                f"{short_txt}\n({pt['accuracy']*100:.1f}%, {pt['mean_tokens']:.0f} tok)",
                xy=(pt["mean_tokens"], pt["accuracy"] * 100.0),
                xytext=(tx, ty),
                fontsize=8.6,
                fontfamily="Times New Roman",
                fontweight="bold",
                color=col,
                arrowprops={"arrowstyle": "-|>", "color": col, "lw": 1.0, "alpha": 0.85},
            )

    ax1.set_xlim(120, 7680)
    ax1.set_ylim(45.0, 108.2)
    ax1.set_xlabel("Mean Live API Token Consumption per Task (Input + Reasoning + Output)")
    ax1.set_ylabel("Exact Deterministic Oracle Accuracy (%)")
    ax1.set_title("(a) Live Multi-Family LLM Cost–Accuracy Pareto Frontier (N = 45)")
    ax1.legend(loc="lower right", fontsize=8.8)
    _polish_journal_ax(ax1)

    # --- Panel (b): Stratified Easy / Medium / Hard Accuracy ---
    sel_ids = [
        ("S1_DeepSeekFlash_NoThink", "DeepSeek\nSinglePass"),
        ("S2_SC5_DeepSeekFlash", "DeepSeek\nSC@5 Vote"),
        ("S1_GLM53_FlashX", "GLM-5.3\nSinglePass"),
        ("S3_GLM53_WeakQwenCritic", "GLM-5.3 +\nWeak Critic"),
        ("S4_HeteroDebate_3Family", "3-Family\nDebate DAG"),
        ("S5_ParetoRouter_Eco", "Router-Eco\n(Ours)"),
        ("S5_ParetoRouter_Strict", "Router-Strict\n(Ours)"),
    ]
    sel_cfgs = [(cfg_map[cid], lbl) for cid, lbl in sel_ids if cid in cfg_map]
    x = np.arange(len(sel_cfgs))
    w = 0.25
    e_vals = [c["accuracy_by_difficulty"]["easy"] * 100.0 for c, _ in sel_cfgs]
    m_vals = [c["accuracy_by_difficulty"]["medium"] * 100.0 for c, _ in sel_cfgs]
    h_vals = [c["accuracy_by_difficulty"]["hard"] * 100.0 for c, _ in sel_cfgs]

    ax2.bar(x - w, e_vals, width=w, color="#4F81BD", edgecolor="#111111", lw=0.95, label="Easy Stratum (15)")
    ax2.bar(x, m_vals, width=w, color="#8064A2", edgecolor="#111111", lw=0.95, hatch="//", label="Medium Stratum (15)")
    ax2.bar(x + w, h_vals, width=w, color="#0B6E4F", edgecolor="#111111", lw=0.95, label="Hard Stratum (15)")

    for idx, (c, _) in enumerate(sel_cfgs):
        ax2.text(
            idx,
            max(e_vals[idx], m_vals[idx], h_vals[idx]) + 1.8,
            f"Avg\n{c['accuracy']*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontfamily="Times New Roman",
            fontweight="bold",
            color="#111111",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in sel_cfgs], fontsize=8.6)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Exact Oracle Accuracy (%)")
    ax2.set_title("(b) Difficulty-Stratified Live Accuracy Breakdown")
    ax2.legend(loc="lower right", fontsize=8.8)
    _polish_journal_ax(ax2, bar_chart=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_live_verification_harm_audit(data: Dict[str, Any], out_path: Path) -> None:
    audits: List[Dict[str, Any]] = data["markov_live_audits"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 6.0), gridspec_kw={"width_ratios": [1.06, 1.18]})

    # --- Panel (a): Empirical (alpha_hat, beta_hat) Overlaid on Markov Phase Diagram ---
    alphas = np.linspace(0.0, 0.99, 140)
    betas = np.linspace(0.0, 0.99, 140)
    p0_ref, q_ic_ref, q_ci_ref = 0.80, 0.60, 0.16
    A, B, delta_p = compute_phase_diagram_grid(
        p0=p0_ref, q_i2c=q_ic_ref, q_c2i=q_ci_ref, alpha_grid=alphas, beta_grid=betas
    )
    Z = delta_p * 100.0
    norm = TwoSlopeNorm(vmin=min(-8.0, float(np.min(Z))), vcenter=0.0, vmax=max(8.0, float(np.max(Z))))
    pcm = ax1.pcolormesh(A, B, Z, cmap="RdYlGn", norm=norm, shading="auto", alpha=0.84)
    cbar = fig.colorbar(pcm, ax=ax1, pad=0.025)
    cbar.outline.set_linewidth(1.2)
    cbar.outline.set_edgecolor("#111111")
    cbar.ax.tick_params(direction="in", width=1.1, length=3.8)
    for tl in cbar.ax.get_yticklabels():
        tl.set_fontfamily("Times New Roman")
        tl.set_fontweight("bold")
    cbar.set_label(
        r"Theoretical Net Gain $\mathbf{\Delta} p_1$ (pp, $p_0=0.80$ ref.)",
        fontsize=9.8,
        fontfamily="Times New Roman",
        fontweight="bold",
    )

    bound_alpha = np.clip(((1.0 - p0_ref) * q_ic_ref / (p0_ref * q_ci_ref)) * (1.0 - betas), 0.0, 1.0)
    ax1.plot(
        bound_alpha,
        betas,
        color="#111111",
        lw=2.3,
        ls="--",
        label=r"Critical Harm Boundary ($\mathbf{\Delta} p_1 = 0$)",
    )

    box_positions = {
        "S3_DeepSeek_ToolGLMVer": (0.07, 0.09),
        "S3_MiniMax_ToolGLMVer": (0.07, 0.56),
        "S3_DeepSeek_WeakQwenCritic": (0.25, 0.24),
        "S3_GLM53_WeakQwenCritic": (0.40, 0.42),
        "S3_MiniMax_WeakQwenCritic": (0.42, 0.69),
    }

    for aud in audits:
        al = aud["alpha_false_rejection"]
        be = aud["beta_false_acceptance"]
        gain = aud["net_gain_pp"]
        is_neg = gain < 0
        mk = "X" if is_neg else "P"
        col = "#B91C1C" if is_neg else "#065F46"
        ax1.scatter(al, be, s=175, c=col, marker=mk, edgecolors="white", linewidths=1.5, zorder=8)
        short = (
            aud["config_id"]
            .replace("S3_", "")
            .replace("_WeakQwenCritic", " + WeakCritic")
            .replace("_ToolGLMVer", " + ToolGLMVer")
        )
        bx, by = box_positions.get(aud["config_id"], (0.35, 0.50))
        ax1.annotate(
            rf"{short}" + "\n" + rf"($\mathbf{{\Delta}}={gain:+.1f}$ pp, $\hat{{\alpha}}={al:.2f}$, $\hat{{\beta}}={be:.2f}$)",
            xy=(al, be),
            xytext=(bx, by),
            fontsize=8.4,
            fontfamily="Times New Roman",
            fontweight="bold",
            color="#7F1D1D" if is_neg else "#064E3B",
            bbox={"boxstyle": "square,pad=0.26", "fc": "white", "ec": col, "lw": 1.25, "alpha": 0.95},
            arrowprops={"arrowstyle": "-|>", "color": col, "lw": 1.15},
            zorder=9,
        )

    ax1.set_xlim(-0.03, 0.99)
    ax1.set_ylim(-0.03, 0.98)
    ax1.set_xlabel(r"Empirical Type-I False-Rejection Rate $\hat{\alpha} = P(\mathrm{reject} \mid \mathrm{Correct})$")
    ax1.set_ylabel(r"Empirical Type-II False-Acceptance Rate $\hat{\beta} = P(\mathrm{accept} \mid \mathrm{Incorrect})$")
    ax1.set_title("(a) Live LLM Verifiers in Theoretical Markov Phase Space")
    ax1.legend(loc="upper right", fontsize=8.8)
    _polish_journal_ax(ax1)

    # --- Panel (b): Before vs After Verification Accuracy + Exact Markov Formula Alignment ---
    labels = [
        a["config_id"]
        .replace("S3_", "")
        .replace("_WeakQwenCritic", "\n+ Weak Qwen Critic")
        .replace("_ToolGLMVer", "\n+ MicroCheck & GLM Ver")
        for a in audits
    ]
    y_pos = np.arange(len(audits))
    h = 0.28
    p0_v = [a["p0_empirical"] * 100.0 for a in audits]
    p1_v = [a["p1_empirical"] * 100.0 for a in audits]
    pt_v = [a["p1_markov_formula"] * 100.0 for a in audits]

    ax2.barh(
        y_pos + h,
        p0_v,
        height=h,
        color="#94A3B8",
        edgecolor="#111111",
        lw=1.0,
        hatch="//",
        label=r"Initial Single-Pass Accuracy $\hat{p}_0$",
    )
    bar_colors = ["#C0392B" if a["net_gain_pp"] < 0 else "#0B6E4F" for a in audits]
    ax2.barh(
        y_pos,
        p1_v,
        height=h,
        color=bar_colors,
        edgecolor="#111111",
        lw=1.0,
        label=r"Post-Verification Live $\hat{p}_1$ (Red: Harm, Green: Gain)",
    )
    ax2.scatter(
        pt_v,
        y_pos,
        color="#FFD700",
        edgecolors="#111111",
        linewidths=1.3,
        marker="D",
        s=58,
        zorder=7,
        label=r"Closed-Form Theorem 1 Prediction $p_1^{\mathrm{theory}}$",
    )

    for i, a in enumerate(audits):
        delta = a["net_gain_pp"]
        tag = f"{delta:+.1f} pp ({a['n_corrupted_by_critic']} corrupted, {a['n_recovered_by_critic']} fixed)"
        ax2.text(
            max(p0_v[i], p1_v[i]) + 1.5,
            y_pos[i],
            tag,
            va="center",
            fontsize=8.6,
            fontfamily="Times New Roman",
            fontweight="bold",
            color="#991B1B" if delta < 0 else "#064E3B",
        )

    ax2.set_yticks(y_pos + h / 2)
    ax2.set_yticklabels(labels, fontsize=9.2)
    ax2.set_xlim(0, 146)
    ax2.set_ylim(-0.45, len(audits) + 0.42)
    ax2.set_xlabel("Exact Deterministic Oracle Accuracy (%)")
    ax2.set_title("(b) Empirical Verification Harm vs. Tool-Grounded Gain (N = 45)")
    ax2.legend(loc="upper right", fontsize=8.4, ncol=1)
    _polish_journal_ax(ax2)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    json_path = ROOT / "results" / "live_llm_benchmark.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    _apply_journal_style()
    plot_live_pareto_frontier(payload, ROOT / "assets" / "live_llm_pareto_frontier.png")
    plot_live_verification_harm_audit(payload, ROOT / "assets" / "live_llm_verification_harm_audit.png")

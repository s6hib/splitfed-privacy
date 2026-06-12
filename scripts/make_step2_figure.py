#!/usr/bin/env python3
"""
make_step2_figure.py -- aggregate the step 2 inversion sweep and plot it.

reads the per-(cut, seed) summaries from run_attacks.py
(results/inversion_<method>_<encoder>_cut<C>_seed<S>.csv), averages PSNR/SSIM/
MSE over whatever seeds are present, writes a summary CSV, and plots
reconstruction quality vs cut depth with error bars. works fine with a single
seed (std=0), so it can run while the full sweep is still going.

    python scripts/make_step2_figure.py                  # defaults: sflv1, trained
    python scripts/make_step2_figure.py --encoder random # untrained-encoder run
"""
import argparse
import csv
import glob
import os
import re
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Q2_DIR = os.path.dirname(SCRIPT_DIR)

METRICS = ("psnr", "ssim", "mse")


def discover_rows(results_dir: str, method: str, encoder: str,
                  cuts: List[int]) -> Dict[int, List[dict]]:
    """every per-seed result row, grouped by cut layer."""
    by_cut: Dict[int, List[dict]] = {c: [] for c in cuts}
    for cut in cuts:
        pattern = os.path.join(
            results_dir, f"inversion_{method}_{encoder}_cut{cut}_seed*.csv")
        for path in sorted(glob.glob(pattern)):
            with open(path, newline="") as f:
                row = next(csv.DictReader(f))  # one data row per file
            by_cut[cut].append(row)
    return by_cut


def aggregate(by_cut: Dict[int, List[dict]]) -> List[dict]:
    """per-cut mean/std for each metric (ddof=1 when >1 seed)."""
    summary = []
    for cut, rows in sorted(by_cut.items()):
        if not rows:
            continue
        seeds = sorted(int(r["seed"]) for r in rows)
        entry = {"cut_layer": cut, "n_seeds": len(rows),
                 "seeds": ";".join(map(str, seeds))}
        for m in METRICS:
            vals = np.array([float(r[m]) for r in rows], dtype=float)
            entry[f"{m}_mean"] = float(vals.mean())
            entry[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary.append(entry)
    return summary


def write_summary_csv(summary: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = (["cut_layer", "n_seeds", "seeds"]
            + [f"{m}_{stat}" for m in METRICS for stat in ("mean", "std")])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in summary:
            w.writerow({k: row[k] for k in cols})


def plot_figure(summary: List[dict], out_path: str, method: str,
                encoder: str) -> None:
    cuts = [s["cut_layer"] for s in summary]
    n_seeds = summary[0]["n_seeds"] if summary else 0

    fig, (ax_psnr, ax_ssim) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    def panel(ax, metric: str, ylabel: str, color: str) -> None:
        means = [s[f"{metric}_mean"] for s in summary]
        stds = [s[f"{metric}_std"] for s in summary]
        ax.errorbar(cuts, means, yerr=stds, marker="o", capsize=4,
                    color=color, linewidth=2, markersize=7)
        ax.set_xlabel("cut layer (split depth →)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(cuts)
        ax.grid(True, alpha=0.3)

    panel(ax_psnr, "psnr", "PSNR (dB)  ↑ = better reconstruction", "#c2410c")
    panel(ax_ssim, "ssim", "SSIM  ↑ = better reconstruction", "#1d4ed8")

    seed_note = f"{n_seeds} seed" + ("s" if n_seeds != 1 else "")
    fig.suptitle(
        f"Feature-inversion leakage vs. cut depth — {method.upper()} "
        f"({encoder} encoder, {seed_note})\n"
        f"deeper cut → lower reconstruction quality → better privacy",
        fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default=os.path.join(Q2_DIR, "results"))
    parser.add_argument("--figures_dir", default=os.path.join(Q2_DIR, "figures"))
    parser.add_argument("--method", default="sflv1")
    parser.add_argument("--encoder", default="trained", choices=["trained", "random"])
    parser.add_argument("--cuts", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()

    by_cut = discover_rows(args.results_dir, args.method, args.encoder, args.cuts)
    summary = aggregate(by_cut)
    if not summary:
        raise SystemExit(
            f"No inversion CSVs found in {args.results_dir} matching "
            f"inversion_{args.method}_{args.encoder}_cut*_seed*.csv")

    # console table
    print(f"\nStep 2 inversion summary — {args.method} ({args.encoder} encoder)")
    print(f"{'cut':<5}{'n':<4}{'PSNR (dB)':<18}{'SSIM':<16}{'MSE':<12}")
    for s in summary:
        cut_col = f"{s['cut_layer']:<5}{s['n_seeds']:<4}"
        psnr_col = f"{s['psnr_mean']:.2f} ± {s['psnr_std']:.2f}".ljust(18)
        ssim_col = f"{s['ssim_mean']:.4f} ± {s['ssim_std']:.4f}".ljust(16)
        print(cut_col + psnr_col + ssim_col + f"{s['mse_mean']:.5f}")

    summary_path = os.path.join(args.results_dir,
                                f"inversion_summary_{args.method}_{args.encoder}.csv")
    write_summary_csv(summary, summary_path)
    print(f"\nSummary CSV -> {summary_path}")

    fig_path = os.path.join(args.figures_dir,
                            f"step2_inversion_vs_cut_{args.method}_{args.encoder}.png")
    plot_figure(summary, fig_path, args.method, args.encoder)
    print(f"Figure      -> {fig_path}")


if __name__ == "__main__":
    main()

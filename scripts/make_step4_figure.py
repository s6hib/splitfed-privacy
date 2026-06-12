#!/usr/bin/env python3
"""
make_step4_figure.py -- plot the step 4 privacy-utility frontier.

reads the defense sweep from run_defenses.py
(results/defense_<defense>_cut<C>_sigma<S>_seed<N>.csv) and plots accuracy
against reconstruction quality as sigma grows. one curve per cut, each point
labelled with its sigma. the sigma=0 point anchors the top-right (leaky but
accurate); more noise pushes down-left.

    python scripts/make_step4_figure.py
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Q2_DIR = os.path.dirname(SCRIPT_DIR)


def load_rows(results_dir: str) -> list:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "defense_*_cut*_sigma*_seed*.csv"))):
        with open(path, newline="") as f:
            rows.append(next(csv.DictReader(f)))
    return rows


def aggregate(rows: list) -> dict:
    """group by (cut, sigma), mean across seeds -> {cut: sorted points}."""
    bucket = defaultdict(list)
    for r in rows:
        bucket[(int(r["cut_layer"]), float(r["sigma"]))].append(r)
    by_cut = defaultdict(list)
    for (cut, sigma), rs in bucket.items():
        by_cut[cut].append({
            "sigma": sigma,
            "acc": float(np.mean([float(r["model_acc"]) for r in rs])),
            "ssim": float(np.mean([float(r["ssim"]) for r in rs])),
            "psnr": float(np.mean([float(r["psnr"]) for r in rs])),
            "n": len(rs),
        })
    for cut in by_cut:
        by_cut[cut].sort(key=lambda d: d["sigma"])
    return dict(by_cut)


def plot_frontier(by_cut: dict, out_path: str) -> None:
    fig, (ax_ssim, ax_psnr) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {1: "#0e7490", 2: "#1d4ed8", 3: "#7c3aed", 4: "#be123c"}

    def panel(ax, xkey: str, xlabel: str) -> None:
        for cut, pts in sorted(by_cut.items()):
            xs = [p[xkey] for p in pts]
            ys = [p["acc"] for p in pts]
            ax.plot(xs, ys, "-o", color=colors.get(cut, "gray"),
                    label=f"cut {cut}", linewidth=2, markersize=6)
            for p in pts:
                ax.annotate(f"σ={p['sigma']:g}", (p[xkey], p["acc"]),
                            textcoords="offset points", xytext=(5, 4), fontsize=7,
                            color=colors.get(cut, "gray"))
        ax.set_xlabel(xlabel + "  (← more private)")
        ax.set_ylabel("model test accuracy (%)  (↑ more useful)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="split depth")

    panel(ax_ssim, "ssim", "reconstruction SSIM")
    panel(ax_psnr, "psnr", "reconstruction PSNR (dB)")
    fig.suptitle("Step 4 — privacy-utility frontier under smashed-data noise (PixelDP)\n"
                 "each point a noise level σ; up-right = leaky, down-left = private",
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
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"No defense_*.csv files in {args.results_dir}")
    by_cut = aggregate(rows)

    print("\nStep 4 privacy-utility frontier")
    print(f"{'cut':<5}{'sigma':<8}{'acc %':<10}{'SSIM':<10}{'PSNR':<8}")
    for cut, pts in sorted(by_cut.items()):
        for p in pts:
            print(f"{cut:<5}{p['sigma']:<8g}{p['acc']:<10.2f}{p['ssim']:<10.4f}{p['psnr']:<8.2f}")

    out_path = os.path.join(args.figures_dir, "step4_privacy_utility_frontier.png")
    plot_frontier(by_cut, out_path)
    print(f"\nFigure -> {out_path}")


if __name__ == "__main__":
    main()

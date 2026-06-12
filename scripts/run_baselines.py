#!/usr/bin/env python3
"""
run_baselines.py -- multi-seed runner for the training baselines.

runs the 5 methods (normal, FL, SL, SFLV1, SFLV2) and/or the SFLV1 cut-layer
sweep across seeds. results land in results/ with seed-aware filenames so the
figure scripts can aggregate mean +/- std.

modes:
  baseline  : all 5 methods @ cut=2, every seed
  cutsweep  : SFLV1 @ cuts 1-4, every seed
  all       : both (default)

examples:
  # smoke test (1 epoch, 1 seed)
  python scripts/run_baselines.py --mode baseline --epochs 1 --seeds 1234

  # full run (50 epochs, 3 seeds, full-width ResNet18)
  python scripts/run_baselines.py --mode all --epochs 50 --base_channels 64 \
      --seeds 1234 2026 42
"""
import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Q2_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(Q2_DIR, "src")

METHODS = {
    "normal": os.path.join(SRC_DIR, "baselines", "normal.py"),
    "fl":     os.path.join(SRC_DIR, "baselines", "fl.py"),
    "sl":     os.path.join(SRC_DIR, "baselines", "sl.py"),
    "sflv1":  os.path.join(SRC_DIR, "baselines", "sflv1.py"),
    "sflv2":  os.path.join(SRC_DIR, "baselines", "sflv2.py"),
}

SPLIT_METHODS = {"sl", "sflv1", "sflv2"}


def run_one(method, seed, cut_layer, epochs, base_channels, results_dir,
            save_client_ckpt=False, ckpt_dir=None):
    cmd = [
        sys.executable, METHODS[method],
        "--epochs", str(epochs),
        "--base_channels", str(base_channels),
        "--seed", str(seed),
        "--output_dir", results_dir,
    ]
    if method in SPLIT_METHODS:
        cmd += ["--cut_layer", str(cut_layer)]
        # only split methods have a client encoder to checkpoint (the
        # inversion attack needs these later)
        if save_client_ckpt:
            cmd += ["--save_client_ckpt"]
            if ckpt_dir:
                cmd += ["--ckpt_dir", ckpt_dir]

    tag = f"{method} seed={seed}"
    if method in SPLIT_METHODS:
        tag += f" cut={cut_layer}"

    print("\n" + "=" * 70)
    print(f"  {tag}")
    print("=" * 70)
    t0 = time.time()
    # cwd matters: torchvision downloads CIFAR-10 relative to it (-> data/)
    result = subprocess.run(cmd, cwd=Q2_DIR, check=False)
    elapsed = (time.time() - t0) / 60
    status = "OK" if result.returncode == 0 else f"FAIL ({result.returncode})"
    print(f"  {tag} -> {status} in {elapsed:.1f} min")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["baseline", "cutsweep", "all"],
                        default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--base_channels", type=int, default=64,
                        help="64=standard ResNet18 (Q1 prod), 16=lightweight CPU")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 2026, 42])
    parser.add_argument("--cut_layers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--baseline_cut", type=int, default=2,
                        help="Cut layer used by SL/SFLV1/SFLV2 in --mode baseline")
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(Q2_DIR, "results"))
    parser.add_argument("--save_client_ckpt", action="store_true",
                        help="Save trained client encoders (split methods only) "
                             "for the Q2 feature-inversion attack")
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.path.join(Q2_DIR, "results", "checkpoints"))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Q2 dir:     {Q2_DIR}")
    print(f"Results dir: {args.results_dir}")
    print(f"Mode:        {args.mode}")
    print(f"Epochs:      {args.epochs}")
    print(f"Width:       base_channels={args.base_channels}")
    print(f"Seeds:       {args.seeds}")

    failures = []

    if args.mode in ("baseline", "all"):
        print("\n" + "#" * 70)
        print(f"#  Experiment A: Baselines at cut={args.baseline_cut}")
        print("#" * 70)
        for method in ["normal", "fl", "sl", "sflv1", "sflv2"]:
            for seed in args.seeds:
                rc = run_one(method, seed, args.baseline_cut, args.epochs,
                             args.base_channels, args.results_dir,
                             args.save_client_ckpt, args.ckpt_dir)
                if rc != 0:
                    failures.append((method, seed, args.baseline_cut))

    if args.mode in ("cutsweep", "all"):
        print("\n" + "#" * 70)
        print("#  Experiment B: SFLV1 cut-layer sweep")
        print("#" * 70)
        for cut in args.cut_layers:
            for seed in args.seeds:
                rc = run_one("sflv1", seed, cut, args.epochs,
                             args.base_channels, args.results_dir,
                             args.save_client_ckpt, args.ckpt_dir)
                if rc != 0:
                    failures.append(("sflv1", seed, cut))

    print("\n" + "=" * 70)
    if failures:
        print(f"  DONE with {len(failures)} failure(s):")
        for method, seed, cut in failures:
            print(f"    - {method} seed={seed} cut={cut}")
        sys.exit(1)
    else:
        print("  ALL RUNS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_attacks.py -- feature-inversion sweep across cut layers (step 2).

for each (cut, seed): train a decoder to reconstruct CIFAR-10 images from the
client's smashed data, record PSNR/SSIM + a reconstruction grid. aggregate
with make_step2_figure.py to see how leakage falls as the cut moves deeper.

needs client checkpoints first (trained-encoder mode):

  python scripts/run_baselines.py --mode cutsweep --epochs 50 \
      --base_channels 64 --seeds 1234 2026 42 --save_client_ckpt

then:

  python scripts/run_attacks.py --seeds 1234 2026 42 --epochs 30

or skip checkpoints and invert untrained encoders:

  python scripts/run_attacks.py --random_encoder --seeds 1234
"""
import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Q2_DIR = os.path.dirname(SCRIPT_DIR)
ATTACK_SCRIPT = os.path.join(Q2_DIR, "src", "attacks", "feature_inversion.py")


def run_one(cut, seed, args):
    cmd = [
        sys.executable, ATTACK_SCRIPT,
        "--cut_layer", str(cut),
        "--seed", str(seed),
        "--method", args.method,
        "--base_channels", str(args.base_channels),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--width", str(args.width),
        "--data_dir", args.data_dir,
        "--ckpt_dir", args.ckpt_dir,
        "--output_dir", args.results_dir,
        "--figures_dir", args.figures_dir,
    ]
    if args.random_encoder:
        cmd.append("--random_encoder")

    tag = f"inversion cut={cut} seed={seed} ({'random' if args.random_encoder else 'trained'})"
    print("\n" + "=" * 70)
    print(f"  {tag}")
    print("=" * 70)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=Q2_DIR, check=False)
    elapsed = (time.time() - t0) / 60
    status = "OK" if result.returncode == 0 else f"FAIL ({result.returncode})"
    print(f"  {tag} -> {status} in {elapsed:.1f} min")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cut_layers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 2026, 42])
    parser.add_argument("--method", type=str, default="sflv1")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30, help="Decoder training epochs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--random_encoder", action="store_true")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(Q2_DIR, "data"))
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.path.join(Q2_DIR, "results", "checkpoints"))
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(Q2_DIR, "results"))
    parser.add_argument("--figures_dir", type=str,
                        default=os.path.join(Q2_DIR, "figures"))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print(f"Q2 dir:      {Q2_DIR}")
    print(f"Cut layers:  {args.cut_layers}")
    print(f"Seeds:       {args.seeds}")
    print(f"Encoder:     {'random (untrained)' if args.random_encoder else 'trained (' + args.method + ')'}")
    print(f"Decoder eps: {args.epochs}")

    failures = []
    for cut in args.cut_layers:
        for seed in args.seeds:
            rc = run_one(cut, seed, args)
            if rc != 0:
                failures.append((cut, seed))

    print("\n" + "=" * 70)
    if failures:
        print(f"  DONE with {len(failures)} failure(s):")
        for cut, seed in failures:
            print(f"    - cut={cut} seed={seed}")
        sys.exit(1)
    print("  ALL INVERSION RUNS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

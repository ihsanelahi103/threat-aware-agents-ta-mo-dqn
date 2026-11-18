"""
run_complete_experiment.py — Complete Agents Experiment Pipeline
-------------------------------------------------------------------
Runs everything in one command:
  1. Main experiment: all 5 datasets with DQN (15 episodes each)
  2. RL comparison: 7 algorithms on MNIST (15 episodes each)
  3. Generates all plots

Usage:
    python3 run_complete_experiment.py
    python3 run_complete_experiment.py --episodes 15
    python3 run_complete_experiment.py --episodes 3 --skip-main    (RL comparison only)
    python3 run_complete_experiment.py --episodes 3 --skip-rl      (main experiment only)
    python3 run_complete_experiment.py --rl-dataset MNIST           (RL comparison on specific dataset)
"""

import argparse
import subprocess
import sys
import os
import time


def run_cmd(cmd, desc):
    """Run a command and stream output in real time."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")

    proc = subprocess.Popen(
        cmd, stdout=sys.stdout, stderr=sys.stderr,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env={**os.environ}
    )
    proc.wait()
    success = proc.returncode == 0
    if success:
        print(f"\n  ✓ {desc} — COMPLETE")
    else:
        print(f"\n  ✗ {desc} — FAILED (exit code {proc.returncode})")
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Complete Agents Experiment Pipeline')
    parser.add_argument('--episodes', type=int, default=15,
        help='Episodes per dataset (default: 15)')
    parser.add_argument('--rpc', type=str,
        default=os.environ.get('RPC_URL', 'http://127.0.0.1:8545'))
    parser.add_argument('--skip-main', action='store_true',
        help='Skip main experiment (datasets)')
    parser.add_argument('--skip-rl', action='store_true',
        help='Skip RL algorithm comparison')
    parser.add_argument('--skip-plots', action='store_true',
        help='Skip plot generation')
    parser.add_argument('--rl-dataset', type=str, default='MNIST',
        help='Dataset for RL comparison (default: MNIST)')
    parser.add_argument('--rl-algorithms', type=str, default=None,
        help='Comma-separated algorithms (default: all 7)')
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("  Agents — Complete Experiment Pipeline")
    print(f"  RPC:           {args.rpc}")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Main exp:      {'SKIP' if args.skip_main else 'All 5 datasets'}")
    print(f"  RL comparison: {'SKIP' if args.skip_rl else f'7 algorithms on {args.rl_dataset}'}")
    print("=" * 60)

    # ── Step 1: Main experiment (all 5 datasets with DQN) ─────
    if not args.skip_main:
        cmd = ['python3', 'main_fisco_real.py',
               '--rpc', args.rpc,
               '--episodes', str(args.episodes)]
        run_cmd(cmd, f"Main Experiment — All 5 Datasets × {args.episodes} Episodes (DQN)")

    # ── Step 2: RL algorithm comparison ───────────────────────
    if not args.skip_rl:
        cmd = ['python3', 'rl_comparison.py',
               '--rpc', args.rpc,
               '--dataset', args.rl_dataset,
               '--episodes', str(args.episodes)]
        if args.rl_algorithms:
            cmd += ['--algorithms', args.rl_algorithms]
        run_cmd(cmd, f"RL Comparison — 7 Algorithms on {args.rl_dataset} × {args.episodes} Episodes")

    # ── Step 3: Generate plots ────────────────────────────────
    if not args.skip_plots:
        # Main experiment plots
        if os.path.exists('results/fisco_real_results.json'):
            run_cmd(['python3', 'plot_results.py'], "Generating Main Experiment Plots")
        else:
            print("\n  ~ Skipping main plots (no results file)")

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t0
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)

    print(f"\n{'='*60}")
    print(f"  ALL COMPLETE!")
    print(f"  Total time: {hours}h {mins}m")
    print(f"{'='*60}")
    print(f"  Results:")
    for f in ['results/fisco_real_results.json', 'results/rl_comparison.json']:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024
            print(f"    ✓ {f} ({size:.0f} KB)")
    print(f"  Plots:")
    if os.path.exists('plots'):
        for f in sorted(os.listdir('plots')):
            if f.endswith('.png'):
                size = os.path.getsize(f'plots/{f}') / 1024
                print(f"    ✓ plots/{f} ({size:.0f} KB)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

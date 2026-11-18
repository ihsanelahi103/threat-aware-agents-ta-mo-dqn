"""
run_all.py — Run complete Agents experiment in Docker
"""
import subprocess, sys, os, time, json
import urllib.request


def run_cmd(cmd, desc):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}\n")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr,
        cwd=os.path.dirname(os.path.abspath(__file__)), env={**os.environ})
    proc.wait()
    return proc.returncode == 0


def wait_for_rpc(rpc_url, timeout=120):
    print(f"  Waiting for FISCO BCOS at {rpc_url}...")
    payload = json.dumps({"jsonrpc":"2.0","method":"getBlockNumber","params":[],"id":1}).encode()
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = urllib.request.Request(rpc_url, data=payload,
                headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(r, timeout=5)
            data = json.loads(resp.read())
            if "jsonrpc" in data or "result" in data:
                print(f"\n  ✓ RPC ready!")
                return True
        except: pass
        time.sleep(2)
        print("  .", end="", flush=True)
    print(f"\n  ✗ RPC not available after {timeout}s")
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=15)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--rpc', type=str, default=os.environ.get('RPC_URL', 'http://172.25.0.10:8545'))
    args = parser.parse_args()
    t0 = time.time()

    print("="*60)
    print("  Agents — Complete Docker Experiment")
    print(f"  RPC:      {args.rpc}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Datasets: {args.dataset or 'ALL 5'}")
    print("="*60)

    if not wait_for_rpc(args.rpc):
        print("[ERROR] FISCO BCOS not reachable.")
        sys.exit(1)

    cmd = ['python3', 'main_fisco_real.py', '--rpc', args.rpc, '--episodes', str(args.episodes)]
    if args.dataset: cmd += ['--dataset', args.dataset]
    if not run_cmd(cmd, "Running Agents Experiment"):
        print("[ERROR] Experiment failed!"); sys.exit(1)

    run_cmd(['python3', 'plot_results.py'], "Generating Plots")

    print(f"\n{'='*60}")
    print(f"  ALL COMPLETE! Total time: {(time.time()-t0)/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
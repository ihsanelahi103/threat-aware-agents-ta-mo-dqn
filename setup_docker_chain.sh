#!/bin/bash
# ============================================================
# setup_docker_chain.sh
# ============================================================
# Prepares FISCO BCOS v3 chain data for Docker containers.
#
# This script:
#   1. Copies your existing node configs into chain_data/
#   2. Updates IPs from 127.0.0.1 to Docker network IPs
#   3. Copies the fisco-bcos binary to bin/
#   4. Cleans old blockchain data for fresh start
#
# Prerequisites:
#   - Existing FISCO BCOS node configs at:
#     ~/edge_lab/Agents_FISCO/fisco_nodes/nodes/127.0.0.1/
#
# Usage:
#   chmod +x setup_docker_chain.sh
#   ./setup_docker_chain.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Source node directory (your existing setup)
SRC_NODES="$HOME/edge_lab/Agents_FISCO/fisco_nodes/nodes/127.0.0.1"

echo "============================================================"
echo "  Agents Docker — Chain Setup"
echo "============================================================"

# ── Step 1: Check source nodes exist ─────────────────────────
echo ""
echo "[1] Checking existing FISCO BCOS nodes..."

if [ ! -d "$SRC_NODES/node0" ]; then
    echo "  ✗ Node configs not found at: $SRC_NODES"
    echo "  Make sure you have run the native setup first."
    exit 1
fi
echo "  ✓ Found nodes at: $SRC_NODES"

# ── Step 2: Copy fisco-bcos binary ───────────────────────────
echo ""
echo "[2] Copying fisco-bcos binary..."

mkdir -p bin
if [ -f "$SRC_NODES/fisco-bcos" ]; then
    cp "$SRC_NODES/fisco-bcos" bin/fisco-bcos
    chmod +x bin/fisco-bcos
    echo "  ✓ Binary copied to bin/fisco-bcos"
    echo "  Version: $(bin/fisco-bcos --version 2>&1 | head -1)"
else
    echo "  ✗ fisco-bcos binary not found at $SRC_NODES/fisco-bcos"
    exit 1
fi

# ── Step 3: Copy and adapt node configs ──────────────────────
echo ""
echo "[3] Copying and adapting node configs for Docker..."

# Docker network IPs
DOCKER_IPS=("172.25.0.10" "172.25.0.11" "172.25.0.12" "172.25.0.13")
PORTS=("8545" "8546" "8547" "8548")
P2P_PORTS=("30300" "30301" "30302" "30303")

for i in 0 1 2 3; do
    NODE_SRC="$SRC_NODES/node$i"
    NODE_DST="chain_data/node$i"
    
    # Clean and create destination
    rm -rf "$NODE_DST"
    mkdir -p "$NODE_DST"
    
    # Copy config files
    cp "$NODE_SRC/config.genesis" "$NODE_DST/"
    cp "$NODE_SRC/config.ini" "$NODE_DST/"
    
    # Copy node identity files
    if [ -d "$NODE_SRC/conf" ]; then
        cp -r "$NODE_SRC/conf" "$NODE_DST/"
    fi
    
    # Copy nodes.json (peer discovery)
    if [ -f "$NODE_SRC/nodes.json" ]; then
        cp "$NODE_SRC/nodes.json" "$NODE_DST/"
    fi
    
    # Create fresh data directory
    mkdir -p "$NODE_DST/data"
    
    # Update config.ini: replace 127.0.0.1 with Docker IPs
    # RPC listen address — bind to 0.0.0.0 so Docker port mapping works
    sed -i "s|listen_ip=127.0.0.1|listen_ip=0.0.0.0|g" "$NODE_DST/config.ini"
    sed -i "s|listen_ip = 127.0.0.1|listen_ip = 0.0.0.0|g" "$NODE_DST/config.ini"
    
    # Update the RPC port to match this node's assigned port
    # (config.ini usually has the correct port already from build_chain.sh)
    
    echo "  ✓ node$i → ${DOCKER_IPS[$i]}:${PORTS[$i]} (P2P: ${P2P_PORTS[$i]})"
done

# Update nodes.json for Docker networking
# Replace 127.0.0.1:3030X with Docker IPs
for i in 0 1 2 3; do
    NODE_DST="chain_data/node$i"
    if [ -f "$NODE_DST/nodes.json" ]; then
        for j in 0 1 2 3; do
            OLD_PORT=$((30300 + j))
            sed -i "s|127.0.0.1:${OLD_PORT}|${DOCKER_IPS[$j]}:${P2P_PORTS[$j]}|g" "$NODE_DST/nodes.json"
        done
    fi
done

echo ""
echo "  ✓ All node configs adapted for Docker network"

# ── Step 4: Copy SDK certs ───────────────────────────────────
echo ""
echo "[4] Copying SDK certificates..."

if [ -d "$SRC_NODES/sdk" ]; then
    mkdir -p chain_data/sdk
    cp -r "$SRC_NODES/sdk/"* chain_data/sdk/
    echo "  ✓ SDK certs copied to chain_data/sdk/"
fi

# ── Step 5: Create output directories ────────────────────────
echo ""
echo "[5] Creating output directories..."
mkdir -p results plots
echo "  ✓ results/ and plots/ created"

# ── Step 6: Verify ───────────────────────────────────────────
echo ""
echo "[6] Verification..."
echo "  Directory structure:"
echo "  $(pwd)/"
echo "  ├── bin/"
echo "  │   └── fisco-bcos ($(du -h bin/fisco-bcos | cut -f1))"
echo "  ├── chain_data/"
for i in 0 1 2 3; do
    CONF_COUNT=$(ls chain_data/node$i/ 2>/dev/null | wc -l)
    echo "  │   ├── node$i/ ($CONF_COUNT items)"
done
echo "  │   └── sdk/"
echo "  ├── docker-compose.yml"
echo "  ├── Dockerfile"
echo "  ├── main_fisco_real.py"
echo "  ├── deploy_contract_python.py"
echo "  ├── plot_results.py"
echo "  ├── run_all.py"
echo "  ├── results/"
echo "  └── plots/"

echo ""
echo "============================================================"
echo "  SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Start the blockchain nodes:"
echo "     docker compose up -d fisco-node0 fisco-node1 fisco-node2 fisco-node3"
echo ""
echo "  2. Wait ~15 seconds, then check they're running:"
echo "     docker compose ps"
echo "     docker compose logs fisco-node0 --tail 5"
echo ""
echo "  3. Build and start the experiment container:"
echo "     docker compose up -d Agents"
echo ""
echo "  4. Run the full experiment (all 5 datasets, 15 episodes):"
echo "     docker compose exec Agents python3 run_all.py --episodes 15"
echo ""
echo "  5. Or run a quick test first:"
echo "     docker compose exec Agents python3 run_all.py --dataset MNIST --episodes 3"
echo ""
echo "  6. View results:"
echo "     cat results/fisco_real_results.json"
echo "     ls plots/"
echo ""
echo "  7. Stop everything:"
echo "     docker compose down"
echo ""
echo "============================================================"

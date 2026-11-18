"""
deploy_contract_python.py — FISCO BCOS v3 web3_rpc (Ethereum-compatible)
-------------------------------------------------------------------------
Uses standard web3.eth.* Ethereum methods via FISCO web3_rpc endpoint.
"""

import json
import os
import sys
import time

from web3 import Web3
from eth_account import Account

try:
    import solcx
except ImportError:
    print("[ERROR] py-solc-x not installed. Run: pip install py-solc-x")
    sys.exit(1)

RPC_URL       = os.environ.get("RPC_URL", "http://172.25.0.10:8545")
CONTRACT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "contracts", "EdgeCommunication.sol")
OUTPUT_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "contract_address.txt")
ABI_OUTPUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "EdgeCommunication_abi.json")
KEY_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "deployer_key.txt")
SOLC_VERSION  = "0.6.10"
FISCO_CHAIN_ID = 20200


def install_solc():
    print(f"  Installing solc {SOLC_VERSION}...")
    solcx.install_solc(SOLC_VERSION, show_progress=True)
    solcx.set_solc_version(SOLC_VERSION)
    print(f"  solc {SOLC_VERSION} ready")


def compile_contract(source):
    print("  Compiling EdgeCommunication.sol...")
    try:
        solcx.set_solc_version(SOLC_VERSION)
    except Exception:
        install_solc()
    compiled = solcx.compile_source(source, output_values=["abi","bin"],
        solc_version=SOLC_VERSION, optimize=True, optimize_runs=200)
    contract_id = next((k for k in compiled if "EdgeCommunication" in k), None)
    if not contract_id:
        raise ValueError(f"Contract not found. Keys: {list(compiled.keys())}")
    abi = compiled[contract_id]["abi"]
    bytecode = compiled[contract_id]["bin"]
    print(f"  Compiled: {len(bytecode)//2} bytes, {len(abi)} ABI entries")
    return abi, bytecode


def deploy(w3, abi, bytecode, deployer, private_key):
    print(f"  Deploying from: {deployer}")
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(deployer)
    tx = contract.constructor().build_transaction({
        "from": deployer, "gas": 3_000_000,
        "gasPrice": w3.to_wei("300","gwei"),
        "nonce": nonce, "chainId": FISCO_CHAIN_ID,
    })
    signed  = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  TX: {tx_hash.hex()}")
    print("  Waiting for confirmation...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    address = receipt.contractAddress
    if not address:
        raise ValueError(f"Deploy failed. Receipt: {dict(receipt)}")
    print(f"  Deployed at: {address}")
    print(f"  Block: #{receipt.blockNumber}  Gas: {receipt.gasUsed:,}")
    return address


def main():
    print("="*60)
    print("  EdgeCommunication.sol — Python Deployer (web3_rpc)")
    print(f"  Target: {RPC_URL}")
    print("="*60)

    print("\n[1] Connecting...")
    w3 = Web3(Web3.HTTPProvider(RPC_URL, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print(f"  Cannot connect to {RPC_URL}"); sys.exit(1)
    print(f"  Connected! Block #{w3.eth.block_number} Chain ID: {w3.eth.chain_id}")

    print("\n[2] Loading deployer account...")
    if not os.path.exists(KEY_FILE):
        print(f"  {KEY_FILE} not found"); sys.exit(1)
    with open(KEY_FILE) as f:
        private_key = f.read().strip()
    deployer = Account.from_key(private_key).address
    print(f"  Deployer: {deployer}")

    print("\n[3] Loading contract...")
    with open(CONTRACT_FILE) as f:
        source = f.read()
    print(f"  Loaded: {CONTRACT_FILE}")

    print("\n[4] Setting up solc...")
    installed = [str(v) for v in solcx.get_installed_solc_versions()]
    if SOLC_VERSION not in installed and f"v{SOLC_VERSION}" not in installed:
        install_solc()
    else:
        solcx.set_solc_version(SOLC_VERSION)
        print(f"  solc {SOLC_VERSION} ready")

    print("\n[5] Compiling...")
    abi, bytecode = compile_contract(source)
    with open(ABI_OUTPUT, "w") as f:
        json.dump(abi, f, indent=2)
    print(f"  ABI saved: {ABI_OUTPUT}")

    print("\n[6] Deploying...")
    address = deploy(w3, abi, bytecode, deployer, private_key)

    print("\n[7] Verifying...")
    try:
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(address), abi=abi)
        total = contract.functions.totalAgents().call({"from": deployer})
        print(f"  Verified: totalAgents() = {total}")
    except Exception as e:
        print(f"  Confirmed (call skipped: {e})")

    with open(OUTPUT_FILE, "w") as f:
        f.write(address)
    print(f"\n  Address saved: {OUTPUT_FILE}")
    print("\n"+"="*60)
    print("  DEPLOYMENT COMPLETE")
    print(f"  Contract: {address}")
    print("="*60)
    return address

if __name__ == "__main__":
    main()
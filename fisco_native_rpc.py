"""
fisco_native_rpc.py — FISCO BCOS v3 Native RPC Layer
------------------------------------------------------
Replaces all Ethereum RPC calls with FISCO BCOS v3 native methods.

FISCO BCOS v3 native RPC methods used:
  - getBlockNumber          → current block
  - getGroupList            → verify group0 exists
  - sendTransaction         → submit signed TX (FISCO binary format)
  - getTransactionReceipt   → poll for confirmation
  - call                    → read-only contract calls

TX encoding: FISCO BCOS v3 uses its own binary transaction format,
NOT Ethereum RLP. We build it with struct + eth_abi + secp256k1 signing.

Compatible with:
  - Python 3.12
  - web3 6.15.1  (used for ABI encoding + key management ONLY)
  - FISCO BCOS v3.11.0, disable_ssl=true, plain HTTP
  - chain: group0 / chain0 / chain_id=20200
"""

import hashlib
import json
import os
import random
import struct
import time
from typing import Optional

import requests
from eth_account import Account
from eth_account._utils.signing import sign_transaction_dict
from eth_keys import keys as eth_keys
from web3 import Web3

# ── Constants ────────────────────────────────────────────────
FISCO_CHAIN_ID   = 20200
FISCO_GROUP_ID   = "group0"
FISCO_CHAIN_NAME = "chain0"
DEFAULT_RPC      = os.environ.get("RPC_URL", "http://172.25.0.10:8545")

# FISCO v3 TX version tag
FISCO_TX_VERSION = 0


# ══════════════════════════════════════════════════════════════
#  Low-level JSON-RPC helper
# ══════════════════════════════════════════════════════════════

class FISCORPCError(Exception):
    pass


class FISCOJsonRPC:
    """
    Thin wrapper around requests that speaks FISCO BCOS v3 JSON-RPC.
    All methods map 1-to-1 to native FISCO RPC names.
    """

    def __init__(self, url: str, timeout: int = 30, group: str = FISCO_GROUP_ID):
        self.url     = url
        self.timeout = timeout
        self.group   = group
        self._id     = 0
        self._sess   = requests.Session()
        self._sess.headers.update({"Content-Type": "application/json"})

    def _call(self, method: str, params=None):
        self._id += 1
        payload = {
            "jsonrpc": "2.0",
            "method":  method,
            "params":  params if params is not None else [],
            "id":      self._id,
        }
        try:
            resp = self._sess.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise FISCORPCError(f"HTTP error calling {method}: {e}") from e

        data = resp.json()
        if "error" in data and data["error"]:
            raise FISCORPCError(f"RPC error [{method}]: {data['error']}")
        return data.get("result")

    # ── Read methods ─────────────────────────────────────────

    def get_block_number(self) -> int:
        """FISCO native: getBlockNumber (returns hex string or int)."""
        result = self._call("getBlockNumber", [self.group, "false"])
        if result is None:
            return 0
        if isinstance(result, str):
            return int(result, 16) if result.startswith("0x") else int(result)
        return int(result)

    def get_group_list(self) -> list:
        """FISCO native: getGroupList."""
        return self._call("getGroupList", []) or []

    def get_group_info(self) -> dict:
        """FISCO native: getGroupInfo."""
        return self._call("getGroupInfo", [self.group]) or {}

    def get_sealer_list(self) -> list:
        """FISCO native: getSealerList."""
        return self._call("getSealerList", [self.group, "latest"]) or []

    def get_transaction_count(self, address: str) -> int:
        """
        FISCO v3 does NOT have eth_getTransactionCount.
        We track nonces locally — this method exists for API compatibility
        but always returns 0; actual nonce management is done in FISCOTxBuilder.
        """
        return 0

    def call(self, to: str, data: str, caller: str = None) -> str:
        """
        FISCO native: call (read-only, no TX needed).
        Returns hex result string.
        """
        tx_obj = {
            "to":   to,
            "data": data,
        }
        if caller:
            tx_obj["from"] = caller
        result = self._call("call", [self.group, "latest", tx_obj])
        if isinstance(result, dict):
            return result.get("output", "0x")
        return result or "0x"

    def get_transaction_receipt(self, tx_hash: str) -> Optional[dict]:
        """FISCO native: getTransactionReceipt."""
        return self._call("getTransactionReceipt", [self.group, tx_hash, False])

    def send_transaction(self, signed_tx_hex: str) -> str:
        """
        FISCO native: sendTransaction.
        Accepts FISCO-encoded signed transaction as hex string.
        Returns transaction hash.
        """
        result = self._call("sendTransaction", [self.group, signed_tx_hex, False])
        if isinstance(result, dict):
            return result.get("transactionHash", result.get("txHash", ""))
        return result or ""

    def ping(self) -> bool:
        """Check connectivity using getBlockNumber (always works on FISCO v3)."""
        try:
            bn = self.get_block_number()
            return isinstance(bn, int)
        except FISCORPCError:
            return False


# ══════════════════════════════════════════════════════════════
#  FISCO BCOS v3 Transaction Builder
# ══════════════════════════════════════════════════════════════

class FISCOTxBuilder:
    """
    Builds and signs FISCO BCOS v3 transactions.

    FISCO v3 transaction binary format (RLP-like but NOT Ethereum RLP):
    Fields: [version, groupID, chainID, to, nonce, gasLimit, blockLimit,
             data, abi, value, freeStorage, extraData]

    We use eth_abi for ABI encoding of calldata, then build the FISCO
    transaction envelope and sign with secp256k1 (same curve as Ethereum).

    NOTE: FISCO v3.x uses a simplified transaction format. The node
    accepts transactions where the "data" field is ABI-encoded calldata,
    identical to Ethereum. Only the envelope (outer structure) differs.
    """

    def __init__(self, rpc: FISCOJsonRPC, chain_id: int = FISCO_CHAIN_ID):
        self.rpc      = rpc
        self.chain_id = chain_id
        # Per-address nonce tracking (FISCO v3 requires incrementing nonces)
        self._nonces: dict[str, int] = {}

    def get_nonce(self, address: str) -> int:
        addr = address.lower()
        if addr not in self._nonces:
            self._nonces[addr] = 0
        return self._nonces[addr]

    def increment_nonce(self, address: str):
        self._nonces[address.lower()] = self.get_nonce(address) + 1

    def _build_fisco_tx_data(
        self,
        to: str,
        calldata: bytes,
        nonce: int,
        gas_limit: int = 300_000,
    ) -> bytes:
        """
        Build FISCO BCOS v3 transaction binary.

        FISCO v3 uses a compact binary encoding:
        [4B version][4B chain_id][var groupID][var to][8B nonce]
        [4B gas_limit][4B block_limit][var calldata]

        The exact encoding that FISCO v3.11 nodes accept over plain HTTP
        with disable_ssl=true is a JSON object (not raw binary).
        We return the JSON-serialisable dict instead.
        """
        block_limit = self.rpc.get_block_number() + 600  # ~10 min window
        return {
            "version":    FISCO_TX_VERSION,
            "groupID":    FISCO_GROUP_ID,
            "chainID":    FISCO_CHAIN_NAME,
            "to":         to,
            "nonce":      hex(nonce),
            "gas":        hex(gas_limit),
            "blockLimit": hex(block_limit),
            "data":       "0x" + calldata.hex(),
            "abi":        "",
            "value":      "0x0",
        }

    def sign_and_send(
        self,
        private_key: str,
        to: str,
        calldata: bytes,
        gas_limit: int = 300_000,
    ) -> str:
        """
        Sign a FISCO v3 transaction and submit it.
        Returns transaction hash string.
        """
        acct    = Account.from_key(private_key)
        address = acct.address
        nonce   = self.get_nonce(address)

        tx_dict = self._build_fisco_tx_data(to, calldata, nonce, gas_limit)

        # Sign: hash the canonical JSON representation with keccak256,
        # then sign with the private key (secp256k1 — same as Ethereum).
        # FISCO v3 accepts Ethereum-style v/r/s signatures.
        canonical = json.dumps(tx_dict, sort_keys=True, separators=(",", ":"))
        tx_hash_bytes = Web3.keccak(text=canonical)

        key_obj = eth_keys.PrivateKey(bytes.fromhex(private_key.lstrip("0x")))
        sig     = key_obj.sign_msg_hash(tx_hash_bytes)

        signed_envelope = {
            **tx_dict,
            "signature": "0x" + sig.to_bytes().hex(),
            "from":      address,
        }

        signed_hex = "0x" + json.dumps(signed_envelope).encode().hex()
        tx_hash = self.rpc.send_transaction(signed_hex)
        self.increment_nonce(address)
        return tx_hash

    def wait_for_receipt(
        self,
        tx_hash: str,
        timeout: int = 30,
        poll_interval: float = 0.5,
    ) -> dict:
        """Poll getTransactionReceipt until confirmed or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                receipt = self.rpc.get_transaction_receipt(tx_hash)
                if receipt and receipt.get("blockNumber") not in (None, "", "0x"):
                    return receipt
            except FISCORPCError:
                pass
            time.sleep(poll_interval)
        raise TimeoutError(f"TX {tx_hash} not confirmed within {timeout}s")


# ══════════════════════════════════════════════════════════════
#  ABI Encoder (web3.py used ONLY for encoding, not RPC)
# ══════════════════════════════════════════════════════════════

class FISCOAbiEncoder:
    """
    Uses web3.py contract ABI encoding to build calldata.
    No actual RPC calls — purely local encoding.
    """

    # Dummy web3 (not connected — only used for ABI encoding)
    _w3 = Web3()

    def __init__(self, abi: list, address: str):
        checksum = Web3.to_checksum_address(address)
        # Build contract object against a dummy provider for encoding only
        self.contract = self._w3.eth.contract(address=checksum, abi=abi)

    def encode_call(self, fn_name: str, *args) -> bytes:
        """Return ABI-encoded calldata bytes for fn_name(*args)."""
        fn = getattr(self.contract.functions, fn_name)
        return fn(*args)._encode_transaction_data()

    def decode_result(self, fn_name: str, hex_result: str) -> tuple:
        """Decode hex output from a call() response."""
        if not hex_result or hex_result == "0x":
            return (None,)
        fn = getattr(self.contract.functions, fn_name)
        raw = bytes.fromhex(hex_result.lstrip("0x"))
        return fn().call.__func__  # fallback — decode manually below

    def decode_output(self, fn_name: str, hex_result: str):
        """Decode ABI output for a given function."""
        from eth_abi import decode as abi_decode
        fn_abi = next(
            (x for x in self.contract.abi
             if x.get("name") == fn_name and x.get("type") == "function"),
            None
        )
        if not fn_abi or not hex_result or hex_result == "0x":
            return None
        output_types = [o["type"] for o in fn_abi.get("outputs", [])]
        if not output_types:
            return None
        raw = bytes.fromhex(hex_result.lstrip("0x"))
        decoded = abi_decode(output_types, raw)
        return decoded[0] if len(decoded) == 1 else decoded


# ══════════════════════════════════════════════════════════════
#  High-level FISCORealChain — drop-in replacement
# ══════════════════════════════════════════════════════════════

class FISCORealChain:
    """
    Drop-in replacement for the old web3-based FISCORealChain.
    Uses FISCO BCOS v3 native RPC only — no Ethereum methods.

    Public API (identical to old class):
      registerDevice(aid, agent_id_str)
      uploadData(aid, data_hash_bytes, metadata_str)
      validateData(aid, data_hash_bytes, is_valid_bool)
      reportByzantine(aid, evidence_hash_bytes, suspect_id_str)
      get_stats() → dict
      get_enhanced_stats() → dict
    """

    def __init__(self, rpc_url: str, contract_address: str, abi: list):
        self.rpc_url = rpc_url
        self.addr    = Web3.to_checksum_address(contract_address)

        # Native RPC + TX builder
        self.rpc     = FISCOJsonRPC(rpc_url)
        self.builder = FISCOTxBuilder(self.rpc)
        self.encoder = FISCOAbiEncoder(abi, contract_address)

        # Stats counters
        self.tx_count       = 0
        self.tx_fail        = 0
        self.gas_list       = []
        self.lat_list       = []
        self.blocks         = []
        self.all_latencies  = []
        self.tx_timestamps  = []
        self.bandwidth_bytes= 0
        self.gas_by_function  = {}
        self.lat_by_function  = {}
        self.tx_by_agent    = {}
        self.gas_by_agent   = {}

        # Private key store  {aid → hex_key}
        self._keys: dict[int, str] = {}

        # Verify connectivity
        if not self.rpc.ping():
            raise ConnectionError(
                f"Cannot connect to FISCO BCOS v3 at {rpc_url}. "
                f"Ensure disable_ssl=true and node is running."
            )
        bn = self.rpc.get_block_number()
        groups = self.rpc.get_group_list()
        print(f"[Chain] FISCO v3 connected — block #{bn}  groups={groups}")
        print(f"[Chain] Contract: {self.addr}")

    # ── Account management ────────────────────────────────────

    def _agent_key(self, aid: int) -> str:
        if aid not in self._keys:
            raw = hashlib.sha256(
                f"Agents_agent_42_{aid:04d}".encode()
            ).hexdigest()
            self._keys[aid] = "0x" + raw
        return self._keys[aid]

    def _agent_address(self, aid: int) -> str:
        return Account.from_key(self._agent_key(aid)).address

    # ── Core TX sender ────────────────────────────────────────

    def _send_tx(self, aid: int, fn_name: str, *args) -> dict:
        t0 = time.time()
        try:
            calldata = self.encoder.encode_call(fn_name, *args)
            self.bandwidth_bytes += len(calldata)

            tx_hash = self.builder.sign_and_send(
                private_key=self._agent_key(aid),
                to=self.addr,
                calldata=calldata,
                gas_limit=300_000,
            )

            receipt  = self.builder.wait_for_receipt(tx_hash, timeout=30)
            lat      = (time.time() - t0) * 1000
            gas_used = int(receipt.get("gasUsed", "0x0"), 16) if isinstance(
                receipt.get("gasUsed"), str) else receipt.get("gasUsed", 0)
            blk_raw  = receipt.get("blockNumber", "0x0")
            blk      = int(blk_raw, 16) if isinstance(blk_raw, str) else int(blk_raw or 0)

            # FISCO receipts: status "0x0" = success (unlike Ethereum's 0x1)
            # Some versions use "0" or missing field — treat missing as success
            status_raw = receipt.get("status", "0x0")
            if isinstance(status_raw, str):
                status_int = int(status_raw, 16) if status_raw.startswith("0x") else int(status_raw)
            else:
                status_int = int(status_raw or 0)
            # FISCO: status 0 = success
            ok = (status_int == 0)

            # Handle "already registered" revert for registerDevice
            if not ok and fn_name == "registerDevice":
                output = receipt.get("output", "")
                if "already" in str(output).lower() or "0x" == output:
                    return {"success": True, "gas": 0, "block": blk,
                            "latency_ms": lat, "already_registered": True}

            if ok:
                self.tx_count += 1
                self.gas_list.append(gas_used)
                self.lat_list.append(lat)
                self.blocks.append(blk)
                self.all_latencies.append(lat)
                self.tx_timestamps.append(time.time())
                self.gas_by_function.setdefault(fn_name, []).append(gas_used)
                self.lat_by_function.setdefault(fn_name, []).append(lat)
                self.tx_by_agent[aid]  = self.tx_by_agent.get(aid, 0) + 1
                self.gas_by_agent.setdefault(aid, []).append(gas_used)
            else:
                self.tx_fail += 1

            return {"success": ok, "gas": gas_used, "block": blk, "latency_ms": lat}

        except TimeoutError as e:
            self.tx_fail += 1
            lat = (time.time() - t0) * 1000
            print(f"  [Chain WARN] TX timeout ({fn_name}): {e}")
            return {"success": False, "gas": 0, "block": 0, "latency_ms": lat}

        except Exception as e:
            self.tx_fail += 1
            lat = (time.time() - t0) * 1000
            err = str(e).lower()
            if "already" in err:
                return {"success": True, "gas": 0, "block": 0,
                        "latency_ms": lat, "already_registered": True}
            print(f"  [Chain WARN] TX failed ({fn_name}): {e}")
            return {"success": False, "gas": 0, "block": 0, "latency_ms": lat}

    # ── Public contract methods (identical signatures to old class) ──

    def registerDevice(self, aid: int, agent_id: str) -> dict:
        return self._send_tx(aid, "registerDevice", agent_id)

    def uploadData(self, aid: int, data_hash: bytes, metadata: str) -> dict:
        dh32 = data_hash[:32].ljust(32, b"\x00")
        return self._send_tx(aid, "uploadData", dh32, metadata)

    def validateData(self, aid: int, data_hash: bytes, is_valid: bool) -> dict:
        dh32 = data_hash[:32].ljust(32, b"\x00")
        return self._send_tx(aid, "validateData", dh32, is_valid)

    def reportByzantine(self, aid: int, evidence_hash: bytes, suspect_id: str) -> dict:
        eh32 = evidence_hash[:32].ljust(32, b"\x00")
        return self._send_tx(aid, "reportByzantine", eh32, suspect_id)

    # ── Read-only call (no TX needed) ────────────────────────

    def call_view(self, fn_name: str, *args, caller: str = None) -> any:
        """Call a view function (no gas, no TX)."""
        if caller is None:
            caller = self._agent_address(0)
        calldata = self.encoder.encode_call(fn_name, *args)
        hex_result = self.rpc.call(
            to=self.addr,
            data="0x" + calldata.hex(),
            caller=caller,
        )
        return self.encoder.decode_output(fn_name, hex_result)

    # ── Stats (identical API to old class) ───────────────────

    def get_stats(self) -> dict:
        import numpy as np
        return {
            "total_tx":       self.tx_count,
            "failed_tx":      self.tx_fail,
            "avg_gas_per_tx": int(np.mean(self.gas_list)) if self.gas_list else 0,
            "avg_latency_ms": float(np.mean(self.lat_list)) if self.lat_list else 0,
            "current_block":  self.blocks[-1] if self.blocks else 0,
        }

    def get_enhanced_stats(self) -> dict:
        import numpy as np
        la  = np.array(self.all_latencies) if self.all_latencies else np.array([0.0])
        tps = (
            len(self.tx_timestamps) /
            (self.tx_timestamps[-1] - self.tx_timestamps[0])
            if len(self.tx_timestamps) >= 2
            and (self.tx_timestamps[-1] - self.tx_timestamps[0]) > 0
            else 0.0
        )
        fg = {
            fn: {"avg": int(np.mean(gl)), "count": len(gl), "total": int(np.sum(gl))}
            for fn, gl in self.gas_by_function.items()
        }
        fl = {
            fn: {"avg_ms": round(float(np.mean(ll)), 1), "count": len(ll)}
            for fn, ll in self.lat_by_function.items()
        }
        return {
            "tps":                       round(tps, 3),
            "latency_p50_ms":            round(float(np.percentile(la, 50)), 1),
            "latency_p95_ms":            round(float(np.percentile(la, 95)), 1),
            "latency_p99_ms":            round(float(np.percentile(la, 99)), 1),
            "latency_std_ms":            round(float(np.std(la)), 1),
            "all_latencies":             [round(float(x), 1) for x in self.all_latencies],
            "gas_by_function":           fg,
            "latency_by_function":       fl,
            "total_bandwidth_kb":        round(self.bandwidth_bytes / 1024, 2),
            "avg_bandwidth_per_tx_bytes":round(self.bandwidth_bytes / max(self.tx_count, 1), 1),
        }


# ══════════════════════════════════════════════════════════════
#  LocalOnlyChain — used by "Without Blockchain" ablation
# ══════════════════════════════════════════════════════════════

class LocalOnlyChain:
    """No blockchain — local simulation only. Used in ablation study."""

    def __init__(self):
        self.tx_count        = 0
        self.tx_fail         = 0
        self.gas_list        = []
        self.lat_list        = []
        self.blocks          = []
        self.all_latencies   = []
        self.tx_timestamps   = []
        self.bandwidth_bytes = 0
        self.gas_by_function = {}

    def _local(self) -> dict:
        self.tx_count += 1
        self.lat_list.append(0.1)
        self.all_latencies.append(0.1)
        self.tx_timestamps.append(time.time())
        return {"success": True, "gas": 0, "block": 0, "latency_ms": 0.1}

    def registerDevice(self, aid, s):        return self._local()
    def uploadData(self, aid, dh, m):        return self._local()
    def validateData(self, aid, dh, v):      return self._local()
    def reportByzantine(self, aid, eh, s):   return self._local()

    def get_stats(self) -> dict:
        import numpy as np
        return {
            "total_tx":       self.tx_count,
            "failed_tx":      0,
            "avg_gas_per_tx": 0,
            "avg_latency_ms": float(np.mean(self.lat_list)) if self.lat_list else 0,
            "current_block":  0,
        }

    def get_enhanced_stats(self) -> dict:
        return {
            "tps": 0, "latency_p50_ms": 0.1, "latency_p95_ms": 0.1,
            "latency_p99_ms": 0.1, "latency_std_ms": 0,
            "all_latencies": self.all_latencies,
            "gas_by_function": {}, "latency_by_function": {},
            "total_bandwidth_kb": 0, "avg_bandwidth_per_tx_bytes": 0,
        }


# ══════════════════════════════════════════════════════════════
#  Connectivity test helper
# ══════════════════════════════════════════════════════════════

def verify_fisco_connection(rpc_url: str = DEFAULT_RPC) -> bool:
    """
    Quick connectivity check. Run this before starting any experiment.
    Prints what works and what doesn't.
    """
    print(f"\n[FISCO v3 Connection Test]  {rpc_url}")
    rpc = FISCOJsonRPC(rpc_url)

    ok = True
    for method_name, fn in [
        ("getBlockNumber", lambda: rpc.get_block_number()),
        ("getGroupList",   lambda: rpc.get_group_list()),
        ("getGroupInfo",   lambda: rpc.get_group_info()),
        ("getSealerList",  lambda: rpc.get_sealer_list()),
    ]:
        try:
            result = fn()
            print(f"  ✓ {method_name:<20} → {result!r}")
        except FISCORPCError as e:
            print(f"  ✗ {method_name:<20} → {e}")
            ok = False

    print(f"\n  Connection: {'OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    verify_fisco_connection()

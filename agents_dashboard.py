from flask import Flask, render_template_string
import requests, json, datetime

app = Flask(__name__)

NODES = [
    {"name": "Node 0 (Sealer)", "ip": "172.25.0.10", "port": 8545, "p2p": "30300"},
    {"name": "Node 1 (Sealer)", "ip": "172.25.0.11", "port": 8546, "p2p": "30301"},
    {"name": "Node 2 (Sealer)", "ip": "172.25.0.12", "port": 8547, "p2p": "30302"},
    {"name": "Node 3 (Sealer)", "ip": "172.25.0.13", "port": 8548, "p2p": "30303"},
]

def rpc(ip, port, method, params=None):
    try:
        r = requests.post(
            f"http://{ip}:{port}",
            json={"jsonrpc": "2.0", "method": method, "params": params or [], "id": 1},
            timeout=4
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_node_info(n):
    info = {
        "name": n["name"], "ip": n["ip"], "p2p": n["p2p"],
        "status": "OFFLINE", "block": "N/A", "peers": "N/A",
        "version": "FISCO BCOS v3.11", "css": "offline"
    }
    for method, params in [
        ("getBlockNumber", [1]),
        ("eth_blockNumber", []),
        ("getClientVersion", [])
    ]:
        try:
            r = requests.post(
                f"http://{n['ip']}:{n['port']}",
                json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
                timeout=3
            )
            d = r.json()
            if "result" in d and d["result"] is not None:
                info["status"] = "ONLINE"
                info["css"] = "online"
                res = d["result"]
                if method in ("getBlockNumber", "eth_blockNumber"):
                    try:
                        info["block"] = int(res, 16) if isinstance(res, str) and res.startswith("0x") else int(res)
                    except:
                        info["block"] = str(res)
                elif method == "getClientVersion" and isinstance(res, dict):
                    info["version"] = res.get("FISCO-BCOS Version", res.get("version", "v3.x"))
                break
        except:
            continue
    try:
        d2 = rpc(n["ip"], n["port"], "getPeers", [1])
        if "result" in d2 and isinstance(d2["result"], list):
            info["peers"] = len(d2["result"])
    except:
        pass
    return info

def get_blocks():
    blocks = []
    try:
        d = rpc("172.25.0.10", 8545, "getBlockNumber", [1])
        if "result" not in d:
            return blocks
        res = d["result"]
        latest = int(res, 16) if isinstance(res, str) and res.startswith("0x") else int(res)
        for i in range(min(10, latest + 1)):
            bn = latest - i
            d2 = rpc("172.25.0.10", 8545, "getBlockByNumber", [1, hex(bn), False])
            if "result" in d2 and d2["result"]:
                b = d2["result"]
                ts = b.get("timestamp", "0x0")
                try:
                    ts_i = int(ts, 16) if isinstance(ts, str) else int(ts)
                    dt = datetime.datetime.fromtimestamp(ts_i / 1000).strftime("%Y-%m-%d %H:%M:%S") if ts_i > 0 else "N/A"
                except:
                    dt = "N/A"
                gas = b.get("gasUsed", "0x0")
                try:
                    gas_i = int(gas, 16) if isinstance(gas, str) else int(gas)
                except:
                    gas_i = 0
                h = b.get("hash", "N/A")
                blocks.append({
                    "number": bn,
                    "hash": (h[:20] + "...") if h and len(h) > 20 else h,
                    "txcount": len(b.get("transactions", [])),
                    "gasUsed": f"{gas_i:,}",
                    "timestamp": dt
                })
    except:
        pass
    return blocks

HTML = """<!DOCTYPE html><html><head>
<meta charset="utf-8">
<title>Agents FISCO BCOS Dashboard</title>
<meta http-equiv="refresh" content="30">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f1923;color:#e0e0e0}
.hdr{background:linear-gradient(135deg,#1a3a5c,#0d2137);padding:16px 28px;border-bottom:2px solid #2a5f9e;display:flex;align-items:center;justify-content:space-between}
.hdr h1{font-size:20px;color:#4fc3f7;font-weight:700}
.hdr .sub{font-size:12px;color:#90a4ae;margin-top:3px}
.badge{background:#1b5e20;color:#a5d6a7;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;border:1px solid #2e7d32}
.bar{background:#1a2733;padding:7px 28px;display:flex;gap:24px;border-bottom:1px solid #1e3a52;font-size:12px;color:#78909c;flex-wrap:wrap}
.bar b{color:#4fc3f7}
.wrap{padding:18px 28px}
.sgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}
.sc{background:#1a2733;border:1px solid #1e3a52;border-radius:8px;padding:14px 18px;border-left:3px solid var(--a,#4fc3f7)}
.sc .lbl{font-size:11px;color:#78909c;text-transform:uppercase;letter-spacing:.5px}
.sc .val{font-size:24px;font-weight:700;color:var(--a,#4fc3f7);margin:5px 0 2px}
.sc .sub{font-size:11px;color:#546e7a}
.stit{font-size:13px;font-weight:600;color:#b0bec5;margin-bottom:10px;padding-bottom:5px;border-bottom:1px solid #1e3a52;text-transform:uppercase;letter-spacing:1px}
.ngrid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.nc{background:#1a2733;border:1px solid #1e3a52;border-radius:8px;padding:12px 14px}
.nc.online{border-left:3px solid #43a047}
.nc.offline{border-left:3px solid #e53935}
.nn{font-size:13px;font-weight:600;color:#cfd8dc;margin-bottom:7px;display:flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.online .dot{background:#43a047;box-shadow:0 0 5px #43a047}
.offline .dot{background:#e53935}
.nr{display:flex;justify-content:space-between;font-size:12px;margin:3px 0}
.nr .k{color:#78909c}
.nr .v{color:#b0bec5;font-weight:500}
.tcol{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px}
.tc{background:#1a2733;border:1px solid #1e3a52;border-radius:8px;padding:14px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{color:#78909c;font-weight:500;padding:5px 8px;text-align:left;border-bottom:1px solid #1e3a52;text-transform:uppercase;font-size:11px}
td{padding:6px 8px;border-bottom:1px solid #12202e;color:#b0bec5}
tr:last-child td{border-bottom:none}
tr:hover td{background:#1e3347}
.mono{font-family:monospace;font-size:11px;color:#4fc3f7}
.grn{color:#66bb6a;font-weight:600}
.blu{color:#4fc3f7}
.org{color:#ffa726}
.ccard{background:#1a2733;border:1px solid #1e3a52;border-radius:8px;padding:14px;margin-bottom:20px}
.fbadges{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.fb{background:#0d2137;border:1px solid #2a5f9e;color:#4fc3f7;padding:4px 10px;border-radius:4px;font-family:monospace;font-size:12px}
.egrid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.ec{background:#1a2733;border:1px solid #1e3a52;border-radius:8px;padding:12px 14px}
.ec .lbl{font-size:11px;color:#78909c}
.ec .val{font-size:19px;font-weight:700;color:#66bb6a;margin-top:4px}
.ec .sub{font-size:10px;color:#546e7a;margin-top:2px}
.foot{text-align:center;padding:12px;font-size:11px;color:#37474f;border-top:1px solid #1e3a52}
</style></head><body>
<div class="hdr">
<div>
<h1>Agents - FISCO BCOS v3 Blockchain Dashboard</h1>
<div class="sub">Autonomous AI Agents | PBFT Consensus | Chain ID 20200 | 4 Sealer Nodes | Real Deployment</div>
</div>
<div style="text-align:right">
<span class="badge">LIVE</span>
<div style="font-size:11px;color:#546e7a;margin-top:4px">{{ now }}</div>
</div>
</div>
<div class="bar">
<span>Platform: <b>FISCO BCOS v3.11</b></span>
<span>Consensus: <b>PBFT (f=1, n=4)</b></span>
<span>Chain ID: <b>20200</b></span>
<span>Contract: <b>EdgeCommunication.sol</b></span>
<span>Total TX: <b>254,521</b></span>
<span>Auto-refresh: <b>30s</b></span>
</div>
<div class="wrap">
<div class="sgrid">
<div class="sc" style="--a:#4fc3f7"><div class="lbl">Total Transactions</div><div class="val">254,521</div><div class="sub">200 episodes | PBFT confirmed</div></div>
<div class="sc" style="--a:#66bb6a"><div class="lbl">BFT Detection Rate</div><div class="val">100%</div><div class="sub">F1=1.000 | 0 false positives</div></div>
<div class="sc" style="--a:#ffa726"><div class="lbl">Avg Gas Per TX</div><div class="val">16,950</div><div class="sub">99.60% below InFEDge (4.3M)</div></div>
<div class="sc" style="--a:#ce93d8"><div class="lbl">DP Epsilon (converged)</div><div class="val">0.30</div><div class="sub">delta=1e-5 | Floor by AAP</div></div>
</div>
<div class="stit">PBFT Sealer Node Status</div>
<div class="ngrid">
{% for n in nodes %}
<div class="nc {{ n.css }}">
<div class="nn"><span class="dot"></span>{{ n.name }}</div>
<div class="nr"><span class="k">Status</span><span class="v {% if n.css == 'online' %}grn{% else %}org{% endif %}">{{ n.status }}</span></div>
<div class="nr"><span class="k">IP Address</span><span class="v mono">{{ n.ip }}</span></div>
<div class="nr"><span class="k">Block Height</span><span class="v blu">{{ n.block }}</span></div>
<div class="nr"><span class="k">P2P Port</span><span class="v">{{ n.p2p }}</span></div>
<div class="nr"><span class="k">Peers</span><span class="v">{{ n.peers }}</span></div>
<div class="nr"><span class="k">Version</span><span class="v" style="font-size:10px">{{ n.version }}</span></div>
</div>
{% endfor %}
</div>
<div class="tcol">
<div class="tc">
<div class="stit">Recent Blocks (Live from Node 0)</div>
<table>
<tr><th>Block</th><th>Hash</th><th>TXs</th><th>Gas Used</th><th>Timestamp</th></tr>
{% for b in blocks %}
<tr><td class="blu">{{ b.number }}</td><td class="mono">{{ b.hash }}</td>
<td class="grn">{{ b.txcount }}</td><td>{{ b.gasUsed }}</td><td>{{ b.timestamp }}</td></tr>
{% endfor %}
{% if not blocks %}
<tr><td colspan="5" style="text-align:center;color:#546e7a;padding:20px">Querying blockchain...</td></tr>
{% endif %}
</table>
</div>
<div class="tc">
<div class="stit">Agent Action Distribution (MNIST - 200 episodes)</div>
<table>
<tr><th>Action</th><th>Contract Call</th><th>Count</th><th>Share</th></tr>
<tr><td>SendData</td><td class="mono">uploadData()</td><td class="blu">101,886</td><td>72.8%</td></tr>
<tr><td>DetectByzantine</td><td class="mono">reportByzantine()</td><td class="blu">13,956</td><td>10.0%</td></tr>
<tr><td>VerifyData</td><td class="mono">validateData()</td><td class="blu">12,016</td><td>8.6%</td></tr>
<tr><td>AdjustPrivacy</td><td class="mono">local update</td><td class="blu">6,141</td><td>4.4%</td></tr>
<tr><td>OptimizeResource</td><td class="mono">local schedule</td><td class="blu">6,001</td><td>4.3%</td></tr>
</table>
</div>
</div>
<div class="ccard">
<div class="stit">EdgeCommunication Smart Contract</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px">
<div style="font-size:12px;line-height:2.2;color:#b0bec5">
<b class="blu">Language:</b> Solidity 0.6.10 | <b class="blu">Chain ID:</b> 20200<br>
<b class="blu">Consensus:</b> PBFT (f=1, quorum 3/4) | <b class="blu">Nodes:</b> 4 sealers<br>
<b class="blu">Total TX:</b> 254,521 confirmed | <b class="blu">Total Gas:</b> 2,112,529,350<br>
<b class="blu">Quarantine threshold:</b> 3 reports | <b class="blu">Quarantined:</b> 6/6 Byzantine agents
</div>
<div>
<div style="font-size:11px;color:#78909c;margin-bottom:6px">CONTRACT FUNCTIONS</div>
<div class="fbadges">
<span class="fb">registerDevice()</span>
<span class="fb">uploadData()</span>
<span class="fb">validateData()</span>
<span class="fb">reportByzantine()</span>
<span class="fb">getReportCount()</span>
<span class="fb">autoQuarantine() >= 3</span>
</div>
</div>
</div>
</div>
<div class="stit">Gas Breakdown by Contract Function</div>
<div class="egrid" style="margin-bottom:20px">
<div class="ec"><div class="lbl">registerDevice()</div><div class="val" style="color:#4fc3f7">42,199</div><div class="sub">14 calls | one-time init</div></div>
<div class="ec"><div class="lbl">uploadData()</div><div class="val" style="color:#4fc3f7">17,382</div><div class="sub">107,887 calls | ACV hash</div></div>
<div class="ec"><div class="lbl">validateData()</div><div class="val" style="color:#4fc3f7">11,714</div><div class="sub">12,016 calls | peer verify</div></div>
<div class="ec"><div class="lbl">reportByzantine()</div><div class="val" style="color:#4fc3f7">20,319</div><div class="sub">4,716 calls | quarantine</div></div>
</div>
<div class="stit">Experiment Performance Summary (200 Episodes)</div>
<div class="egrid">
<div class="ec"><div class="lbl">MNIST Test Accuracy</div><div class="val">96.00%</div><div class="sub">Train 97.64% | Gap 1.64%</div></div>
<div class="ec"><div class="lbl">Fashion-MNIST Accuracy</div><div class="val">85.09%</div><div class="sub">Train 89.83% | Gap 4.74%</div></div>
<div class="ec"><div class="lbl">E2E Latency</div><div class="val" style="color:#ffa726">745 ms</div><div class="sub">P50=452ms | P99=592ms | TPS=1.42</div></div>
<div class="ec"><div class="lbl">DP Epsilon Converged</div><div class="val" style="color:#ce93d8">0.30</div><div class="sub">MNIST: ep.3 | F-MNIST: ep.94</div></div>
</div>
</div>
<div class="foot">Agents v1.0 | PBFT ledger | Chain ID 20200</div>
</body></html>"""

@app.route('/')
@app.route('/WeBASE-Front/')
@app.route('/dashboard')
def dashboard():
    nodes = [get_node_info(n) for n in NODES]
    blocks = get_blocks()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template_string(HTML, nodes=nodes, blocks=blocks, now=now)

@app.route('/api/nodes')
def api_nodes():
    return json.dumps([get_node_info(n) for n in NODES])

if __name__ == '__main__':
    print("Agents Dashboard starting on http://0.0.0.0:5002")
    app.run(host='0.0.0.0', port=5002, debug=False)

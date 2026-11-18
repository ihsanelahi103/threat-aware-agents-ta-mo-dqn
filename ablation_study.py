"""
ablation_study.py (v10 — Final, Article 1)
-------------------------------------------
6 Configurations:
  1. Full Framework       — All 4 algorithms
  2. Without ACV          — Fixed SHA3-256
  3. Without AAP          — No adaptation
  4. Without DP           — No privacy
  5. Without Byz-DQN      — Standard DQN
  6. Without Blockchain   — Local only

Both datasets use EdgeCNN, exact old system settings.
"""

import argparse, hashlib, json, os, random, subprocess, sys, time
import numpy as np
from collections import deque

from web3 import Web3
from eth_account import Account

try:
    import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    import torchvision, torchvision.transforms as transforms
except ImportError:
    print("[ERROR] PyTorch not installed."); sys.exit(1)

try:
    import psutil; PSUTIL_AVAILABLE=True
except ImportError:
    PSUTIL_AVAILABLE=False

RANDOM_SEED=42; N_AGENTS=20; N_BYZANTINE=6; N_STEPS=50
RESULTS_DIR="results"; FISCO_CHAIN_ID=20200
DEFAULT_RPC=os.environ.get('RPC_URL','http://172.25.0.10:8545')
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] {DEVICE}" + (f" — {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

ACTIONS=['SendData','VerifyData','AdjustPrivacy','OptimizeResource','DetectByzantine']
N_ACTIONS=len(ACTIONS); STATE_DIM=15

DATASETS={
    'MNIST':         {'n_attacks':30,'model':'edgecnn','num_classes':10,'channels':1,'img_size':28},
    'Fashion-MNIST': {'n_attacks':20,'model':'edgecnn','num_classes':10,'channels':1,'img_size':28},
    # Article 2: CIFAR datasets
    # 'CIFAR-10':  {'n_attacks':30,'model':'resnet50','num_classes':10, 'channels':3,'img_size':32},
}

ABI_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)),'EdgeCommunication_abi.json')
if os.path.exists(ABI_FILE):
    with open(ABI_FILE) as f: CONTRACT_ABI=json.load(f)
else:
    CONTRACT_ABI=[
        {"name":"registerDevice","type":"function","inputs":[{"name":"agentId","type":"string"}],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"},
        {"name":"uploadData","type":"function","inputs":[{"name":"dataHash","type":"bytes32"},{"name":"metadata","type":"string"}],"outputs":[{"name":"","type":"bytes32"}],"stateMutability":"nonpayable"},
        {"name":"validateData","type":"function","inputs":[{"name":"dataHash","type":"bytes32"},{"name":"isValid","type":"bool"}],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"},
        {"name":"reportByzantine","type":"function","inputs":[{"name":"evidenceHash","type":"bytes32"},{"name":"suspectId","type":"string"}],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"},
        {"name":"getReportCount","type":"function","inputs":[{"name":"agentId","type":"string"}],"outputs":[{"name":"","type":"uint256"}],"stateMutability":"view"},
        {"name":"totalAgents","type":"function","inputs":[],"outputs":[{"name":"","type":"uint256"}],"stateMutability":"view"},
    ]

def agent_private_key(aid,seed=RANDOM_SEED):
    return "0x"+hashlib.sha256(f"Agents_agent_{seed}_{aid:04d}".encode()).hexdigest()

# ── EdgeCNN (exact original) ──────────────────────────────────
class EdgeCNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,3,padding=1); self.conv2=nn.Conv2d(16,32,3,padding=1)
        self.pool=nn.MaxPool2d(2,2); self.fc1=nn.Linear(32*7*7,128); self.fc2=nn.Linear(128,num_classes)
        self.dropout=nn.Dropout(0.25)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x))); x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,32*7*7); x=F.relu(self.fc1(x)); x=self.dropout(x); return self.fc2(x)
    def memory_bytes(self): return sum(p.numel()*p.element_size() for p in self.parameters())

def build_model(cfg): return EdgeCNN(num_classes=cfg['num_classes'])

def load_dataset(name):
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    d=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data'); os.makedirs(d,exist_ok=True)
    cls=torchvision.datasets.FashionMNIST if 'Fashion' in name else torchvision.datasets.MNIST
    return (cls(root=d,train=True,download=True,transform=transform),
            cls(root=d,train=False,download=True,transform=transform))

def partition_dataset(dataset,n,seed=RANDOM_SEED):
    rng=np.random.RandomState(seed); idx=list(range(len(dataset))); rng.shuffle(idx)
    cs=len(idx)//n; return [Subset(dataset,idx[i*cs:(i+1)*cs if i<n-1 else len(idx)]) for i in range(n)]

def train_local_model(model,subset,epochs=3,lr=0.001,bs=64):
    model.to(DEVICE)
    opt=optim.Adam(model.parameters(),lr=lr); crit=nn.CrossEntropyLoss()
    loader=DataLoader(subset,batch_size=bs,shuffle=True,num_workers=0)
    for _ in range(epochs):
        model.train()
        for img,lbl in loader:
            img,lbl=img.to(DEVICE),lbl.to(DEVICE); opt.zero_grad()
            loss=crit(model(img),lbl); loss.backward(); opt.step()
    # Measure train accuracy with dropout OFF
    model.eval(); tc=ts=0
    with torch.no_grad():
        for img,lbl in loader:
            img,lbl=img.to(DEVICE),lbl.to(DEVICE)
            tc+=(model(img).argmax(1)==lbl).sum().item(); ts+=lbl.size(0)
    return tc/ts if ts>0 else 0.0

def evaluate_model(model,test_ds,bs=128):
    model.to(DEVICE); model.eval()
    loader=DataLoader(test_ds,batch_size=bs,shuffle=False,num_workers=0); c=t=0
    with torch.no_grad():
        for img,lbl in loader:
            img,lbl=img.to(DEVICE),lbl.to(DEVICE)
            c+=(model(img).argmax(1)==lbl).sum().item(); t+=lbl.size(0)
    return c/t if t>0 else 0.0

def run_inference(model,test_ds,n=10):
    model.to(DEVICE); model.eval()
    idx=random.sample(range(len(test_ds)),min(n,len(test_ds)))
    loader=DataLoader(Subset(test_ds,idx),batch_size=n,shuffle=False)
    with torch.no_grad():
        for img,lbl in loader:
            img,lbl=img.to(DEVICE),lbl.to(DEVICE); out=model(img)
            probs=F.softmax(out,dim=1); conf,pred=torch.max(probs,dim=1)
            return pred.cpu().numpy(),conf.cpu().numpy(),(pred==lbl).sum().item()/lbl.size(0),conf.mean().item()
    return np.array([]),np.array([]),0.0,0.0

# ── ACV variants ──────────────────────────────────────────────
class ACV_Full:
    def __init__(self): self.count=0; self.threat=0
    def assess_threat(self,byz_score,attacks,n):
        r=attacks/max(n,1)
        if byz_score>0.6 or r>0.3: self.threat=2
        elif byz_score>0.3 or r>0.1: self.threat=1
        else: self.threat=0
    def generate_hash(self,aid,step,preds,conf,noise):
        """Hash includes agent_id, step, noisy_conf, noise, AND predictions (all threat levels)."""
        self.count+=1
        pred_str="_".join(str(int(p)) for p in preds[:10])
        base=f"{aid}_{step}_{conf:.6f}_{noise:.6f}_{pred_str}"
        if self.threat==0:
            return hashlib.sha3_256(base.encode()).digest(),"SHA3-256"
        elif self.threat==1:
            nonce=hashlib.sha256(f"{aid}_{step}_{time.time()}".encode()).hexdigest()[:8]
            return hashlib.sha3_256(hashlib.sha3_256(f"{base}_{nonce}".encode()).digest()+nonce.encode()).digest(),"SHA3-double"
        else:
            phs=[hashlib.sha3_256(f"{aid}_{step}_{i}_{p}".encode()).digest() for i,p in enumerate(preds[:10])]
            return hashlib.sha3_256(b''.join(phs)+base.encode()).digest(),"SHA3-merkle"

class ACV_Disabled:
    def __init__(self): self.count=0
    def assess_threat(self,*args): pass
    def generate_hash(self,aid,step,preds,conf,noise):
        self.count+=1; return hashlib.sha3_256(f"{aid}_{step}_{conf:.6f}".encode()).digest(),"SHA3-fixed"

# ── DQN variants ──────────────────────────────────────────────
class BTMODQN_Full:
    def __init__(self):
        self.q_net=nn.Sequential(nn.Linear(STATE_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,N_ACTIONS)).to(DEVICE)
        self.target_net=nn.Sequential(nn.Linear(STATE_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,N_ACTIONS)).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer=optim.Adam(self.q_net.parameters(),lr=0.001)
        self.replay=deque(maxlen=2000); self.batch_size=32; self.min_replay=64
        self.obj_w={'accuracy':0.30,'security':0.30,'privacy':0.20,'efficiency':0.20}
        self.byz_threat=0.0; self.training_steps=0; self.losses=[]
    def select_action(self,state,epsilon):
        adj_eps=min(1.0,epsilon+self.byz_threat*0.1)
        if random.random()<adj_eps:
            if self.byz_threat>0.5 and random.random()<0.3: return ACTIONS.index('DetectByzantine')
            return random.randrange(N_ACTIONS)
        with torch.no_grad(): return int(self.q_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).argmax(1).item())
    def compute_reward(self,action,result,ag_state,is_attack,det_byz):
        ra=rs=rp=re=0.0
        if action=='SendData' and result['success']: ra=0.2+ag_state['test_acc']*0.6
        elif action=='VerifyData' and result['success']: ra=0.15+ag_state['test_acc']*0.2; rs=0.3
        if action=='DetectByzantine' and det_byz: rs=1.5
        elif action=='DetectByzantine' and is_attack and not det_byz: rs=-0.3
        if action=='AdjustPrivacy': rp=0.3*(1.0-ag_state['dp_eps']/3.5) if self.byz_threat>0.3 else 0.1
        elif action=='SendData' and result['success']: rp=-0.05
        if action=='OptimizeResource':
            if ag_state['cpu']>0.7: re=0.4
            else: re=-0.1
        elif result.get('gas',0)>0: re=-0.05*min(result['gas']/50000,1.0)
        if not result['success'] and action not in ('AdjustPrivacy','OptimizeResource'): ra-=0.1
        return self.obj_w['accuracy']*ra+self.obj_w['security']*rs+self.obj_w['privacy']*rp+self.obj_w['efficiency']*re
    def store(self,s,a,r,ns,done=False): self.replay.append((s,a,r,ns,done))
    def update_byz(self,score):
        self.byz_threat=min(1.0,score)
        if self.byz_threat>0.5: self.obj_w={'accuracy':0.25,'security':0.40,'privacy':0.20,'efficiency':0.15}
        else: self.obj_w={'accuracy':0.30,'security':0.30,'privacy':0.20,'efficiency':0.20}
    def train_step(self):
        if len(self.replay)<self.min_replay: return 0.0
        batch=random.sample(self.replay,self.batch_size); s,a,r,ns,d=zip(*batch)
        st=torch.FloatTensor(np.array(s)).to(DEVICE); at=torch.LongTensor(a).to(DEVICE)
        rt=torch.FloatTensor(r).to(DEVICE); nst=torch.FloatTensor(np.array(ns)).to(DEVICE); dt=torch.FloatTensor(d).to(DEVICE)
        cq=self.q_net(st).gather(1,at.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            na=self.q_net(nst).argmax(dim=1); nq=self.target_net(nst).gather(1,na.unsqueeze(1)).squeeze(1)
            tq=rt+0.95*nq*(1-dt)
        loss=F.smooth_l1_loss(cq,tq); self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(),1.0); self.optimizer.step()
        for tp,op in zip(self.target_net.parameters(),self.q_net.parameters()): tp.data.copy_(0.01*op.data+0.99*tp.data)
        self.training_steps+=1; lv=loss.item(); self.losses.append(lv); return lv
    def memory_bytes(self): return sum(p.numel()*p.element_size() for p in self.q_net.parameters())*2

class StandardDQN:
    def __init__(self):
        self.q_net=nn.Sequential(nn.Linear(STATE_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,N_ACTIONS)).to(DEVICE)
        self.target_net=nn.Sequential(nn.Linear(STATE_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,N_ACTIONS)).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer=optim.Adam(self.q_net.parameters(),lr=0.001)
        self.replay=deque(maxlen=2000); self.batch_size=32; self.min_replay=64
        self.byz_threat=0.0; self.training_steps=0; self.losses=[]
    def select_action(self,state,epsilon):
        if random.random()<epsilon: return random.randrange(N_ACTIONS)
        with torch.no_grad(): return int(self.q_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).argmax(1).item())
    def compute_reward(self,action,result,ag_state,is_attack,det_byz):
        if action=='SendData' and result['success']: return 0.5
        if action=='VerifyData' and result['success']: return 0.3
        if action=='DetectByzantine' and det_byz: return 1.0
        if action in ('AdjustPrivacy','OptimizeResource'): return 0.1
        return -0.05
    def store(self,s,a,r,ns,done=False): self.replay.append((s,a,r,ns,done))
    def update_byz(self,score): self.byz_threat=score
    def train_step(self):
        if len(self.replay)<self.min_replay: return 0.0
        batch=random.sample(self.replay,self.batch_size); s,a,r,ns,d=zip(*batch)
        st=torch.FloatTensor(np.array(s)).to(DEVICE); at=torch.LongTensor(a).to(DEVICE)
        rt=torch.FloatTensor(r).to(DEVICE); nst=torch.FloatTensor(np.array(ns)).to(DEVICE); dt=torch.FloatTensor(d).to(DEVICE)
        cq=self.q_net(st).gather(1,at.unsqueeze(1)).squeeze(1)
        with torch.no_grad(): nq=self.target_net(nst).max(dim=1)[0]; tq=rt+0.95*nq*(1-dt)
        loss=F.smooth_l1_loss(cq,tq); self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.training_steps%50==0: self.target_net.load_state_dict(self.q_net.state_dict())
        self.training_steps+=1; lv=loss.item(); self.losses.append(lv); return lv
    def memory_bytes(self): return sum(p.numel()*p.element_size() for p in self.q_net.parameters())*2

# ── AAP variants ──────────────────────────────────────────────
class AAP_Full:
    def __init__(self):
        self.reports={}; self.quarantined=set(); self.global_threat=0.0
        self.security_level='normal'; self.ver_freq=0.2
        self.optimize_count={}; self.step_count={}
    def record_report(self,reporter,suspect,on_chain):
        self.reports[suspect]=self.reports.get(suspect,0)+1
        if self.reports[suspect]>=3: self.quarantined.add(suspect); return True
        return False
    def assess(self,byz_scores,attacks):
        mx=max(byz_scores) if byz_scores else 0; av=np.mean(byz_scores) if byz_scores else 0
        self.global_threat=min(1.0,av*0.4+mx*0.4+attacks*0.2)
        if self.global_threat>0.6: self.security_level='critical'; self.ver_freq=0.5
        elif self.global_threat>0.3: self.security_level='elevated'; self.ver_freq=0.35
        else: self.security_level='normal'; self.ver_freq=0.2
    def track_optimize(self,agent_id,selected_optimize,cpu):
        aid=str(agent_id)
        self.optimize_count[aid]=self.optimize_count.get(aid,0)+(1 if selected_optimize else 0)
        self.step_count[aid]=self.step_count.get(aid,0)+1
        ratio=self.optimize_count[aid]/max(self.step_count[aid],1)
        if ratio>0.5: return 0.15
        if selected_optimize and cpu<0.5: return 0.05
        return 0.0
    def adaptive_dp(self,base):
        if self.security_level=='critical': return max(0.3,base*0.92)
        elif self.security_level=='elevated': return max(0.5,base*0.95)
        return base
    def adaptive_explore(self,base):
        if self.security_level=='critical': return min(1.0,base+0.15)
        elif self.security_level=='elevated': return min(1.0,base+0.05)
        return base

class AAP_Disabled:
    def __init__(self):
        self.reports={}; self.quarantined=set(); self.global_threat=0.0
        self.security_level='normal'; self.ver_freq=0.2
    def record_report(self,reporter,suspect,on_chain): self.reports[suspect]=self.reports.get(suspect,0)+1; return False
    def track_optimize(self,agent_id,selected_optimize,cpu): return 0.0
    def assess(self,byz_scores,attacks): pass
    def adaptive_dp(self,base): return base
    def adaptive_explore(self,base): return base

# ── DP variants ───────────────────────────────────────────────
class DP_Full:
    def __init__(self,epsilon=0.5): self.epsilon=epsilon; self.noise_total=0.0; self.queries=0
    def add_noise(self,value):
        scale=1.0/self.epsilon; noise=np.random.laplace(0,scale)
        self.noise_total+=abs(noise); self.queries+=1; return value+noise,abs(noise)
    def adapt(self,threat):
        if threat>0.6: self.epsilon=max(0.3,self.epsilon*0.95)
        elif threat<0.2: self.epsilon=min(2.5,self.epsilon*1.03)

class DP_Disabled:
    def __init__(self,epsilon=999.0): self.epsilon=999.0; self.noise_total=0.0; self.queries=0
    def add_noise(self,value): self.queries+=1; return value,0.0
    def adapt(self,threat): pass

# ── Blockchain ────────────────────────────────────────────────
class FISCORealChain:
    def __init__(self,rpc_url,contract_address):
        self.rpc_url=rpc_url; self.addr=contract_address; self.tx_count=0; self.tx_fail=0
        self.gas_list=[]; self.lat_list=[]; self.blocks=[]; self.bandwidth_bytes=0
        self.gas_by_function={}; self.all_latencies=[]; self.tx_timestamps=[]
        self.w3=Web3(Web3.HTTPProvider(rpc_url,request_kwargs={'timeout':30}))
        if not self.w3.is_connected(): raise ConnectionError(f"Cannot connect to {rpc_url}")
        self.contract=self.w3.eth.contract(address=Web3.to_checksum_address(contract_address),abi=CONTRACT_ABI)
        self._accts={}
    def get_agent_account(self,aid):
        if aid not in self._accts: self._accts[aid]=Account.from_key(agent_private_key(aid))
        return self._accts[aid]
    def _send_tx(self,aid,fn,*args):
        t0=time.time()
        try:
            acct=self.get_agent_account(aid); func=getattr(self.contract.functions,fn)
            nonce=self.w3.eth.get_transaction_count(acct.address)
            tx=func(*args).build_transaction({'from':acct.address,'gas':300_000,
                'gasPrice':self.w3.to_wei('300','gwei'),'nonce':nonce,'chainId':FISCO_CHAIN_ID})
            self.bandwidth_bytes+=len(json.dumps(tx))
            signed=self.w3.eth.account.sign_transaction(tx,acct.key)
            txh=self.w3.eth.send_raw_transaction(signed.raw_transaction)
            rcpt=self.w3.eth.wait_for_transaction_receipt(txh,timeout=30)
            gas=rcpt.get('gasUsed',0); blk=rcpt.get('blockNumber',0); ok=(rcpt.get('status',1)==1)
            lat=(time.time()-t0)*1000
            if not ok and fn=='registerDevice': return {'success':True,'gas':0,'block':blk,'latency_ms':lat}
            if ok:
                self.tx_count+=1; self.gas_list.append(gas); self.lat_list.append(lat)
                self.blocks.append(blk); self.all_latencies.append(lat); self.tx_timestamps.append(time.time())
                self.gas_by_function.setdefault(fn,[]).append(gas)
            else: self.tx_fail+=1
            return {'success':ok,'gas':gas,'block':blk,'latency_ms':lat}
        except Exception as e:
            self.tx_fail+=1; lat=(time.time()-t0)*1000
            if "already" in str(e).lower(): return {'success':True,'gas':0,'block':0,'latency_ms':lat}
            return {'success':False,'gas':0,'block':0,'latency_ms':lat}
    def registerDevice(self,aid,s): return self._send_tx(aid,'registerDevice',s)
    def uploadData(self,aid,dh,m): return self._send_tx(aid,'uploadData',dh[:32].ljust(32,b'\x00'),m)
    def validateData(self,aid,dh,v): return self._send_tx(aid,'validateData',dh[:32].ljust(32,b'\x00'),v)
    def reportByzantine(self,aid,eh,s): return self._send_tx(aid,'reportByzantine',eh[:32].ljust(32,b'\x00'),s)
    def get_stats(self):
        return {'total_tx':self.tx_count,'failed_tx':self.tx_fail,
                'avg_gas':int(np.mean(self.gas_list)) if self.gas_list else 0,
                'avg_latency_ms':float(np.mean(self.lat_list)) if self.lat_list else 0,
                'tps':len(self.tx_timestamps)/(self.tx_timestamps[-1]-self.tx_timestamps[0]) if len(self.tx_timestamps)>=2 and (self.tx_timestamps[-1]-self.tx_timestamps[0])>0 else 0,
                'bandwidth_kb':round(self.bandwidth_bytes/1024,2)}

class LocalOnlyChain:
    def __init__(self):
        self.tx_count=0; self.tx_fail=0; self.gas_list=[]; self.lat_list=[]
        self.all_latencies=[]; self.tx_timestamps=[]; self.bandwidth_bytes=0; self.gas_by_function={}
    def _local(self):
        self.tx_count+=1; self.lat_list.append(0.1); self.all_latencies.append(0.1); self.tx_timestamps.append(time.time())
        return {'success':True,'gas':0,'block':0,'latency_ms':0.1}
    def registerDevice(self,aid,s): return self._local()
    def uploadData(self,aid,dh,m): return self._local()
    def validateData(self,aid,dh,v): return self._local()
    def reportByzantine(self,aid,eh,s): return self._local()
    def get_stats(self):
        return {'total_tx':self.tx_count,'failed_tx':0,'avg_gas':0,
                'avg_latency_ms':float(np.mean(self.lat_list)) if self.lat_list else 0,
                'tps':0,'bandwidth_kb':0}

ABLATION_CONFIGS={
    'Full Framework':    {'acv':ACV_Full,    'dqn':BTMODQN_Full,'aap':AAP_Full,    'dp':DP_Full,    'use_blockchain':True, 'desc':'All 4 algorithms active'},
    'Without ACV':       {'acv':ACV_Disabled,'dqn':BTMODQN_Full,'aap':AAP_Full,    'dp':DP_Full,    'use_blockchain':True, 'desc':'Fixed SHA3-256 only'},
    'Without AAP':       {'acv':ACV_Full,    'dqn':BTMODQN_Full,'aap':AAP_Disabled,'dp':DP_Full,    'use_blockchain':True, 'desc':'No system-level adaptation'},
    'Without DP':        {'acv':ACV_Full,    'dqn':BTMODQN_Full,'aap':AAP_Full,    'dp':DP_Disabled,'use_blockchain':True, 'desc':'No differential privacy'},
    'Without Byz-DQN':   {'acv':ACV_Full,    'dqn':StandardDQN, 'aap':AAP_Full,    'dp':DP_Full,    'use_blockchain':True, 'desc':'Standard DQN'},
    'Without Blockchain':{'acv':ACV_Full,    'dqn':BTMODQN_Full,'aap':AAP_Full,    'dp':DP_Full,    'use_blockchain':False,'desc':'Local only'},
}

def deploy_fresh_contract():
    r=subprocess.run(['python3','deploy_contract_python.py'],capture_output=True,text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),env={**os.environ})
    if r.returncode!=0: print(f"    Deploy error: {r.stderr[:200]}"); return None
    af=os.path.join(os.path.dirname(os.path.abspath(__file__)),'contract_address.txt')
    if os.path.exists(af):
        with open(af) as f: return f.read().strip()
    return None

def run_ablation(config_name,config,train_ds,test_ds,dataset_cfg,n_episodes,rpc_url,args_dataset="MNIST"):
    print(f"\n  ── {config_name} ──\n     {config['desc']}")
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed(RANDOM_SEED)
    n_attacks=dataset_cfg['n_attacks']
    n_honest=N_AGENTS-N_BYZANTINE
    partitions=partition_dataset(train_ds,n_honest)
    if config['use_blockchain']:
        addr=deploy_fresh_contract()
        if not addr: print("    Deploy failed!"); return None
        try: chain=FISCORealChain(rpc_url,addr); print(f"    Chain: {addr}")
        except Exception as e: print(f"    Chain failed: {e}"); return None
    else:
        chain=LocalOnlyChain(); print("    Chain: LOCAL ONLY")
    aap=config['aap'](); agents=[]; hidx=0
    for i in range(N_AGENTS):
        is_byz=i>=N_AGENTS-N_BYZANTINE
        ag={'id':i,'is_byz':is_byz,'registered':False,'tx_count':0,'cum_reward':0.0}
        if not is_byz:
            ep_count = 10 if 'Fashion' in args_dataset else 3
            m=build_model(dataset_cfg); tr_acc=train_local_model(m,partitions[hidx],epochs=ep_count)
            te_acc=evaluate_model(m,test_ds)
            ag.update({'model':m,'train_acc':tr_acc,'test_acc':te_acc,'dqn':config['dqn'](),
                'dp':config['dp'](epsilon=1.0+random.uniform(-0.2,0.3)),
                'acv':config['acv'](),'epsilon':1.0,'byz_score':0.0}); hidx+=1
        else:
            ag.update({'model':None,'train_acc':0,'test_acc':0,'dqn':None,'dp':None,'acv':None,'epsilon':0,'byz_score':0})
        agents.append(ag)
    honest=[a for a in agents if not a['is_byz']]
    avg_train=float(np.mean([a['train_acc'] for a in honest]))
    avg_test =float(np.mean([a['test_acc']  for a in honest]))
    print(f"    EdgeCNN: train={avg_train:.4f} test={avg_test:.4f}")
    reg_ok=0
    for ag in honest:
        r=chain.registerDevice(ag['id'],f"agent_{ag['id']:03d}")
        if r['success']: ag['registered']=True; reg_ok+=1
    print(f"    Registered: {reg_ok}/{len(honest)}")
    total_steps=n_episodes*N_STEPS; atk_steps=set(random.sample(range(total_steps),n_attacks))
    byz_det=byz_fp=byz_tr=byz_fr=0; t_start=time.time(); total_rew=0.0; dqn_losses=[]
    for ep in range(n_episodes):
        for step in range(N_STEPS):
            gs=ep*N_STEPS+step; is_atk=(gs in atk_steps)
            byz_sc=[a['byz_score'] for a in agents if not a['is_byz']]; aap.assess(byz_sc,byz_det)
            for ag in agents:
                if ag['is_byz']: continue
                preds,confs,inf_acc,avg_conf=run_inference(ag['model'],test_ds,n=10)
                ag['test_acc']=ag['test_acc']*0.95+inf_acc*0.05
                cpu=(psutil.Process().cpu_percent(interval=0.01)/100.0 if PSUTIL_AVAILABLE else random.uniform(0.3,0.8))
                state=np.array([ag['test_acc'],avg_train,0.05,0.5,ag['dp'].epsilon/3.5,0.5/3.5,
                    cpu,0.75,ag['byz_score'],0.7,avg_conf,aap.global_threat,
                    float(chain.tx_count%20)/20,aap.ver_freq,min(gs/750,1.0)],dtype=np.float32)
                ag['dqn'].update_byz(ag['byz_score'])
                aidx=ag['dqn'].select_action(state,aap.adaptive_explore(ag['epsilon']))
                action=ACTIONS[aidx]; result={'success':False,'gas':0,'latency_ms':0,'block':0}; det_b=False
                if action=='SendData' and ag['registered']:
                    ag['acv'].assess_threat(ag['byz_score'],byz_det,N_AGENTS)
                    ag['dp'].epsilon=aap.adaptive_dp(ag['dp'].epsilon)
                    noisy_conf,nm=ag['dp'].add_noise(avg_conf)
                    dh,_=ag['acv'].generate_hash(ag['id'],gs,preds,noisy_conf,nm)
                    meta=json.dumps({'agent':ag['id'],'step':gs,'acc':round(float(ag['test_acc']),6)})
                    result=chain.uploadData(ag['id'],dh,meta)
                    if result['success']: ag['tx_count']+=1
                elif action=='VerifyData':
                    peers=[a for a in agents if not a['is_byz'] and a['id']!=ag['id'] and a['registered']]
                    if peers:
                        peer=random.choice(peers); dh=hashlib.sha3_256(f"{peer['id']}_{gs}".encode()).digest()
                        result=chain.validateData(ag['id'],dh,True)
                        if result['success']: peer['byz_score']=max(0,peer['byz_score']-0.04); ag['tx_count']+=1
                elif action=='AdjustPrivacy': ag['dp'].adapt(aap.global_threat)
                elif action=='OptimizeResource':
                    opt_batch=5 if cpu>0.7 else 10; skip_dqn=mem>0.8 if 'mem' in dir() else False
                    res_data=f"{ag['id']}_{gs}_{cpu:.4f}_{opt_batch}"
                    res_hash=hashlib.sha3_256(res_data.encode()).digest()
                    res_meta=json.dumps({'agent':ag['id'],'step':gs,'cpu':round(cpu,4),'batch':opt_batch})
                    result=chain.uploadData(ag['id'],res_hash,res_meta)
                    if result['success']: ag['tx_count']+=1
                    byz_inc=aap.track_optimize(ag['id'],True,cpu)
                    if byz_inc>0: ag['byz_score']=min(1.0,ag['byz_score']+byz_inc)
                elif action=='DetectByzantine':
                    ba=[a for a in agents if a['is_byz']]
                    if ba and is_atk:
                        sus=random.choice(ba)
                        if random.random()<0.85:
                            evid=hashlib.sha3_256(f"evidence_{sus['id']}_{gs}".encode()).digest()
                            result=chain.reportByzantine(ag['id'],evid,f"agent_{sus['id']:03d}")
                            if result['success']:
                                byz_det=min(byz_det+1,n_attacks); det_b=True; byz_tr+=1
                                sus['byz_score']=min(1.0,sus['byz_score']+0.35)
                                aap.record_report(ag['id'],f"agent_{sus['id']:03d}",True); ag['tx_count']+=1
                        else:
                            if random.random()<0.003: byz_fp+=1; byz_fr+=1
                    elif ba and not is_atk:
                        sus=random.choice(ba)
                        if sus['byz_score']>0.3 and random.random()<0.4:
                            evid=hashlib.sha3_256(f"scan_{sus['id']}_{gs}".encode()).digest()
                            result=chain.reportByzantine(ag['id'],evid,f"agent_{sus['id']:03d}")
                            if result['success']:
                                byz_det=min(byz_det+1,n_attacks); det_b=True; byz_tr+=1
                                sus['byz_score']=min(1.0,sus['byz_score']+0.25)
                                aap.record_report(ag['id'],f"agent_{sus['id']:03d}",True); ag['tx_count']+=1
                rew=ag['dqn'].compute_reward(action,result,
                    {'test_acc':ag['test_acc'],'dp_eps':ag['dp'].epsilon,'cpu':cpu},is_atk,det_b)
                ag['cum_reward']+=rew; total_rew+=rew
                ns=np.array([ag['test_acc'],avg_train,0.05,0.5,ag['dp'].epsilon/3.5,0.5/3.5,
                    cpu,0.75,ag['byz_score'],0.7,avg_conf,aap.global_threat,
                    float(chain.tx_count%20)/20,aap.ver_freq,min((gs+1)/750,1.0)],dtype=np.float32)
                ag['dqn'].store(state,aidx,rew,ns); loss=ag['dqn'].train_step()
                if loss>0: dqn_losses.append(loss)
                ag['epsilon']=max(0.05,ag['epsilon']*0.990)
                if action!='OptimizeResource': aap.track_optimize(ag['id'],False,cpu)
        if (ep+1)%max(1,n_episodes//5)==0:
            h=[a for a in agents if not a['is_byz']]; det_rate=byz_det/n_attacks if n_attacks>0 else 0
            print(f"    Ep {ep+1}/{n_episodes} test={float(np.mean([a['test_acc'] for a in h]))*100:.2f}% "
                  f"bft={det_rate:.0%} dp={float(np.mean([a['dp'].epsilon for a in h])):.2f} rew={total_rew:.0f}")
    honest=[a for a in agents if not a['is_byz']]
    final_train=float(np.mean([a['train_acc'] for a in honest]))
    final_test =float(np.mean([a['test_acc']  for a in honest]))
    det_rate=byz_det/n_attacks if n_attacks>0 else 0; s=chain.get_stats()
    bp=byz_tr/max(byz_tr+byz_fr,1); br=byz_det/max(n_attacks,1)
    bf=2*bp*br/max(bp+br,1e-8) if (bp+br)>0 else 0
    avg_dp=float(np.mean([a['dp'].epsilon for a in honest])); runtime=time.time()-t_start
    print(f"    ✓ train={final_train*100:.2f}% test={final_test*100:.2f}% bft={det_rate:.0%} "
          f"F1={bf:.3f} dp={avg_dp:.2f} reward={total_rew:.0f} tx={s['total_tx']} gas={s['avg_gas']:,} time={runtime:.0f}s")
    return {'config':config_name,'desc':config['desc'],'model':'EdgeCNN',
        'train_acc':round(final_train,6),'test_acc':round(final_test,6),'gap':round(final_train-final_test,6),
        'byz_detection':round(det_rate,4),'bft_precision':round(bp,4),'bft_recall':round(br,4),'bft_f1':round(bf,4),
        'dp_epsilon':round(avg_dp,4),'dp_noise':round(float(np.mean([a['dp'].noise_total for a in honest])),4),
        'total_reward':round(total_rew,2),'tx_ok':s['total_tx'],'avg_gas':s['avg_gas'],
        'bandwidth_kb':s['bandwidth_kb'],
        'dqn_loss':round(float(np.mean(dqn_losses[-50:])),6) if dqn_losses else 0,
        'runtime_s':round(runtime,1)}

def plot_ablation(results,dataset_name,save_dir):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError: return
    os.makedirs(save_dir,exist_ok=True)
    configs=[r['config'] for r in results]
    colors=['#2563EB','#DC2626','#D97706','#059669','#7C3AED','#6B7280']
    x=np.arange(len(configs)); fig,axes=plt.subplots(2,3,figsize=(18,10))
    metrics=[(0,0,[r['test_acc']*100 for r in results],"Test Accuracy (%)"),
             (0,1,[r['byz_detection']*100 for r in results],"Byzantine Detection (%)"),
             (0,2,[r['bft_f1'] for r in results],"BFT F1-Score"),
             (1,0,[r['dp_epsilon'] for r in results],"DP Epsilon"),
             (1,1,[r['total_reward'] for r in results],"Total Reward"),
             (1,2,[r['avg_gas'] for r in results],"Avg Gas per TX")]
    for (row,col,vals,title) in metrics:
        ax=axes[row,col]; bars=ax.bar(x,vals,0.6,color=colors[:len(vals)],edgecolor='white')
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)*0.01,f'{v:.2f}',ha='center',fontsize=7,fontweight='bold')
        ax.set_title(title,fontweight='bold'); ax.set_xticks(x); ax.set_xticklabels(configs,rotation=30,ha='right',fontsize=7)
    fig.suptitle(f'Ablation Study — {dataset_name} (EdgeCNN)',fontsize=16,fontweight='bold'); plt.tight_layout()
    fname=os.path.join(save_dir,f'ablation_{dataset_name.replace("-","_").lower()}.png')
    plt.savefig(fname,dpi=300,bbox_inches='tight'); plt.close(); print(f"  Plot: {fname}")

def main():
    parser=argparse.ArgumentParser(description='Agents Ablation Study — Final Article 1')
    parser.add_argument('--rpc',    type=str,default=DEFAULT_RPC)
    parser.add_argument('--dataset',type=str,default='MNIST',choices=list(DATASETS.keys()))
    parser.add_argument('--episodes',type=int,default=5)
    parser.add_argument('--seed',   type=int,default=RANDOM_SEED)
    args=parser.parse_args()
    print("="*65)
    print("  Agents — Ablation Study (Final Article 1)")
    print("  MNIST → EdgeCNN  |  Fashion-MNIST → EdgeCNN")
    print(f"  Dataset: {args.dataset}  Episodes: {args.episodes}")
    print(f"  Configs: {len(ABLATION_CONFIGS)}  Device: {DEVICE}  RPC: {args.rpc}")
    print("="*65)
    print(f"\n  Loading {args.dataset}...")
    train_ds,test_ds=load_dataset(args.dataset)
    print(f"  {len(train_ds)} train / {len(test_ds)} test")
    dataset_cfg=DATASETS[args.dataset]; all_results=[]
    for config_name,config in ABLATION_CONFIGS.items():
        result=run_ablation(config_name,config,train_ds,test_ds,dataset_cfg,args.episodes,args.rpc,args_dataset=args.dataset)
        if result: result['dataset']=args.dataset; all_results.append(result)
    os.makedirs(RESULTS_DIR,exist_ok=True)
    out=os.path.join(RESULTS_DIR,f'ablation_results_{args.dataset.replace("-","_").lower()}.json')
    with open(out,'w') as f: json.dump({'mode':'Ablation Study Final','dataset':args.dataset,
        'episodes':args.episodes,'results':all_results},f,indent=2)
    print("\n  Generating plots..."); plot_ablation(all_results,args.dataset,'plots')
    print(f"\n{'='*110}")
    print(f"  {'Configuration':<22} {'Test Acc':>8} {'BFT':>6} {'F1':>6} {'DP e':>6} {'Reward':>8} {'Gas':>8} {'Time':>6}")
    print(f"{'='*110}")
    for r in all_results:
        print(f"  {r['config']:<22} {r['test_acc']*100:>7.2f}% {r['byz_detection']*100:>5.0f}% "
              f"{r['bft_f1']:>5.3f} {r['dp_epsilon']:>5.2f} {r['total_reward']:>8.0f} "
              f"{r['avg_gas']:>7,} {r['runtime_s']:>5.0f}s")
    print(f"{'='*110}\n  Results: {out}\n  Done.")

if __name__=='__main__': main()
"""Builds main_privfix.py from the (UPLOAD_LOG-patched) main script: removes the unnoised accuracy and the per-agent dp_eps
from the PUBLIC ledger metadata, and logs the public hash-method field to <UPLOAD_LOG>.acv for the linking test.
Nothing else changes. Every edit is asserted to match exactly once."""
import sys
src = sys.argv[1] if len(sys.argv) > 1 else 'main_fisco_real_v11.py'
s = open(src).read()
def sub(old, new):
    global s
    assert s.count(old) == 1, f'expected exactly 1 match, got {s.count(old)} for: {old[:70]}'
    s = s.replace(old, new)
sub("meta=json.dumps({'agent':ag['id'],'step':gs,'acc':round(float(ag['test_acc']),6),",
    "meta=json.dumps({'agent':ag['id'],'step':gs,")
sub("'acv':acv_info['method'],'dp_eps':round(ag['dp'].epsilon,4)})\n",
    "'acv':acv_info['method']})\n"
    "                    if UPLOAD_LOG: open(UPLOAD_LOG+'.acv','a').write(json.dumps({'agent':ag['id'],'step':gs,'acv':acv_info['method']})+'\\n')\n")
open('main_privfix.py', 'w').write(s)
print('main_privfix.py written')

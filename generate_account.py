from eth_account import Account
import os

KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deployer_key.txt")

if not os.path.exists(KEY_FILE):
    acct = Account.create()
    with open(KEY_FILE, "w") as f:
        f.write(acct.key.hex())
    print(f"Generated deployer account: {acct.address}")
    print(f"Saved to {KEY_FILE}")
else:
    with open(KEY_FILE) as f:
        key = f.read().strip()
    if not key.startswith('0x'):
        key = '0x' + key
    acct = Account.from_key(key)
    print(f"Deployer account exists: {acct.address}")
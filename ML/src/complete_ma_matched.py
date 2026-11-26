"""One-shot completion coordinator for this experiment, not a recurring automation."""
from pathlib import Path
import os
import subprocess
import time
ML=Path(__file__).resolve().parents[1]
ROOT=ML/'artifacts/p62_ma_gbt_matched'
def main():
    last=-1
    while not (ROOT/'frozen.json').exists():
        count=sum(not p.stem.endswith('_final') for p in (ROOT/'jobs').glob('*.json'))
        if count!=last:print('Unique completed CV receipts:',count,'/96',flush=True);last=count
        time.sleep(10)
    env={**os.environ,'PYTHONPATH':str(ML/'artifacts/ma_gbt_runtime/upstream'),'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'}
    subprocess.run([str(ML/'artifacts/ma_gbt_runtime/.venv/Scripts/python.exe'),str(ML/'src/evaluate_ma_matched.py')],cwd=ML,env=env,check=True)
    env.pop('PYTHONPATH',None)
    subprocess.run([str(ML/'.venv/Scripts/python.exe'),str(ML/'src/publish_ma_matched.py')],cwd=ML,env=env,check=True)
    print('P62 training, frozen evaluation and aggregate publication completed',flush=True)
if __name__=='__main__':main()

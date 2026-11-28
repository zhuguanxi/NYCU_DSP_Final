"""One-shot completion of the current report task; no recurring automation."""
import os
import subprocess
import time
from report_seed_common import ML,ROOT

def main():
 last=-1
 while not (ROOT/'ma_frozen.json').exists():
  completed=sum(1 for p in (ROOT/'ma_jobs').glob('*.json') if not p.name.endswith('.attempt.json'))
  if completed!=last:print('Completed new MA seed weights:',completed,'/24',flush=True);last=completed
  time.sleep(10)
 main_python=ML/'.venv/Scripts/python.exe';ma_python=ML/'artifacts/ma_gbt_runtime/.venv/Scripts/python.exe'
 env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'};env.pop('PYTHONPATH',None)
 def run(python,script,ma=False):
  current={**env}
  if ma:current['PYTHONPATH']=str(ML/'artifacts/ma_gbt_runtime/upstream')
  subprocess.run([str(python),str(ML/'src'/script)],cwd=ML.parent,env=current,check=True)
 run(main_python,'freeze_report_seed_study.py')
 run(ma_python,'evaluate_report_ma_seeds.py',True)
 run(main_python,'evaluate_report_seeds.py')
 run(main_python,'publish_report_seed_results.py')
 run(main_python,'verify_report_seed_results.py')
 print('Report five-seed completion and all audits passed.',flush=True)

if __name__=='__main__':main()

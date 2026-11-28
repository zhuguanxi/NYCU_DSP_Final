"""Capture only important non-identifying research environment versions."""
import json,subprocess,sys
from importlib.metadata import version
from pathlib import Path

ML=Path(__file__).resolve().parents[1]
PACKAGES=('xgboost','lightgbm','scikit-learn','numpy','pandas','scipy','joblib','pyarrow','matplotlib')
def main():
    command="[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);$o=Get-CimInstance Win32_OperatingSystem;$r=Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion';[PSCustomObject]@{name=$o.Caption;architecture=$o.OSArchitecture;kernel=$o.Version;build=$o.BuildNumber;revision=$r.UBR;display_version=$r.DisplayVersion}|ConvertTo-Json -Compress"
    operating=json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],encoding='utf-8-sig'))
    main_env=dict(python=sys.version.split()[0],libraries={k:version(k) for k in PACKAGES})
    other={}
    snippet="import sys,json,importlib.metadata as m;print(json.dumps(dict(python=sys.version.split()[0],libraries={k:m.version(k) for k in ['torch','scikit-learn','numpy','pandas']})))"
    for key,path in [('KEDGN_GPU','artifacts/related_work/kedgn_gpu_runtime/.venv/Scripts/python.exe'),('MA_GBT','artifacts/ma_gbt_runtime/.venv/Scripts/python.exe')]:
        other[key]=json.loads(subprocess.check_output([str(ML/path),'-c',snippet],encoding='utf-8'))
    git=subprocess.check_output(['git','--version'],encoding='utf-8').strip()
    uv=subprocess.check_output(['uv','--version'],encoding='utf-8').strip()
    data=dict(date='2026-11-28',OS=operating,main_environment=main_env,separate_environments=other,tools=dict(git=git,uv=uv),
        scope='Current important runtime versions, not a claim that every historical experiment used one environment')
    (ML/'reports/P65_environment.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    libs=main_env['libraries'];kg=other['KEDGN_GPU'];ma=other['MA_GBT']
    lines=['# 重要實驗環境','',
        '讀值日期：2026-11-28。主環境用於表格模型與結果整理；KEDGN及MA-GBT保留獨立環境。以下列重要套件，不宣稱歷史所有實驗都用同一環境。','',
        '| 類別 | 重要資訊 |','| --- | --- |',
        f'| OS | {operating["name"]}，64位元；Build {operating["build"]}.{operating["revision"]} |',
        f'| Programming Language | Python {main_env["python"]}（主環境）；Python {kg["python"]}（KEDGN／MA） |',
        f'| Tools | {git}；{uv.split(" (")[0]} |',
        f'| Libraries | XGBoost {libs["xgboost"]}、LightGBM {libs["lightgbm"]}、scikit-learn {libs["scikit-learn"]}、NumPy {libs["numpy"]}、pandas {libs["pandas"]}、Matplotlib {libs["matplotlib"]}；KEDGN用PyTorch {kg["libraries"]["torch"]} |','',
        '## 主要套件在不同環境的版本','',
        '| 套件 | 主環境 | KEDGN GPU | MA-GBT |','| --- | --- | --- | --- |',
        *[f'| {name} | {libs.get(name,"—")} | {kg["libraries"].get(name,"—")} | {ma["libraries"].get(name,"—")} |' for name in ('xgboost','lightgbm','scikit-learn','numpy','pandas','torch')],'',
        f'主環境另用SciPy {libs["scipy"]}、Joblib {libs["joblib"]}、PyArrow {libs["pyarrow"]}。表格中的「—」表示此摘要未列版本，不代表一定未安裝。','',
        'KEDGN使用PyTorch隨附的CUDA12.1 runtime（cu121）。GPU為NVIDIA GeForce RTX4060 Laptop GPU，8GB；不將此runtime版本寫成另外安裝的完整CUDA Toolkit版本。Full／R1本輪colsample0.9保持原CPU hist演算法，以隔離特徵抽樣變更。','',
        '[機器讀取摘要](reports/P65_environment.json)。完整歷史套件與裝置紀錄以各階段本機environment／protocol檔為準；KEDGN作者requirements與本機環境的差異已在[P64稽核](reports/P64_kedgn_implementation_audit.md)揭露。']
    (ML/'ENVIRONMENT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(data,ensure_ascii=True,indent=2))

if __name__=='__main__':main()

# P13 原版 MA-GBT 對照規格

使用 Stempfle 等人公布的 `malearn.estimators.classification.MAGBTClassifier`，固定作者程式庫提交 `fbbdd0ca2c9d0d8072f49d82ad35f4391098863f`。以 P12 A/B 既有的 Full／Elastic 特徵和折分配做對照。[ICML 2025 論文](https://proceedings.mlr.press/v267/stempfle25a.html) · [作者程式](https://github.com/Healthy-AI/malearn/tree/fbbdd0ca2c9d0d8072f49d82ad35f4391098863f)

## 前處理

- 沿用 P12 首 48 小時特徵、住院死亡標籤、原始三折和凍結 XGBoost 的欄位順序。每折各用當初該折訓練的中位數及類別編碼。最後用原全 A/B 擬合的中位數及類別編碼。
- 分離兩套 Python 環境以配合作者使用的 scikit-learn 私有介面。用原環境讀取凍結的八套前處理，匯出參數，對每套的 8,000 筆 A/B 輸出逐值核對，也驗證類別缺失和未知類別的行為。
- MA-GBT 取得同一份轉換值，另帶一份由補值前 NaN 形成的缺失矩陣。同一原始類別欄位的各 one-hot 欄共用該欄原始缺失狀態。已定義的量測次數、缺失指標和統計量定義指標視為已知值。此設定保留 P12 既有前處理，作者論文中的零值／MICE 補值屬於論文原有實驗設定。

## 訓練與選擇

- 只使用 A/B 的 8,000 次 ICU stay、1,122 次住院死亡與原本的三折。以作者設定範圍中的 10 棵樹、log loss、學習率 0.1、深度 2、全量抽樣，固定隨機種子 42。
- Full 比較三個缺失懲罰 `alpha` 0.1、10、1000。沿用論文選擇原則，在平均驗證 AUROC 至少達到候選最佳值 95% 的設定中，選擇平均缺失依賴率 ρ 最低者。ρ 是預測路徑使用至少一個補值特徵的樣本比例。同值依較高 AUROC、較小 alpha 排序。Elastic 共用 Full 所選的 alpha，並回報同一三折的驗證表現。
- Full／Elastic 最後各以全部 A/B 擬合一次。凍結模型及記錄 SHA-256 後，才讀取 Set C、eICU Demo 及測試結果。
- 全程 CPU。原版近似相等比較的迴圈在 Windows 較慢，因此訓練時僅替換該模組的 NumPy 有限純量比較入口，計算同一個 NumPy 1.25.2 不等式。固定提交的作者原始檔與樹切分、梯度、缺失懲罰邏輯原樣執行。正式訓練前，程式以 A/B 核對比較邊界、三棵原版樹和加速版樹的切分與節點值、全部第 0 折的預測和 ρ，要求逐值一致才繼續。

## 評估

- Set C 使用 P9 的同一 4,000 筆資料、13 個遮蔽情境和 32 次匹配遮蔽抽樣。兩套 Python 環境重建 P9 抽樣後核對位置雜湊，再逐次配對 P9 原凍結 XGBoost 與 P11 GCS-only 結果。
- eICU Demo 使用 P8 的 782 次 stay、719 人和 91 次住院死亡。同時評估主要的整點設定與原始取樣間隔設定，沿用 1,000 次以患者為單位的 bootstrap，計算相對兩種對照的 AP、AUROC 配對差與 95% 區間。
- 各情境保留 AP、AUROC、Brier 和 0.5 門檻敏感度。完整摘要、各次遮蔽彙總指標、模型與個別預測存於 Git 排除的 `ML/artifacts/ma_gbt_p12/`。Git 中只提交程式、規格與不含個別資料的結果報告。

## 重跑

在 `ML` 目錄執行。先準備原 XGBoost 環境及 P7、P8、P9、P11 產物。以下為 Windows PowerShell 命令，另需可由 `uv` 找到的 Python 3.11。

```powershell
$maRuntime = Join-Path (Get-Location) 'artifacts\ma_gbt_runtime'
New-Item -ItemType Directory -Path $maRuntime -Force | Out-Null
$env:UV_CACHE_DIR = Join-Path $maRuntime 'uv-cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$maSource = Join-Path $maRuntime 'upstream'
$maVenv = Join-Path $maRuntime '.venv'
$maPython = Join-Path $maVenv 'Scripts\python.exe'
git clone https://github.com/Healthy-AI/malearn.git $maSource
git -C $maSource checkout --detach fbbdd0ca2c9d0d8072f49d82ad35f4391098863f
uv venv --python 3.11 $maVenv
uv pip install --python $maPython --index-url https://download.pytorch.org/whl/cpu 'torch==2.1.2+cpu'
uv pip install --python $maPython numpy==1.25.2 scipy==1.9.3 pandas==2.2.3 scikit-learn==1.5.2 xgboost==2.1.2 skorch==1.0.0 gurobipy==12.0.0 pyyaml==6.0.2 pyarrow==18.1.0 joblib==1.4.2
$env:PYTHONPATH = $maSource
& .\.venv\Scripts\python.exe src/ma_gbt_p12_inputs.py
& $maPython src/run_ma_gbt_p12.py --workers 4
& .\.venv\Scripts\python.exe src/evaluate_ma_gbt_p12.py --export-references
& $maPython src/evaluate_ma_gbt_p12.py
```

腳本保護原輸入及既有輸出。重跑時先保留上一輪 `artifacts/ma_gbt_p12/` 的結果，再從新的同名輸出路徑執行。

# P21 預測效能與跨資料集的樹模型比較

P12 預測窗、住院死亡標籤、8,000 筆 A/B、原三折、4,000 筆 Set C 和 eICU Demo 原對齊沿用前輪。p12 原始 435 欄保持可作基準。新候選把數值 NaN 交給模型，Gender／ICU 類型採固定 one-hot 及缺失 NaN。

## A/B 設計

- 比較 XGBoost、LightGBM 和 CatBoost 各兩種樹設定。每種設定比較 300 與 650 棵樹，使用同一組套件和 CPU。完整特徵組和跨資料集對齊特徵組都做原始訓練。對齊組移除 eICU 原管線完全未映射的鎂、尿量與機械通氣欄位，另含一種缺失增強。淺 XGBoost 與淺 CatBoost 也比較完整特徵加增強。
- A/B 的跨資料集缺失模擬固定遮蔽鎂、尿量、機械通氣 100%、GCS 25%、乳酸 45%、侵入式 MAP 70%、ICU 類型 90% 的原有觀測。模型增強以原列 0.75、遮蔽列 0.25 的權重訓練。驗證使用兩組獨立固定抽樣，另有 GCS 全遮。
- 原凍結 XGBoost Full 用既有每折模型並核對 OOF。GCS-only Full 依 P11 60% 強度和原折重擬以取得相同情境的 A/B 預測。最終兩者直接使用凍結全 A/B 模型。
- 候選的分數以折 AP 等權平均。兩個事前目標分別是未遮蔽 AP，以及 0.65 × 未遮蔽 AP + 0.25 × 兩組跨資料集缺失模擬平均 AP + 0.10 × GCS 全遮 AP。每個目標的前四個新候選可和兩個凍結基準用 25%、50%、75% 權重做機率混合。每個目標選一個配置，權重與棵數均只由 A/B 決定。

## 比較

在兩個配置凍結後才讀 Set C 和 eICU 評估資料。Set C 比較未遮蔽和原 P9 的 13 種遮蔽，且使用同一組 32 次匹配抽樣。未遮蔽的 AP／AUROC 差採原 P7 2,000 次配對 bootstrap。eICU 兩種取樣採原 1,000 次患者層級配對 bootstrap。對照包含原凍結 Full、GCS-only Full 與 Elastic。Set C 與 eICU 都已在前輪使用，這輪的分數衡量與歷史構型的同群比較。

## 重跑

在 `ML` 執行，使用既有 `.venv`。CatBoost 額外套件已安裝到 Git 排除目錄。下列安裝命令可在新的副本準備同一版本。

```powershell
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$packageRoot = Join-Path (Get-Location) 'artifacts\p21_cpu_packages'
$env:UV_CACHE_DIR = Join-Path $packageRoot 'uv-cache'
$env:PYTHONPATH = Join-Path $packageRoot 'site'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
uv pip install --python $py --target $env:PYTHONPATH --no-deps 'catboost==1.2.10' 'graphviz==0.21' 'plotly==6.5.2'
& $py src/train_predictive_boosting.py
if ($LASTEXITCODE -eq 0) {
    & $py src/evaluate_predictive_boosting.py
}
```

來源雜湊、配置、完整彙總和模型保留在 Git 排除的 `ML/artifacts/p21_predictive_boosting/`。重跑前先保留既有輸出。

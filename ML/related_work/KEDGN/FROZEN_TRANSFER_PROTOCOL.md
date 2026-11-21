# KEDGN 凍結模型的 Set C 與 eICU Demo 評估

依使用者 2026 年 11 月 21 日的指示，R1 取消，保留 KEDGN 凍結模型的轉移評估。全程 CPU，模型擬合 0 次。KEDGN 沿用已完成作者復現的五個驗證集最佳 checkpoint，不用外部測試分數選種子或調參。

## 凍結來源與比較問題

五個模型使用作者 Physionet 第 5 折的同一訓練／驗證／測試切分。訓練為 Set A 的 3,198 筆住院，驗證 400 筆，作者測試 399 筆。先逐種子重現原作者測試 AP／AUROC，以核對模型權重、欄位順序、靜態編碼、縮放與推論。

訓練資料差異需與結果一起呈現。本輪衡量 Set A 訓練之凍結 KEDGN 的轉移能力，本組 Full／Elastic、GCS-only、P26／P28／P33 使用 A+B 訓練。若要評估共同訓練量下的架構差異，可再用 A+B 重新訓練 KEDGN。這是額外實驗，不混入本輪凍結比較。

## 輸入與隊列

- Set C 保留本組原 4,000 筆住院及 585 次住院死亡。動態與靜態序列使用作者原始解析規則重建，先在 30 筆 Set A 紀錄與已保存的作者前處理陣列逐值核對。核對 Set C 病例與 KEDGN 訓練病例互斥。所有病例保留，包括作者先前 blacklist 可能排除的零觀測病例。
- eICU Demo 沿用原 782 次住院、91 次住院死亡，保留整點與原始間隔兩種取樣，尿量採既定 mL 對齊。共用輸入對齊作者的 36 個動態通道及 5 個靜態欄位，靜態缺失轉為 −1，再使用作者的 9 維擴展編碼。未知 ICU 類別使用全零 one-hot。
- 作者時間窗為分鐘小於 2,880，依此重建輸入。KEDGN 的動態缺失沿用數值大於零的原作者 mask。數值與靜態縮放只由原 3,198 筆訓練資料重建，凍結用於所有測試。作者的非正值處理和靜態標準化行為保留，與本組表格清理方式的差異在報告揭露。
- 原始間隔 eICU 可能超過 215 個時間點。模型圖循環依實際長度運算，保留全部時間點，依 batch 最大長度 padding，padding 至少 215，避免作者前處理在 T=60 時套用另一種時間單位。逐一保持病例順序和原始長度。

## 指標與統計

主要 KEDGN 分數事先定為五個凍結種子的機率等權平均。另報每個種子的 AP、AUROC、Brier 和五種子指標平均／標準差。平均機率模型的效能與種子指標平均各自列出。

比較原凍結 Full／Elastic、GCS-only Full／Elastic、P26、P28 和 P33。所有分數在相同病例、標籤和相應取樣設定上核對，原模型的 AP／AUROC／Brier 須重現既有紀錄。Set C 使用既定 2,000 次住院次配對重抽，eICU 使用既定 1,000 次患者配對重抽。報告差值、95% 區間與固定 0.5 門檻敏感度。本輪評估未遮蔽資料。

## 重跑

從 ML 執行，以原主環境準備輸入和比較，以文獻 CPU Torch 環境做凍結 KEDGN 推論。日誌、逐筆預測、輸入、scaler 和 checkpoint 雜湊位於 Git 排除的 artifacts/related_work/KEDGN/frozen_transfer_20261121/。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.\.venv\Scripts\python.exe' related_work/kedgn_transfer_inputs.py
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts/related_work/cpu_packages'
& '.\artifacts\ma_gbt_runtime\.venv\Scripts\python.exe' related_work/infer_frozen_kedgn.py
# 比較階段使用 P21 主環境附加套件
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts/p21_cpu_packages/site'
& '.\.venv\Scripts\python.exe' related_work/compare_frozen_kedgn.py
```

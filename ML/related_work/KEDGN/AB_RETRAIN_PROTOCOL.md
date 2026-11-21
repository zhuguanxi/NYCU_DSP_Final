# P54：KEDGN A+B 重訓與凍結轉移規格

2026 年 11 月 21 日，依使用者指示執行 KEDGN A+B 重訓，允許 GPU。R1 不復現。程式與結果分開提交並推 GitHub，資料、逐筆預測、折外預測、checkpoint 與權重只留在本機 Git 排除目錄。

## A/B 開發與凍結規則

使用原 P12 A/B 8,000 次住院、1,122 次死亡及既有 `folds3.csv` 三折。保留全部病例，包括零觀測病例。每折用其餘兩折估計動態與靜態縮放，使用本折驗證；全 A+B 最終模型的縮放只由全 A+B 估計。任何選參或輪次選擇都不讀 Set C 或 eICU 的分數。

固定作者已完成復現的模型參數：36 動態通道、9 維靜態、hidden 8、query 5、node 9、BERT 768 維與 GPT 欄位表示、rarity alpha 1、Adam lr 0.001。這輪不搜尋其他模型參數。每折 seed 0 訓練 10 輪，保存每輪驗證機率；將三個驗證折還原成完整 A/B 折外預測，按 **pooled OOF AP 最大**選共同輪次，平手取較早輪次。

沿用作者兩類平衡採樣：陽性索引複製三次，每次更新各取陰性與陽性 128 筆，有效 batch 256。預設整批 256，若指定較小 microbatch 則累積梯度後做一次 optimizer 更新，保留作者每輪 batch 計算與採樣；模型無 batch normalization 或 dropout。FP32，不使用混合精度或 TF32；啟用確定性運算。

原作者逐觀測節點的動態索引在 GPU 上效率偏低，改用 `kedgn_dense_runtime.py` 的等價遮罩運算：參數、圖、門控與每病例結束狀態保留，文字表示及其門控投影在時間迴圈外重用，未觀測節點用遮罩保留原狀態。正式訓練前核對原作者 CPU 與 GPU 批次版本的輸出及全部有效參數梯度，包含零觀測病例，另核對整批與分批梯度。初次 microbatch 64 的逐點執行因效率中止，紀錄保留在 `ab_gpu_20261121/`，未用於選輪次或測試。

作者驗證程式以單一正類 logit 的 sigmoid 算 AP，原最終測試使用雙類 softmax。本輪為配合最終評估，輪次選擇統一用雙類 softmax AP；此差異明確保留。固定參數仍來自作者復現，未用外部成績修改。

共同輪次封存後，以全 A+B 訓練 seed 0–4 五個最終模型，不另從訓練輪次或種子中挑最佳測試分數。主要分數為五種子機率等權平均，並列每種子及指標平均／SD。總計 3 次開發擬合、5 次全 A+B 擬合。每輪保存 optimizer、抽樣索引、亂數狀態及模型以支援恢復；已完成的權重以雜湊核對。

## 輸入與測試

P12 A/B 從原始 Set A/B 紀錄，使用 P53 同一 `raw_record` 解析：依欄位名稱對齊入院靜態資料、36 通道、正值 mask、分鐘小於 2,880，保留所有時間點，缺失靜態先為 −1，沿用作者靜態轉換。前處理公式固定，只有縮放統計依訓練折重新估計。

完成 `selection.json` 和五個最終權重的 `frozen.json` 後才執行測試推論。Set C 4,000／585、eICU Demo 782／91（尿量 mL 對齊，整點及原始間隔）直接沿用 P53 已核對的輸入及死亡標籤。Set C 與 A/B 的病例互斥需再核對。eICU 長序列保留，至少 padding 215，避免作者 T=60 的不同時間單位分支。

比較 P53 Set A 凍結五種子平均、Full／Elastic、GCS-only Full／Elastic、P26／P28／P33。先核對舊模型來源與保存預測，再於相同病例計算 AP／AUROC／Brier／0.5 門檻敏感度。Set C 沿用 2,000 次住院配對重抽（seed 20261013）；eICU 沿用 1,000 次患者配對重抽（seed 42）。只做未遮蔽測試，本輪不新增缺失壓力測試。

## 執行與產物

從 ML 執行；先建立獨立環境，不修改 P53 的 CPU 環境。

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) 'artifacts/related_work/uv_cache'
uv venv artifacts/related_work/kedgn_gpu_runtime/.venv --python 'C:/Users/user/AppData/Local/Programs/Python/Python311/python.exe'
$gpu = '.\artifacts\related_work\kedgn_gpu_runtime\.venv\Scripts\python.exe'
uv pip install --python $gpu --index-url https://download.pytorch.org/whl/cu121 'torch==2.1.2+cu121'
uv pip install --python $gpu 'numpy==1.25.2' 'scikit-learn==1.3.2' 'pandas==2.1.4' 'pyarrow==14.0.2' 'einops==0.8.0' 'packaging==24.2'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.\.venv\Scripts\python.exe' related_work/prepare_kedgn_ab.py
uv pip freeze --python $gpu | Out-File artifacts/related_work/KEDGN/ab_dense_gpu_20261121/environment.txt -Encoding utf8
& $gpu related_work/run_kedgn_ab.py verify
& $gpu related_work/run_kedgn_ab.py fit
& $gpu related_work/verify_kedgn_ab_frozen.py
& $gpu related_work/run_kedgn_ab.py infer
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts/p21_cpu_packages/site'
& '.\.venv\Scripts\python.exe' related_work/evaluate_kedgn_ab.py
```

預設根目錄 `artifacts/related_work/KEDGN/ab_dense_gpu_20261121/`，準備／訓練／比較腳本均支援 `--output` 指定新的目錄（準備腳本指向其 inputs 子目錄）。原始輸入準備若已有輸出會停止以保護既有檔案；訓練從完整輪次恢復，已完成模型不覆寫。訓練需提供可用 CUDA GPU；可明確指定 `--device cpu` 另做 CPU 執行。完整套件清單保存於本機產物及結果彙總。

CUDA 安裝版本依 [PyTorch 官方歷史版本說明](https://pytorch.org/get-started/previous-versions/#v212)。原作者版本與文獻見 [KEDGN README](README.md)，前輪測試見 [P53](../../reports/P53_kedgn_frozen_transfer_results.md)。

## 判讀限制

A+B KEDGN 與本組表格模型的訓練病例數對齊，但特徵表示、作者採樣與前處理仍不同；本輪也使用五種子平均。原 Set A 版本有 3,198 訓練病例及獨立驗證選 checkpoint；A+B 版本使用三折選共同輪次，兩者差值不能完全歸因於訓練量。Set C 已反覆查看，eICU Demo 僅 91 次死亡；配對區間描述固定模型的抽樣差異，未包含完整重新訓練與跨醫院變異。

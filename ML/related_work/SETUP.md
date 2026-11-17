# 文獻復現的 CPU 入口

作者程式、公開資料、CPU 工作副本、日誌及模型都放在 `ML/artifacts/related_work/`。每篇有自己的子目錄。作者程式的原封存版本在 `upstream/`，每次執行再用獨立名稱建立工作副本。提交中只保留重跑程式、套件清單和彙總。

## 共用資料與環境

- WaveGNN、ATENet、ViTST 使用作者所引用的 Raindrop P12 前處理資料。ZIP 為 Figshare 檔案 34683085，MD5 `ec95a6cae6077e4e7bd14c3b1719ddb8`。將其中的 `P12data/processed_data/` 與 `P12data/splits/` 放在 `artifacts/related_work/paper_data/`。本機已核對 11,988 筆死亡標籤與官方 A/B/C 一致。 [作者引用的前處理資料](https://doi.org/10.6084/m9.figshare.19514341.v1) · [官方原始資料](https://physionet.org/content/challenge-2012/1.0.0/)
- KEDGN 與 Warpformer 的作者死亡設定用官方 Set A。讀取專案上層 `data/` 內既有的 `set-a/` 和 `Outcomes-a.txt`。
- Python 3.11.2 與 Torch 2.1.2+cpu 沿用既有 MA-GBT 環境。額外套件裝入隔離的 `cpu_packages/`。本機已完成，不必再安裝。
- ViTST 預訓練權重已下載並核對。Swin 版本 `68dc76680a5bf3bdf670669f3025dc9be2e30781`，RoBERTa 版本 `e2da8e2f811d1448a5b465c236feacd80ffbac7b`。本機位置是 `ViTST/pretrained/swin/`、`ViTST/pretrained/roberta/`。 [Swin 權重版本](https://huggingface.co/microsoft/swin-base-patch4-window7-224-in22k/tree/68dc76680a5bf3bdf670669f3025dc9be2e30781) · [RoBERTa 權重版本](https://huggingface.co/FacebookAI/roberta-base/tree/e2da8e2f811d1448a5b465c236feacd80ffbac7b)

在 `ML` 目錄執行。安裝指令只用於缺少隔離套件時。

```powershell
$py = '.\artifacts\ma_gbt_runtime\.venv\Scripts\python.exe'
$overlay = Join-Path (Get-Location) 'artifacts/related_work/cpu_packages'
$env:UV_CACHE_DIR = Join-Path (Get-Location) 'artifacts/related_work/uv_cache'
uv pip install --python $py --target $overlay --no-deps -r related_work/requirements-cpu-overlay.txt
uv pip install --python $py --target $overlay --no-deps --index-url https://download.pytorch.org/whl/cpu 'torchvision==0.16.2+cpu'
```

## 執行

```powershell
$py = '.\artifacts\ma_gbt_runtime\.venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts/related_work/cpu_packages'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $py related_work/run_cpu.py WaveGNN --profile smoke --tag local_wave_check
& $py related_work/run_cpu.py ATENet --profile smoke --tag local_ate_check
& $py related_work/run_cpu.py KEDGN --profile smoke --tag local_kedgn_check
& $py related_work/run_cpu.py Warpformer --profile smoke --tag local_warp_check
& $py related_work/run_cpu.py ViTST --profile smoke --tag local_vit_check
```

`smoke` 只驗證作者程式在真實資料上的前處理、訓練、驗證及測試。採固定種子從作者切分內各自抽出兩個類別，所以 AP 和 AUROC 不能拿來對比原論文。較大的 `paper_schedule` 才使用完整資料及論文排程，其驗證狀態以各模型報告為準。ViTST 另將訓練期間的評估資料改為驗證折，並在試跑凍結公開預訓練編碼器。

正式比較 DSP P33 應另建 A/B → Set C 的適配。作者 P12 第一折訓練含 3,201 筆官方 Set C，驗證含 389 筆，測試含 408 筆。兩個評估設定會用到不同的訓練病例。

# P55：競爭基準的共同缺失測試

2026-11-22，依研究目標補齊 KEDGN 的缺失韌性比較。使用者確認本輪只有 eICU Demo，不等待完整版；不製作簡報或封面，交付完整實驗、證據索引與研究敘事。R1 與其他先前取消的論文排程不自動重啟。

## 固定設計

本輪不訓練、不選參。使用 P53 的 Set A 五個 KEDGN 和 P54 的 A+B 五個 KEDGN，沿用各自凍結權重與訓練統計。比較原 Full／Elastic、GCS-only Full／Elastic、P26／P28／P33。全部保留 Set C 4,000 次住院與 585 次住院死亡，清潔分數先重現 P53/P54；病例與標籤逐值核對。

預先固定八個情境：未遮蔽、GCS 全遮、乳酸全遮、侵入式 MAP 全遮、鎂全遮、尿量全遮、機械通氣全遮、上述六項同時全遮。表格清除相應數值、可用旗標與時間摘要，P33 同步遮蔽序列通道。KEDGN 將對應通道原始值改為 0，再依凍結統計重建作者正值 mask；時間網格與原長度保留，避免遮蔽時重新編排時間位置或改變其他通道的間隔定義。

GPU 僅執行已驗證的 KEDGN 等價批次推論，CPU 推論其他凍結模型與統計。主要 KEDGN 仍為五個種子的機率等權平均，另列種子結果。Set A 版訓練量與 A+B 版不同，兩者差異包含選輪次方式；與本組表格模型則仍有表示及前處理差異。

## 評估

所有情境列 AP、AUROC、Brier、0.5 門檻敏感度及相對自身未遮蔽的變化。主要配對推論限定 GCS 全遮與六項全遮，計算 A+B KEDGN 相對 Set A KEDGN、Full、P26、P33 的差，沿用 2,000 次 Set C 住院配對重抽、seed 20261013。其餘單欄結果用於機制描述，不用來挑模型或參數。這不是正式醫院量測缺失機制的完整模擬，也不把固定隊列區間當成重新訓練或跨院變異。

逐筆輸出在 `artifacts/p55_kedgn_missingness/`，資料與權重不進 Git。程式與彙總結果分開提交。

## 執行

從 ML 執行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=Join-Path (Get-Location) 'artifacts/p21_cpu_packages/site'
& '.\.venv\Scripts\python.exe' related_work/evaluate_kedgn_missingness.py prepare
$env:CUDA_VISIBLE_DEVICES='0'
& '.\artifacts\related_work\kedgn_gpu_runtime\.venv\Scripts\python.exe' related_work/infer_kedgn_missingness.py
$env:CUDA_VISIBLE_DEVICES='-1'
& '.\.venv\Scripts\python.exe' related_work/evaluate_kedgn_missingness.py compare
```

P56 的候選單一配置會另外寫 A/B 選擇規格，且在讀取新候選的 Set C／Demo 結果前封存。eICU 改善是否成立由資料決定，不以反覆查看 Demo 來保證正向結果。

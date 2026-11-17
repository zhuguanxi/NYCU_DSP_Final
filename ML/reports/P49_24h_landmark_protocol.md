# P49 ICU 前 24 小時住院死亡預測

P48 的外部比較中，研究模型使用 48 小時，APACHE IVa 系統使用 ICU 前 24 小時。這輪用 24 小時各自計分，並檢查加上時間窗摘要與 GCS 缺失分流是否優於同資料下的簡單 XGBoost。 [P48 結果](P48_clinical_score_baselines_results.md)

## 開發與凍結

- P12 限年齡至少 18、官方住院天數至少 2，且 Survival 至少 2 或已知存活者的 −1。這是以整天欄位定義的 24 小時風險集代理。所有預測欄位從原始觀測重新建立，範圍為 ICU 分鐘 0–1,440（含）。住院時間與結果欄位只供納入與標籤，未提供給模型。
- 延用 A/B 三折、435 個表格欄位、原變數清理。時間摘要使用同樣 20 通道與每小時中位數，早／晚分別為 0–6 小時與 18–24 小時，距最後觀測以 24 小時計算。
- 比較同資料的摘要 XGBoost、加時間摘要 XGBoost、時間摘要加 GCS 60% 遮蔽增強、時間摘要加原 P26 的跨庫遮蔽增強。三款時間摘要模型在 GCS 無觀測時可混合 0%、50%、100% 的 P26 式無 GCS LightGBM。延用 P26 的樹模型、缺失視圖及加權目標。
- A/B 未遮蔽和 GCS 全遮的 AP 皆須達基準摘要 XGBoost 減 0.005，再選加權 AP 最高者。兩個分數同分時取未遮蔽 AP 較高者。凍結設定和模型後才讀取 Set C 與 eICU。

## 測試

- Set C 使用同一 24 小時代理隊列。比較未遮蔽、GCS 全遮與一組按 P26 的尿量部分恢復模擬遮蔽。以 2,000 次住院次配對重抽估計相對 24 小時摘要 XGBoost 的 AP、AUROC 差；同隊列的原凍結 Full／Elastic 48 小時分數供觀察窗背景。
- eICU Demo 以成人、本次住院第一次 ICU、在 ICU 至少 24 小時、醫院結局發生於 24 小時後且有住院死亡標籤為入組條件。所有量測與可用的登錄／修訂時間均不得超過 1,440 分鐘。有登錄欄但時點缺失的列排除。監測表使用發布的觀測時點。沿用整點、原始間隔和尿量毫升對齊。
- 主要外部比較用原 24 小時新隊列中有有效 APACHE IVa 住院死亡機率者。另在 P48 原 48 小時隊列的 APACHE 可用子集進行觀察窗敏感度比較。1,000 次患者配對重抽，報告凍結選定模型對 APACHE IVa、對 24 小時摘要 XGBoost 的差值。APACHE 的資料窗口有文獻定義，個別分數實際可用時間在發布表無法確認。 [eICU 資料論文](https://doi.org/10.1038/sdata.2018.178)

## 重跑

在 `ML` 使用 P21 環境與原始 P12、eICU Demo 2.0.1（含 P48 的 APACHE 結果表）。全程 CPU，擬合與評估分開，既有輸出不覆寫。模型、雜湊和彙總留在 Git 排除的 `artifacts/p49_24h_landmark/`。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts\p21_cpu_packages\site'
$py = '.\.venv\Scripts\python.exe'
& $py src/evaluate_24h_landmark.py fit
if ($LASTEXITCODE -eq 0) {
    & $py src/evaluate_24h_landmark.py evaluate
}
```

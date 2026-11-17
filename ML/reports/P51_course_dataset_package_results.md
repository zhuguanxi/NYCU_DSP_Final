# P51 課程資料包與核對結果

依課程的資料交付規格，把本研究主要模型使用的前處理資料與 README 放在同一 ZIP。資料包留在本機 Git 排除的 `artifacts/course_submission/icu_mortality_dataset/`。ZIP 有 28 個成員、33,239,856 bytes，SHA-256 為 `7144a58d082a3cbec1ca03bdbbb0ce4c0959e3d24b2f90cb795063e918bd0922`。

| 資料與觀察窗 | 住院次 | 住院死亡 | 有效 APACHE IVa |
| --- | ---: | ---: | ---: |
| P12 A/B，48 小時 | 8,000 | 1,122 | — |
| P12 Set C，48 小時 | 4,000 | 585 | — |
| P12 A/B，24 小時代理隊列 | 7,817 | 1,071 | — |
| P12 Set C，24 小時代理隊列 | 3,906 | 549 | — |
| eICU Demo，48 小時 | 782 | 91 | 660 |
| eICU Demo，24 小時 | 1,398 | 121 | 1,166 |

入包資料含每個隊列的 435 個表格欄與 140 個時間摘要、48 小時序列輸入、標籤、A/B 原三折、eICU 原五折、Set C 既有遮蔽、SAPS-I／SOFA 和 APACHE IVa 比較值。eICU 48 小時保留原／尿量對齊及兩種取樣，24 小時保留尿量對齊及兩種取樣。欄位、標籤、臨床比較分數與遮蔽分檔，包內替代鍵供對齊。來源資料與衍生表授權在包內 README。 [P12 資料及授權](https://physionet.org/content/challenge-2012/1.0.0/) · [eICU Demo 資料及授權](https://physionet.org/content/eicu-crd-demo/2.0.1/)

CPU 匯出後重新讀取每一個 Parquet 和 NPZ，逐值核對欄位、列序和陣列。再用 ZIP 成員逐一檢查 CRC、SHA-256、樣本數、死亡數、組別、遮蔽列序及兩個 eICU 隊列的共同資料。核對了 Demo 的患者鍵、院鍵、院鍵區塊在每套五折都各自只落在一折。ZIP 中沒有完整 eICU、原始資料表、受限資料、模型權重或模型預測。

## 重建

在 `ML` 使用 P21 環境與截至 P50 的本機資料和產物。輸出位置已有資料時停止，避免覆寫。並列的 `SHA256SUMS.txt` 可用來核對整個 ZIP。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts\p21_cpu_packages\site'
& '.\.venv\Scripts\python.exe' src/package_course_dataset.py
```

ZIP 使用通用檔名，正式上傳檔名需代入實際組別、Part 和隊號。本輪完成資料交付的本機檔案與驗證。P46 的完整版 eICU 獨立驗證仍待有授權的資料。 [P46 固定規格](P46_eicu_adaptation_confirmation_protocol.md)

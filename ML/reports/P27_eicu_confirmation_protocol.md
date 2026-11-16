# P27 完整 eICU 的凍結確認流程

使用已凍結的 P26 分流模型、P22 時間窗與 GCS-only Elastic。以 A/B 選定的原設定給分，評估時沿用 P8 入 ICU 48 小時窗與住院死亡，以及 P23 明確 mL 尿量列。整點為主，保留原始間隔敏感性。使用患者層級配對 bootstrap，比較 AP、AUROC、Brier 與 0.5 門檻敏感度。

本機現有 eICU Demo。完整版需要經認證帳號、規定訓練和資料使用協議，由有權限的研究者提供原始表。公開 Demo 是完整版的子集，所以完整檔案不能直接當全新的確認隊列。[完整版](https://physionet.org/content/eicu-crd/2.0/) · [Demo](https://physionet.org/content/eicu-crd-demo/2.0.1/)

## 隊列與排除

1. 用原規則取得已滿 48 小時、成人、住院內首個 ICU stay 和可判定院內死亡標籤的隊列。
2. 逐一確認 Demo 的公開 stay 與患者鍵能在來源檔以同一對鍵核對。排除 Demo 全部已知患者的所有 stay。若版本造成鍵無法核對，程式報錯並保留問題，需取得能核對的來源關係後才能主張患者獨立。
3. 以固定雜湊順序最多選 6,000 位合格患者，保留其所有合格 stay。程式先固定這個抽樣上限及模型，再計算分數。分批讀取原始表，只保留選定 stay 的資料供本機提取特徵。
4. Demo 測試模式只驗證隊列、分批抽取、時間窗、尿量映射及凍結計分。輸出明確標為 Demo 執行測試。

## 重跑

在 `ML` 使用 P21 CPU 環境與固定 P22、P26 模型。已授權的完整版資料可直接使用原始 `.csv.gz` 所在資料夾。

```powershell
& $py src/evaluate_eicu_confirmation.py --smoke-demo
& $py src/evaluate_eicu_confirmation.py --full-directory 'C:\path\to\eicu-crd-2.0'
```

完整模式只保留整點與原始間隔的彙總分數、涵蓋率與檔案雜湊。原始檔及中間子隊列留在本機 Git 排除目錄。完整測試與 Demo 執行測試使用各自的輸出路徑。

# P40 完整 eICU 的 P38 次要比較規格

凍結 [P38](P38_sequence_channel_ablation_results.md) 作次要外部候選。先在同一評估中重現 [P29](P29_eicu_ensemble_confirmation_protocol.md) 的 P28 對原凍結 Elastic。主要對照、6000 名合格患者上限、固定雜湊順序、原 Demo 患者排除條件、時間窗及尿量 mL 對齊沿用 P29。

- P38 使用原凍結的 18 通道四小時序列、靜態欄位、25% 權重與 ±2 對數勝算差上限。尿量與呼吸率只從卷積輸入移除。P28 表格成分沿用 P29 的尿量對齊及同一批 stay。
- 次要比較為同一 stay 的 P38 減 P28，以及 P38 減凍結 Elastic。主要讀取整點 AP，在原始間隔另報同樣比較。AP 與 AUROC 的區間沿用 1,000 次患者層級配對 bootstrap。另報 Brier 和 0.5 門檻敏感度。這是次要分析，須連同 P29 主要對照及區間解讀。
- Demo 試跑涵蓋全部合格 Demo 患者，逐值核對既有尿量對齊的 P28、P38 和原凍結 Elastic 指標。正式結果必須先與 Demo 核對識別鍵，再排除所有 Demo 患者。任何對照都不做再訓練。

在 `ML` 使用 [P21 CPU 環境](P21_predictive_boosting_protocol.md) 及 [P33 的 Torch CPU 產物](P33_hourly_sequence_protocol.md)。

```powershell
& $py src/evaluate_eicu_sequence_confirmation.py --smoke-demo
& $py src/evaluate_eicu_sequence_confirmation.py --full-directory 'C:\path\to\eicu-crd-2.0'
```

正式彙總存於 Git 排除的 `artifacts/p40_independent_eicu/`。Demo 試跑另存 `artifacts/p40_demo_smoke/`。原始資料、暫存病例資料和模型保留於 Git 排除路徑。

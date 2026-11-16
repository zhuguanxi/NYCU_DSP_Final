# P31 量測頻率欄位消融規格

比較降低對明示量測模式欄位的依賴，是否能在 P12 保留預測表現並提高模擬跨庫表現。刪除量測次數、缺失與統計量定義旗標、時間窗涵蓋小時數、距最後量測時間。保留量測值、早晚時間窗、變化摘要、靜態欄位和既有治療資訊。原生 NaN 路徑仍可利用欄位缺失，本實驗衡量移除這些明示欄位的整體影響。量測模式的可轉移性參考 [Generalization in Clinical Prediction Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC8238368/)。

沿用原 A/B 三折與 P28 的六個驗證畫面。重現 P28 折外得分後，固定比較縮減欄位的 650 棵 XGBoost 原始訓練，以及 XGBoost 和 LightGBM 的尿量部分恢復情境增強。增強使用 P26 同版的模擬遮蔽，保留原列及遮蔽副本，權重分別為 0.75 與 0.25。分數比較 P28 單獨及加入一個縮減欄位模型，機率權重 0.25、0.50、0.75、1.00。相對 P28 的未遮蔽 AP 容許降 0.003，GCS 全遮容許降 0.005，使用 P28 原加權分數與同分排序。選擇不讀評估資料。

如果 A/B 選擇加入縮減欄位模型，以全 A/B 擬合該模型，凍結後用原 Set C 和 eICU Demo 設定比較。若保留已凍結的 P28，記錄 A/B 結果，沿用已有的 P28 評估。全部 CPU。沿用 [P21 環境](P21_predictive_boosting_protocol.md) 和原 P28 產物。

```powershell
& $py src/evaluate_monitoring_ablation.py select
& $py src/evaluate_monitoring_ablation.py evaluate
```

選型、模型、完整彙總和來源雜湊保存於 Git 排除的 `ML/artifacts/p31_monitoring_ablation/`。

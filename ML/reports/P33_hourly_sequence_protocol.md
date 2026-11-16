# P33 逐小時序列模型規格

評估直接使用逐小時資料的卷積網路，能否補足 P28 基於摘要與時間窗模型的預測。沿用 48 小時輸入、住院死亡、P12 A/B 三折。取 P22 的 20 個通道，每個通道以逐小時中位數表示，將 2,880 分鐘紀錄納入最後一格，保留觀測旗標。靜態輸入使用原五欄，Gender 與 ICUType 依本研究既有值域編碼。動態與編碼後的靜態欄用各折訓練資料的中位數與 1–99 百分位截尾後標準差縮放，缺值保留獨立旗標。逐小時標準化和缺失旗標見臨床序列基準相關做法。[Harutyunyan 等](https://pmc.ncbi.nlm.nih.gov/articles/PMC6572845/) · [Che 等](https://pmc.ncbi.nlm.nih.gov/articles/PMC5904216/)

固定兩層 24 通道一維卷積，第二層膨脹係數為 2。每 12 小時對卷積表示取平均與最大值，再和靜態輸入合併。比較無缺失增強及原資料／跨庫模擬資料 0.75／0.25 加權增強，兩者架構相同。採 CPU PyTorch、三執行緒、批次 128、AdamW、學習率 0.0015，保留 8、16、24 回合的折外分數。

沿用 P28 的六個驗證畫面、分數加權和兩個 AP 門檻。先逐值重現 P28 的 A/B 分數。比較 P28 單獨，以及卷積分數的機率混合權重 0.15、0.25、0.50、1.00。若 A/B 選定卷積配置，才用全 A/B 重擬並凍結，在原 Set C 及 eICU Demo 的 P28 設定確認。若 P28 保留，沿用已完成的評估。

`ML` 的 CPU 環境使用 P21 與原 P13 的 Torch 2.1.2+cpu。輸入和模型、選型彙總存於 Git 排除的 `ML/artifacts/p33_hourly_sequence/`。

```powershell
& $py src/prepare_hourly_sequence.py
& .\artifacts\ma_gbt_runtime\.venv\Scripts\python.exe src/train_hourly_sequence.py
```

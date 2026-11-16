# P36 按通道的觀測分布增強規格

P35 的量測門檻以固定 P34 為首。本輪檢查同一個序列網路若在訓練時就接觸按通道的觀測差異，能否提高 A/B 模擬和凍結後的表現。沿用 48 小時窗、住院死亡及原 P12 A/B 三折。全程 CPU。

- 分別訓練逐小時和固定四小時表示。保持 P33 網路、CPU 最佳化器、種子與批次，取 8、16 回合折外分數。
- 每次 stay 保留原版本，另用原 P26 部分尿量恢復的跨庫缺失副本，按 P35 固定分布再對 GCS、尿量與侵入式 MAP 取樣。原與副本的損失權重分別為 0.75、0.25。只移除已觀測小時。eICU Demo 的直方圖描述觀測覆蓋，選型只使用 A/B 結果標籤。
- 比較 P28 加入一個新網路的機率權重 10%、15%、25%、40%、100%。沿用 P35 驗證情境、加權分數和以原 P33 為參照的未遮蔽與 GCS 全遮門檻。已有 P33、P34 固定配置也納入。兩個新觀測模擬只擾動序列成分，P28 表格分數不變。
- 如選中增強配置才用全 A/B 擬合一次，凍結後評估與 P33 相同的 Set C 和 eICU。若已有配置保留，既定結果適用。P29 完整版規格維持原模型。

沿用 [P33 CPU 環境](P33_hourly_sequence_protocol.md) 與 [P35 的觀測分布](P35_sequence_gate_protocol.md)。在 `ML` 執行 `& .\artifacts\ma_gbt_runtime\.venv\Scripts\python.exe src/train_profile_sequence.py`，完整選型和模型將保留於 Git 排除的 `ML/artifacts/p36_profile_augmentation/`。

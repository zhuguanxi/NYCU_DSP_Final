# P26 時間窗與 GCS 缺失分流

保留原 48 小時窗、住院死亡、A/B 原三折及凍結原評估資料。使用 P22 的 140 個時間窗特徵與原 435 欄，連帶遮蔽每個通道的原摘要及時間窗特徵。

## A/B 設計

- 一般模型比較原時間窗 XGBoost、GCS 60% 與 85% 增強、跨資料集缺失增強及 CatBoost。GCS 增強用 0.5／0.5 加權原列與受遮蔽副本，跨資料集增強為 0.75／0.25。
- 比較訓練時始終遮蔽 GCS 的 XGBoost、LightGBM、CatBoost，以及沿用的 GCS-only Full。分流候選只在 GCS 次數為零時使用 50% 或 100% 的專用模型預測。有 GCS 時使用所選一般模型。無分流的候選也在 A/B 比較。
- 驗證使用未遮蔽、GCS 全遮、兩組原 P21 跨資料集模擬及兩組尿量部分恢復模擬。後者保留原六欄遮蔽設計中的其他比例，尿量改遮掉原有觀測的 30%，近似 P23 修正後覆蓋率。P22 未遮蔽、原跨資料集模擬及 GCS 全遮 AP 逐值重現歷史 A/B 指標。
- 符合 P22 未遮蔽 AP 和 GCS-only Full 全遮 AP 各下降至多 0.005 的候選內，最大化 0.55 × 未遮蔽 AP + 0.20 × 尿量部分恢復模擬 AP + 0.10 × 原跨資料集模擬 AP + 0.15 × GCS 全遮 AP。若無符合者，於全部候選中選最高並記錄。本輪只有一個最終配置，分流條件與權重由 A/B 確定。

## 凍結評估

Set C 使用原 13 種遮蔽與配對抽樣，與凍結原 Full、GCS-only Full／Elastic、P22 時間窗比較。未遮蔽 AP 差為 2,000 次 stay 配對 bootstrap。eICU 同時比較 P8 原對齊、P23 尿量對齊（包含相應時間窗），各保留整點與原始間隔。以原 1,000 次患者層級 bootstrap 計算成對差。報告 eICU 中 GCS 無觀測的比例，以說明分流覆蓋。兩個評估隊列曾在前輪使用。

## 重跑

在 `ML` 使用 [P21 CPU 環境](P21_predictive_boosting_protocol.md) 及原 P22 到 P24 產物。

```powershell
& $py src/train_gcs_routed_temporal.py
if ($LASTEXITCODE -eq 0) {
    & $py src/evaluate_gcs_routed_temporal.py
}
```

完整模型、折外預測與來源雜湊留在 Git 排除的 `ML/artifacts/p26_gcs_routed_temporal/`。重跑前先保留既有輸出。

# P28 時間窗分流與 Elastic 的混合模型

在原 P12 A/B 三折比較 P26 分流、凍結 Elastic 與 GCS-only Elastic。研究問題是保留 P26 未遮蔽與 GCS 全遮表現時，Elastic 的分數能否增加跨資料集表現。沿用 48 小時窗與住院死亡。

## A/B 選擇

- 重建凍結 Elastic 的三折既有驗證分數，並以歷史 60% GCS 遮蔽位置重擬三個 GCS-only Elastic 訓練折。重建 P26 分流的折外分數，逐值核對六種情境彙總與選型紀錄。最終模型沿用原凍結產物。
- 比較 P26 單獨使用及與任一 Elastic 混合。按 GCS 有無觀測，Elastic 權重各選 0、0.25、0.5 或 0.75，排除雙零。比較在原機率或截斷至 [0.0000001, 0.9999999] 的 log odds 混合。P26 與兩個 Elastic 單獨的分數均留作對照，純 Elastic 不參與混合選擇。
- 沿用 P26 的未遮蔽、GCS 全遮、原跨資料集模擬與尿量部分恢復模擬，每種 AP 先按原三折等權，兩組抽樣各自等權。P26 作參照，未遮蔽和 GCS 全遮 AP 容許分別低 0.003、0.005。通過門檻者依 0.35 × 未遮蔽 AP + 0.35 × 尿量部分恢復 AP + 0.15 × 原跨資料集模擬 AP + 0.15 × GCS 全遮 AP 排序，分數同分時先看未遮蔽 AP。選擇相對 P26 提高兩種跨資料集模擬的總權重。

## 凍結評估

只評估一個 A/B 選定配置。Set C 使用原遮蔽，與 P26、凍結 Full、GCS-only Full／Elastic 比較。eICU Demo 使用原對齊和原 P23 尿量對齊，各保留整點及原始間隔，與 P26、P22 和相同對齊的 Elastic 對照。沿用未遮蔽 Set C 2,000 次 stay 配對抽樣及 eICU 1,000 次患者配對抽樣。Set C 和 Demo 皆曾用於前輪研究。

## 重跑

在 `ML` 使用 [P21 CPU 環境](P21_predictive_boosting_protocol.md)、P26 凍結產物與前輪對照。

```powershell
& $py src/select_robust_ensemble.py
if ($LASTEXITCODE -eq 0) {
    & $py src/evaluate_robust_ensemble.py
}
```

完整選型、配對摘要及來源雜湊保留於 Git 排除的 `ML/artifacts/p28_robust_ensemble/`。

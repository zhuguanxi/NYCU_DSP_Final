# P29 混合模型的獨立 eICU 確認規格

以 [P28](P28_robust_ensemble_results.md) 在 A/B 凍結的模型作候選，原凍結 Elastic 作主要對照。Demo 上兩者的差值只是選擇驗證對照的依據。確認流程固定於看到完整版結果前，使用與 [P27](P27_eicu_confirmation_protocol.md) 相同的隊列、時間窗、原始特徵與尿量對齊。

- 完整版用固定雜湊順序最多選 6,000 名合格患者，先核對 Demo 的 stay 與患者鍵，再排除所有 Demo 患者。資料來源若無法逐一核對，流程停止。抽樣上限、模型及映射先於計分固定。
- 主要對比為整點的 P28 AP 減凍結 Elastic AP，使用同一批患者的 1,000 次配對 bootstrap。若 95% 區間下界大於零，該隊列支持更高 AP。若區間跨零，改善仍不確定。P26、P22、GCS-only Elastic 的比較與原始間隔分析也保留。
- 所有模型同 cohort、相同尿量 mL 對齊，維持原參數與混合權重。衡量 AP、AUROC、Brier 及 0.5 門檻敏感度。Demo 模式只驗證新入口，完整結果另存。

在 `ML` 使用 [P21 CPU 環境](P21_predictive_boosting_protocol.md) 及既有 P26、P28 產物。

```powershell
& $py src/evaluate_eicu_confirmation.py --ensemble --smoke-demo
& $py src/evaluate_eicu_confirmation.py --ensemble --full-directory 'C:\path\to\eicu-crd-2.0'
```

正式輸出只保存彙總、來源雜湊及隊列計數，存於 Git 排除的 `ML/artifacts/p29_independent_eicu/`。試跑輸出在 `ML/artifacts/p29_demo_smoke/`。

# P46 完整 eICU 的 P45 固定模型確認

依 [P45 分支消融](P45_augmentation_branch_ablation_results.md) 凍結只增強有 GCS 分支的配置。P46 的主要比較沿用 P29、P40 的 P28 對原凍結 Elastic。P38 作既有次要比較，本輪另加入 P45 對 P28、P38 與凍結 Elastic 的次要配對比較。

- P45 只換上 P44 以全 A/B 與 782 次合格 Demo stay 訓練的時間窗 XGBoost。原無 GCS 的 LightGBM、GCS-only Elastic 和 P28 的對數勝算混合權重沿用。核對最終模型與資料參照雜湊。每次推論都逐筆核對無 GCS 之分數等於 P28。
- 完整資料沿用 P40 的前 48 小時、住院死亡、尿量 mL 對齊、患者固定抽樣與最高 6,000 名合格患者。先與 P44 訓練用的原 Demo 患者表核對識別鍵，排除發布的所有 Demo 患者，再用同一批 stay 比較所有模型。主要讀取整點 AP，另報原始間隔、AUROC、Brier 與 0.5 門檻敏感度。1,000 次患者層級配對 bootstrap 的 AP／AUROC 區間與原規格一致。全程 CPU，擬合 0 次。
- Demo 模式重現 P40 的已知指標，只記錄 P45 載入、推論、筆數與無 GCS 時的分數一致性檢查。Demo 有 P45 訓練病例，所以 P45 的效能統計限於獨立模式發布。

在 `ML` 使用 [P21 CPU 環境](P21_predictive_boosting_protocol.md) 與 [P38 凍結模型](P38_sequence_channel_ablation_results.md)。

```powershell
& $py src/evaluate_eicu_adaptation_confirmation.py --smoke-demo
& $py src/evaluate_eicu_adaptation_confirmation.py --full-directory 'C:\path\to\eicu-crd-2.0'
```

獨立結果只有彙總，位於 Git 排除的 `artifacts/p46_independent_eicu/`。流程試跑位於 `artifacts/p46_demo_smoke/`。

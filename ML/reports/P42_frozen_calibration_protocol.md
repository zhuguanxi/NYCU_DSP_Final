# P42 凍結模型的機率校準與分類門檻

比較原凍結 Full、Elastic、P28、P33、P38。對 A/B 8,000 次 stay 的原折外分數，以住院死亡為結局擬合一個截距和正斜率的邏輯校準式。先將原機率限制在 `[10⁻⁷, 1−10⁻⁷]` 再轉為對數勝算。Full 與 Elastic 逐折核對原凍結預測，P28 使用原 A/B 折外分數。P33 與 P38 用原 CPU 種子各重現三折。Set C 和 eICU 使用各自原凍結的全 A/B 模型。

- 每個模型的分類門檻在 A/B 折外分數上最大化敏感度與 PPV 的較小值。同分取較高門檻。本輪在載入 Set C 和 eICU 前固定係數和門檻。原分數門檻經校準式轉換，逐筆驗證兩個版本做相同分類。
- 在 Set C 未遮蔽、GCS 全遮和六項全遮以及 eICU Demo 的兩種取樣與對齊比較。固定記錄原／校準 Brier、AP、AUROC、0.5 門檻與 A/B 所選門檻的敏感度與 PPV。Brier 差使用同一次 stay／患者配對，分別做 2,000／1,000 次重抽。單調校準前後核對 AP 和 AUROC。
- Set C 另依 PhysioNet `score.c` v1.4 比較 Event 2 指標。4000 次 stay 依機率再按編號排序，切成十組，目標為降低以最高與最低十分位平均風險差正規化的 H。分母採官方程式的 `400 × (p × (1−p) + 0.001)`。Event 1 型的門檻指標用同一公式計算，eICU 作門檻移轉敏感度比較。 [PhysioNet 評分程式](https://physionet.org/content/challenge-2012/1.0.0/score.c)

以 [P21 CPU 環境](P21_predictive_boosting_protocol.md) 在 `ML` 執行 `& $py src/evaluate_frozen_calibration.py`。程式會使用既有 Torch CPU 環境重現折外分數。只持久保存係數、門檻、評估彙總與來源雜湊到 Git 排除的 `artifacts/p42_frozen_calibration/`。

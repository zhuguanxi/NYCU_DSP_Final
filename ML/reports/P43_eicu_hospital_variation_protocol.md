# P43 eICU Demo 的醫院層級敏感度

用原 P28、P38 和凍結 Elastic，比較三組同一批 stay 的差值。主要讀取尿量對齊的整點 AP，另報其餘取樣及對齊。所有模型與資料對齊沿用。P28 和 P38 的機率校準只套用 P42 已固定的係數，使用 Brier 差評估。全程 CPU，無重新訓練。

- 重現既有患者配對區間。另以醫院為抽樣區塊。若有跨院患者，將共享患者的醫院合成同一區塊，包含經第三家醫院間接相連的情況。抽同樣數目的區塊各 1,000 次，兩模型共用同一抽樣並回報 AP、AUROC 及 Brier 差的配對 95% 區間。
- 依序省略每個醫院區塊，重算剩餘 stay 的合併 AP 差，只彙總最小、最大及仍為正的次數。另把同一醫院內所有死亡與存活 stay 的可比較配對合併，依配對數加權算院內 AUROC。單一結局的醫院無可比較配對。院內 AUROC 的區間沿用醫院區塊抽樣。
- Demo 已重複評估，區間用於敏感度判讀。P40 仍是預先指定的獨立患者驗證。只保存彙總及雜湊。

在 `ML` 使用 [P21 CPU 環境](P21_predictive_boosting_protocol.md) 與 P42 既有產物，執行 `& $py src/audit_eicu_hospital_variation.py`。輸出位於 Git 排除目錄 `artifacts/p43_eicu_hospital_variation/`。

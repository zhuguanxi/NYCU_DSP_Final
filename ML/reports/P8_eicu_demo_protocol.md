# P8 eICU Demo 外部驗證規格

以 P12 A/B 訓練並保存的 Full、Elastic XGBoost 管線，在 eICU-CRD Demo 2.0.1 的既有 12 張資料表上做凍結外部驗證。保留原模型、補值、特徵選取和 0.5 閾值。全程使用 CPU。eICU 的結果只用於評估。以下規格在預測前固定。

## 隊列及結果欄位

| 項目 | 規則 |
| --- | --- |
| 評估單位 | `patientunitstayid`，須唯一 |
| 年齡與入院次序 | 年齡至少 18 歲，`unitvisitnumber=1`，選住院內第一個 ICU stay。遮蔽年齡 `> 89` 以 90 歲作為下界近似 |
| 時間 | `unitdischargeoffset >= 2880` 且 `hospitaldischargeoffset > 2880`。量測 offset 使用入 ICU 後 0 至 2,880 分鐘，含端點 |
| 結果 | `hospitaldischargestatus=Expired` 為 1，`Alive` 為 0。排除其餘狀態。已知住院結束時間須晚於預測時間 |
| 洩漏防範 | 有登錄或修訂 offset 的表排除在窗口結束後登錄的紀錄。ID、出院狀態、出院時間、住院天數與 APACHE 相關欄位不進模型 |
| 重複病人 | 不同住院可來自同一個人，使用 `uniquepid` 做病人叢集 bootstrap。隊列須有病人索引 |

P12 挑戰排除 ICU 停留不足 48 小時的個案。現有 eICU 候選隊列另外有 1 筆在 48 小時結束住院，本規格予以排除。eICU 的第一個 ICU stay 定義在單次住院內，P12 提供的是可得的首個 ICU stay。來源為 [P12 挑戰說明](https://moody-challenge.physionet.org/2012/) 與 [eICU patient 字典](https://eicu.mit.edu/eicutables/patient/)。

## 輸入映射

| P12 變數 | eICU 來源與轉換 |
| --- | --- |
| Age、Gender、Height、入院 Weight | `patient` 的 `age`、`gender`、`admissionheight`、`admissionweight`。女性 0、男性 1，未定義性別保留缺失。身高用 cm、體重用 kg。排除零與負的身高、體重 |
| ICUType | `MICU → 3`、`SICU → 4`、`CSICU → 2`。其餘類別留缺失。凍結的 one-hot 編碼可能把未知類別表示成全零，與參考類別重疊 |
| Albumin、ALP、ALT、AST、Bilirubin、BUN、Cholesterol、Creatinine、Glucose、HCT、K、Lactate、Na、Platelets、WBC | `lab` 的 serum／chemistry／hematology 對應標籤，依名稱、lab type 與系統單位核對。K、Na 的 mmol/L 與 mEq/L 數值相同，血球計數 K/µL 與 cells/nL 數值相同 |
| HCO3 | `lab` 的 chemistry `bicarbonate`，mmol/L |
| PaCO2、PaO2、pH、SaO2 | `lab` 的 ABG 類別 7，使用 `paCO2`、`paO2`、`pH`、`O2 Sat (%)`。檢體以類別近似。監視器 `sao2` 依 eICU 字典屬 SpO2 |
| TroponinI、TroponinT | `lab` 的 `troponin - I`、`troponin - T`，ng/mL 與 µg/L 數值相同。方法與試劑可比性仍有限 |
| FiO2 | `lab` 類別 7 的百分比除以 100。另接受呼吸表明確標為吸入氧濃度的紀錄，21–100 轉比例，0.21–1 保留。排除氧流量及含混單位文字。同分鐘優先用檢驗值 |
| HR、RespRate、SysABP、DiasABP、MAP | `vitalPeriodic` 的心率、呼吸率與 systemic arterial BP。主分析以每小時中位數重抽樣，時間取該小時最後一筆有效紀錄。敏感性分析保留原生監視器頻率 |
| NISysABP、NIDiasABP、NIMAP | `vitalAperiodic` 的 `noninvasive*` 欄位，保留原生頻率 |
| GCS | 護理表兩種明確總分標籤，使用可解析 3–15 的整數。同分鐘衝突取較低總分 |
| Temp | 護理表攝氏體溫優先。該 stay 沒有有效攝氏護理紀錄時，採監視器攝氏體溫，主分析依小時重抽樣 |
| 動態 Weight | `intakeOutput` 的 `Bodyweight (kg)`。入院體重也依 P12 抽取方式作為 0 分鐘觀測 |
| Mg、MechVent、Urine | 保留無觀測。Mg 的 P12 文件單位與實值仍有疑義。通氣狀態、尿量事件及累積方式需額外核對 |

同來源、同一變數與分鐘的重複數值取中位數，GCS 採前表規則。使用 P12 的 `prepare.extract_record` 計算 435 個欄位，包含缺失、次數和時間統計；凍結管線依自身有序欄位取用子集。資料來源及時間定義見 [lab](https://eicu.mit.edu/eicutables/lab/)、[nurseCharting](https://eicu.mit.edu/eicutables/nursecharting/)、[respiratoryCharting](https://eicu.mit.edu/eicutables/respiratorycharting/)、[intakeOutput](https://eicu.mit.edu/eicutables/intakeoutput/)、[vitalPeriodic](https://eicu.mit.edu/eicutables/vitalperiodic/) 與 [vitalAperiodic](https://eicu.mit.edu/eicutables/vitalaperiodic/)。

## 回報項目

- 兩個凍結模型皆執行主分析與預先指定的監視器頻率敏感性分析。以 AP、AUROC、Brier 分數、閾值 0.5 下的混淆矩陣與死亡比例為主。報告 AP、AUROC 的 95% 病人叢集 bootstrap 百分位區間（1,000 次、固定種子）。
- 比較特徵覆蓋率、觀測次數與 P12 A/B 成人資料，分開呈現結構性缺失和量測頻率差異。以既有 P12 C 點估計提供歷史參考。
- 輸入、逐筆特徵與分數留在本機 Git 排除目錄。版本庫保留規格、程式與彙總結果。

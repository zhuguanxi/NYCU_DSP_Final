# P60／P61：五種子與缺失曲線的重跑方式

2026-11-22。參數、原A/B選欄、增強強度、P26/P28路由及P33權重／8輪均固定，不用Set C／Demo挑種子或改配置。以下從ML目錄執行，先按[既有重跑入口](研究重跑與驗收.md)備妥P0–P59的資料、環境及凍結產物。

## 既有結果的驗收

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONPATH=Join-Path (Get-Location) 'artifacts/p21_cpu_packages/site'
& .venv/Scripts/python.exe related_work/verify_p60_p61.py
```

這不重新訓練。檢查35次元件擬合、五套來源與權重、39個固定小時遮蔽、未受影響欄位不變、表格count與序列mask一致、逐種子AP重算、全部公開列數及資料排除。驗收不要求實驗一定正向。

## 從前置產物完整重播

既有凍結訓練與輸入完成檔會阻止覆寫。要重播須在獨立專案副本／新產物目錄保留舊成果，不直接刪除原資料或覆蓋凍結模型。各階段來源、參數與結果皆可追溯，程式中的預設產物根目錄分別為p60_five_seed_stability及p61_progressive_block_missingness。

1. 主環境搭配P21套件，執行`related_work/p60_seed_stability.py`，35次固定元件擬合，先產生frozen.json。所有元件只用A+B；GCS增強位置固定，模型種子42–46。
2. 同環境執行`related_work/p61_missingness_inputs.py`，生成label-blind的39個案例並核對乾淨重建。這一步可與訓練獨立進行。
3. 確認兩項完成後，同環境執行`related_work/p60_evaluate_seeds.py`，完整評估本組七個模型×五種子×39案例及兩種乾淨Demo取樣。
4. 清除主環境的PYTHONPATH，以P54 GPU Python執行`related_work/p61_kedgn_stress.py`；重用既有A+B KEDGN五種子，只推論。部分案例完成後可恢復，來源雜湊不符會拒絕。0／100%端點沿P55保存分數，先以原權重核對一小批乾淨推論。
5. 主環境執行`related_work/publish_p60_p61.py`，產生全部逐種子整體指標、曲線、預先指定50%比較與bootstrap，再執行上述驗收。

KEDGN執行範例：

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& artifacts/related_work/kedgn_gpu_runtime/.venv/Scripts/python.exe related_work/p61_kedgn_stress.py
```

彙總範例：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONPATH=Join-Path (Get-Location) 'artifacts/p21_cpu_packages/site'
& .venv/Scripts/python.exe related_work/publish_p60_p61.py
& .venv/Scripts/python.exe related_work/verify_p60_p61.py
& .venv/Scripts/python.exe related_work/integrate_p60_p61.py
```

五種子CNN實際均在CPU完成，CNN推論也固定CPU。KEDGN可用閒置GPU，每批檢查（查詢最多快取3秒）；有其他運算程序或無自身運算時的顯著負載則改CPU。閒置桌面顯示程式不視為另一個模型訓練工作；未知GPU程序保守視為忙碌。絕不停止他人的程序。

KEDGN前段案例使用256推論批次；之後以`related_work/validate_kedgn_inference_batch.py`在A/B固定1024筆核對256／512／1024批次，最大預測差約1.49×10⁻⁷，依等價預測下的執行速度改用1024。這僅為推論記憶體／效率設定，沒有更改訓練batch、權重或依測試AP選參。驗證JSON與各案例實際批次都保存於本機及彙總。

## 結果與判讀

- [P60結果](P60_five_seed_stability_results.md)：200列種子×隊列整體指標，逐種子數值、平均／SD／範圍與各種子差值。
- [P61結果](P61_progressive_block_results.md)：1,560列種子×遮蔽整體指標，分開訓練種子及位置波動，五種子集成的固定模型配對區間另列。
- [驗收](P60_P61_verification.json)：來源、表示、完整性與公開檔案雜湊。

五種子機率平均後的AP與五次AP平均不相同，不能擇高者充當同一主結果。15組種子×位置不是15個新測試集。Set C／Demo先前已反覆查看，人工缺失與獨立跨院驗證仍有差距。原始資料、遮蔽位置、逐筆預測、模型及日誌全部留Git排除的artifacts，本輪公開CSV每列都是整體指標。

# 完整研究報告的重建

這裡只編排已公開的整體結果，不會訓練模型、重抽統計或接觸原始患者資料。完整報告為 [Markdown](../reports/期末研究報告.md) 與 [PDF](../reports/期末研究報告.pdf)。原 P51 報告保存為 [歷史版](../reports/期末研究報告_P51歷史版.md)。

從 `ML` 執行：

```powershell
# 圖：使用既有專案環境的 Matplotlib 3.11.2
& .venv/Scripts/python.exe reporting/plot_final_report.py

# 文字、表格、PDF：使用含 reportlab / pypdf / pillow 的 Python
python reporting/build_final_report.py compose
python reporting/build_final_report.py pdf
python reporting/build_final_report.py check
```

本機已使用 Codex 隨附的 ReportLab 4.4.9、pypdf 6.21.4、Pillow 13.0.0 完成 PDF。一般 Python 可依 [報告套件](../requirements-report.txt) 安裝；圖表使用主研究環境已有的 Matplotlib，不必把它混入舊模型環境。

PDF 預設嵌入 Windows 的 Microsoft JhengHei 字型 `msjh.ttc`／`msjhbd.ttc`。其他系統可用 `--font` 與 `--bold-font` 指定合法可用、涵蓋繁體中文的 TrueType 字型。字型不提交 Git。若手動修改最終 Markdown，只執行 `pdf` 即可重新排版；執行 `compose` 會依內容範本與公開彙總重建 Markdown，因此應把持續保留的文字修改寫回 `final_report_content.md`。

`compose` 首次會保留原 P51 歷史版本，後續不覆寫它。`check` 核對來源雜湊、PDF／Markdown 身分、重要數值與中文文字、相對連結、公開結果列數，以及資料目錄未被 Git 追蹤。視覺檢查另以 Poppler 將 PDF 轉 PNG，再檢查所有頁的字型、表格、換頁與圖表。來源與輸出紀錄在 [來源清單](../reports/final_report_sources.json)、[排版紀錄](../reports/final_report_build.json)。

此流程不改 P60/P61 的模型、種子、測試配置或既有 JSON。報告中的五次指標平均、五種子機率集成與原單一凍結模型必須維持分開，並保留部分缺失退化、外部 AP 與模型取捨的限制。

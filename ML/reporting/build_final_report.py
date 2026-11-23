"""Compose/render the final report from public aggregate evidence; never fit a model."""
import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

ML=Path(__file__).resolve().parents[1]
ROOT=ML.parent
REPORT=ML/'reports'
MD=REPORT/'期末研究報告.md'
PDF=REPORT/'期末研究報告.pdf'
DATE='2026-11-23'
BASELINE='863dff107554fb1cc6a99c533be301283cd427e8'
NAMES={'frozen_full':'Full','frozen_elastic':'Elastic','gcs_only_full':'GCS-only Full',
 'gcs_only_elastic':'GCS-only Elastic','p26':'P26','p28':'P28','p33':'P33','kedgn_ab':'KEDGN A+B'}

def read(name): return json.loads((REPORT/name).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def num(x): return f'{x:.5f}'
def ci(x): return f'[{x[0]:+.5f}, {x[1]:+.5f}]'
def delta(x,bounds): return f'{x:+.5f}<br/>{ci(bounds)}'
def ms(row): return f'{row["mean"]:.5f} ± {row["sd"]:.5f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])

def compose():
    p60=read('P60_five_seed_stability_summary.json');p61=read('P61_progressive_block_summary.json')
    p54=read('P54_kedgn_ab_transfer_summary.json');p58=read('P58_joint_objectives_summary.json')
    p57=read('P57_demo_paired_summary.json');p59=read('P59_ma_alignment_summary.json')
    audit=read('P60_P61_verification.json')
    for name,digest in audit['published_summary_sha256'].items(): assert sha(REPORT/name)==digest
    assert audit['complete'] and len(p60['per_seed'])==200 and p61['individual_metric_rows']==1560
    archive=REPORT/'期末研究報告_P51歷史版.md'
    if not archive.exists():
        assert '2026 年 11 月 17 日' in MD.read_text(encoding='utf-8'),'Refusing to mislabel a newer report as P51'
        shutil.copyfile(MD,archive)
    inserts={}
    inserts['COHORTS']=table(['資料與觀察窗','住院次','死亡','用途'],[
      ['P12 A/B 48h','8,000','1,122','原三折開發與全 A/B 擬合'],['P12 Set C 48h','4,000','585','凍結比較與缺失測試'],
      ['P12 A/B 24h 代理','7,817','1,071','24h 模型開發'],['P12 Set C 24h 代理','3,906','549','24h 凍結評估'],
      ['eICU Demo 48h','782','91','純轉移與分組折外增強'],['eICU Demo 24h','1,398','121','24h 轉移與增強'],
      ['Demo 24h APACHE 共同隊列','1,166','100','同窗臨床評分比較']])
    inserts['ALIGNMENT']=table(['欄位','主分析處理與限制'],[
      ['入院描述／ICU 類型','年齡、性別、身高 cm、體重 kg；ICUType 僅映射明確相容類別，其餘缺失，未知 one-hot 仍可能與參考類別重疊。'],
      ['GCS','護理總分依明確標籤與 3-15 值域解析；同分鐘衝突取較低總分。分項探索另列，未根據測試收益任意替換。'],
      ['乳酸與化學／血液檢驗','依名稱、類別與系統單位對齊；保留檢驗定義與可得時間限制。'],
      ['侵入式 MAP 與非侵入式 BP','分別來自 systemic arterial 與 noninvasive 來源，不把不同量測方式直接視為同一通道。'],
      ['FiO2／ABG','百分比與比例依明確標籤轉換；排除氧流量及含混單位，動脈檢驗和 SpO2 分開。'],
      ['尿量','使用明確的 mL 事件，不以模糊累積欄直接當同義量測；主分析已採 P23 對齊。'],
      ['鎂與機械通氣','原主線保留未觀測，避免不確定單位／狀態語意的硬映射；後續探索未冒充已解決。'],
      ['HR、RespRate、Temp 與 BP','保留整點／原始間隔兩套取樣，未依較好的測試結果選其中一種。']])
    inserts['ESTIMANDS']=table(['口徑','例子','如何解讀'],[
      ['原單一凍結模型','P33 clean AP 0.60447','原先保存的權重；配對病例區間有明確對應'],
      ['五次指標平均 ± SD',ms(p60['summary']['clean']['p33']['metrics']['ap']),'描述五個固定訓練種子的變異'],
      ['五種子機率集成',num(p60['summary']['clean']['p33']['ensemble']['ap']),'先平均逐病例機率再算 AP，不能與指標平均互換']])
    inserts['SEED_TABLE']=table(['模型','未遮蔽 AP','GCS 全遮 AP','六欄全遮 AP'],[
      [label,*[ms(p60['summary'][case][key]['metrics']['ap']) for case in ('clean','gcs_full','six_full')]] for key,label in NAMES.items()])
    inserts['SECONDARY_METRICS']=table(['模型','AUROC 平均','Brier 平均','敏感度平均 @0.5'],[
      [NAMES[k],*[num(p60['summary']['clean'][k]['metrics'][m]['mean']) for m in ('auroc','brier','sensitivity')]] for k in ('frozen_full','p28','p33','kedgn_ab')])
    inserts['ORIGINAL_MASK_TABLE']=table(['原單一模型情境','Full AP','P33 AP','AP 差與 95% 區間'],[
      [label,num(r['metrics']['frozen_full']['ap']),num(r['metrics']['p33']['ap']),delta(r['delta']['ap'],r['paired_ci']['ap'])]
      for case,label in [('clean','未遮蔽'),('full_GCS','GCS 全遮'),('full_all_six','六欄全遮')]
      for r in [p58['set_c']['scenarios'][case]]])
    inserts['KG_TRANSFER']=table(['凍結評估隊列','Set A KEDGN 集成 AP','A+B KEDGN 集成 AP'],[
      [label,num(p54['results'][case]['controls']['set_a_kedgn']['ap']),num(p54['results'][case]['ensemble']['ap'])]
      for case,label in [('set_c','Set C 48h'),('eicu_hourly','Demo 48h 整點'),('eicu_native','Demo 48h 原始間隔')]])
    rows=[]
    for target in ('gcs','six'):
        for design in ('random','block'):
            for length in (12,24,36):
                key=f'{target}_{design}_{length}';models=p61['curves'][key]
                cases=[r for r in p61['cases'] if r['target']==target and r['design']==design and r['hours']==length]
                fraction=sum(r['actual_removed_fraction'] for r in cases)/len(cases)
                rows.append([f'{"GCS" if target=="gcs" else "六欄"} {"隨機" if design=="random" else "連續"} {length}h',f'{fraction*100:.2f}%',
                    *[num(models[m]['mean']) for m in ('frozen_full','p28','p33','kedgn_ab')]])
    inserts['MASK_CURVE_TABLE']=table(['情境','刪除事件','Full','P28','P33','KEDGN'],rows)
    primary=p61['primary_50_percent_ensemble_contrasts']
    def case_label(key): return key.replace('gcs','GCS').replace('six','六欄').replace('_random_24',' 隨機24h').replace('_block_24',' 連續24h')
    inserts['PARTIAL_CI']=table(['50% 情境','AP 差','95% 區間','少退化 AP','95% 區間'],[
      [case_label(k),f'{r["ap_delta"]:+.5f}',ci(r['ap_ci']),f'{r["retention_gain"]:+.5f}',ci(r['retention_ci'])]
      for k,v in primary.items() for r in [v['frozen_full']]])
    inserts['PARTIAL_COMPARATORS']=table(['50% 情境','P33 減 P28 AP 差與區間','P33 減 KEDGN AP 差與區間'],[
      [case_label(k),delta(v['p28']['ap_delta'],v['p28']['ap_ci']),delta(v['kedgn_ab']['ap_delta'],v['kedgn_ab']['ap_ci'])] for k,v in primary.items()])
    inserts['BLOCK_RANDOM']=table(['目標','模型','連續減隨機 AP 差','95% 區間'],[
      ['GCS' if target=='gcs' else '六欄',NAMES[m],f'{r["block_minus_random_ap"]:+.5f}',ci(r['ci'])]
      for target,models in p61['block_vs_random_24h'].items() for m,r in models.items()])
    inserts['SEED_DEMO_TABLE']=table(['P12-only 模型','Demo 整點 AP','Demo 原始間隔 AP'],[
      [label,*[ms(p60['summary'][c][k]['metrics']['ap']) for c in ('eicu_hourly','eicu_native')]] for k,label in NAMES.items()])
    inserts['TRANSFER_TABLE']=table(['原凍結 P33 取樣','對照','AP 差與患者 95% 區間'],[
      ['整點' if mode=='hourly' else '原始間隔',NAMES[ref],delta(r['delta'][ref]['ap'],r['paired_ci'][ref]['ap'])]
      for mode,r in p58['eicu_demo']['modes'].items() for ref in ('frozen_full','p28')])
    variants={'original_1000':'原版 alpha1000','original_10':'原版 alpha10','augmented_10':'五欄增強 alpha10'}
    inserts['MA_TABLE']=table(['MA-GBT 整點配置','原 AP','對齊 AP','原補值依賴','對齊補值依賴'],[
      [variants[r['variant']]+' '+r['model'],num(r['before']['ap']),num(r['after']['ap']),f'{r["rho_before"]*100:.2f}%',f'{r["rho_after"]*100:.2f}%']
      for r in p59['eicu'] if r['mode']=='hourly'])
    inserts['H24_TABLE']=table(['取樣','24h 模型','AP','AUROC','Brier'],[
      ['整點' if mode=='hourly' else '原始間隔',m.upper(),*[num(r['metrics'][m][metric]) for metric in ('ap','auroc','brier')]]
      for mode,r in p57['results'].items() for m in ('p49','p50','p52')])
    for metric,token in [('ap','H24_AP_CI'),('brier','H24_BRIER_CI')]:
        inserts[token]=table(['取樣','P52 對照',metric.upper()+' 差','患者 95% 區間','院鍵區塊 95% 區間'],[
          ['整點' if mode=='hourly' else '原始間隔',ref.upper(),f'{r["metrics"]["p52"][metric]-r["metrics"][ref][metric]:+.5f}',
           ci(r['patient_ci'][f'p52_vs_{ref}'][metric]),ci(r['hospital_block_ci'][f'p52_vs_{ref}'][metric])]
          for mode,r in p57['results'].items() for ref in ('p49','p50')])
    inserts['APACHE_TABLE']=table(['24h 共同隊列取樣','模型','AP','AUROC','Brier'],[
      ['整點' if mode=='hourly' else '原始間隔',m.upper().replace('APACHE_IVA','APACHE IVa'),*[num(v[q]) for q in ('ap','auroc','brier')]]
      for mode,r in p57['results'].items() for m,v in r['matched']['metrics'].items()])
    inserts['STAGE_INDEX']=table(['階段','問題與主要內容','證據入口'],[
      ['P0-P7','原始清理、四類基準、調參、選欄、解釋與凍結 C','[原索引](../README.md)'],
      ['P8-P12','Demo 對齊、單欄／聯合壓力與 GCS／尿量增強','[P8](P8_eicu_demo_results.md)' if (REPORT/'P8_eicu_demo_results.md').exists() else '[P8](P8_eicu_demo_results.md)'],
      ['P13-P20','MA-GBT alpha、逐欄增強與補值路徑','[P20](P20_ma_gbt_alpha10_bilirubin_results.md)'],
      ['P21-P28','缺失 boosting、140 時間摘要、尿量映射、分流與混合','[P21-P25](P21-P25_predictive_performance_results.md) / [P28](P28_robust_ensemble_results.md)'],
      ['P29-P40','確認入口、序列／頻率／通道／種子消融','[P33](P33_hourly_sequence_results.md) / [P39](P39_sequence_seed_ensemble_results.md)'],
      ['P41-P48','結局時鐘、校準、院鍵、外部增強、重疊及臨床分數','[P41](P41_p12_outcome_timing_results.md) / [P45](P45_augmentation_branch_ablation_results.md)'],
      ['P49-P52','24h 風險集、增強、資料包與 CatBoost 混合','[P49](P49_24h_landmark_results.md) / [P52](P52_24h_catboost_blend_results.md)'],
      ['P53-P55','KEDGN Set A 轉移、A+B 重訓與共同缺失','[P54](P54_kedgn_ab_transfer_results.md) / [P55](P55_kedgn_missingness_results.md)'],
      ['P56-P59','條件混合負向消融、24h 配對、同模型退化與 MA 映射','[P57](P57_demo_paired_results.md) / [P59](P59_ma_alignment_results.md)'],
      ['P60','35 元件擬合，七管線五種子，全部保留','[P60](P60_five_seed_stability_results.md)'],
      ['P61','39 案例、時鐘／區段缺失、預定配對與變異分解','[P61](P61_progressive_block_results.md)']])
    inserts['RESULT_INDEX']=table(['內容','公開檔案','口徑或用途'],[
      ['P0-P59 整體表','[CSV](研究結果總表.csv)','167 列歷史整合，含不同窗與映射標記'],
      ['五種子各隊列','[200 列 CSV](P60_all_seed_results.csv)','五個固定種子，非患者紀錄'],
      ['五種子全部缺失','[1,560 列 CSV](P61_all_seed_mask_results.csv)','39 案例，種子／位置交叉'],
      ['原單一模型與退化','[P58 JSON](P58_joint_objectives_summary.json)','原固定分數的配對差與退化差'],
      ['24h 外部配對','[P57 JSON](P57_demo_paired_summary.json)','患者／院鍵區塊 AP／Brier／AUROC'],
      ['五種子詳細統計','[P60 JSON](P60_five_seed_stability_summary.json)','個別指標平均與集成分開'],
      ['缺失詳細統計','[P61 JSON](P61_progressive_block_summary.json)','曲線、主要區間、實際刪除比例與推論批次'],
      ['原整合驗收','[JSON](研究完成驗收.json)','P55-P59 等來源與重播核對'],
      ['最新補強驗收','[JSON](P60_P61_verification.json)','35 fits、39 masks、200/1560 列與無資料提交'],
      ['重跑入口','[說明](P60_P61_reproduction.md)','環境、恢復、驗收及資料邊界']])
    template=(ML/'reporting/final_report_content.md').read_text(encoding='utf-8')
    for key,value in inserts.items(): template=template.replace('{{'+key+'}}',value)
    assert not re.search(r'\{\{[^}]+\}\}',template),'Unfilled report placeholder'
    MD.write_text(template.rstrip()+'\n',encoding='utf-8',newline='\n')
    source_names=['P54_kedgn_ab_transfer_summary.json','P55_kedgn_missingness_summary.json','P57_demo_paired_summary.json',
      'P58_joint_objectives_summary.json','P59_ma_alignment_summary.json','P60_five_seed_stability_summary.json',
      'P61_progressive_block_summary.json','P60_P61_verification.json','資料來源核對.json','P8_eicu_demo_protocol.md',
      'P20_ma_gbt_alpha10_bilirubin_results.md','P41_p12_outcome_timing_results.md','P45_augmentation_branch_ablation_results.md',
      'P49_24h_landmark_protocol.md','P52_24h_catboost_blend_results.md','文獻定位與可比性.md']
    sources={str((REPORT/name).relative_to(ML)):sha(REPORT/name) for name in source_names}
    sources['reporting/final_report_content.md']=sha(ML/'reporting/final_report_content.md')
    save(REPORT/'final_report_sources.json',dict(report_date=DATE,experiment_result_commit=BASELINE,
      builder_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      new_model_fits=0,source_sha256=sources,report_markdown_sha256=sha(MD),
      statements=['Single frozen model, mean metrics and probability ensemble are separate',
       'Masked accuracy is distinct from retention','No patient records embedded; only public aggregate evidence']))
    print(f'Composed complete report: {len(template)} characters; {len(sources)} public evidence sources.')

def inline(text):
    text=text.replace('−','-').replace('–','-').replace('—','-')
    parts=re.split(r'(\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|`[^`]+`|<br\s*/?>)',text)
    rendered=[]
    for part in parts:
        link=re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',part)
        if link:
            label,target=link.groups()
            if not target.startswith(('http://','https://','#')):
                path=(REPORT/target.split('#')[0]).resolve().relative_to(ROOT)
                target='https://github.com/zhuguanxi/NYCU_DSP_Final/blob/main/'+path.as_posix()
            rendered.append(f'<link href="{html.escape(quote(target,safe=":/?&=#%"),quote=True)}" color="#1d5272">{html.escape(label)}</link>')
        elif part.startswith('**') and part.endswith('**'): rendered.append('<b>'+html.escape(part[2:-2])+'</b>')
        elif part.startswith('`') and part.endswith('`'): rendered.append(html.escape(part[1:-1]))
        elif re.fullmatch(r'<br\s*/?>',part): rendered.append('<br/>')
        else: rendered.append(html.escape(part))
    return ''.join(rendered)

def render(font,bold):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
    pdfmetrics.registerFont(TTFont('CJK',font,subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CJK-Bold',bold,subfontIndex=0))
    pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJK-Bold',italic='CJK',boldItalic='CJK-Bold')
    if Path('C:/Windows/Fonts/consola.ttf').exists(): pdfmetrics.registerFont(TTFont('Code','C:/Windows/Fonts/consola.ttf'))
    else: pdfmetrics.registerFontFamily('Code',normal='Courier')
    W,H=A4;margin=49;content=W-2*margin
    body=ParagraphStyle('Body',fontName='CJK',fontSize=10.5,leading=16.2,spaceAfter=7,wordWrap='CJK',splitLongWords=1)
    styles={'body':body,'title':ParagraphStyle('Title',parent=body,fontName='CJK-Bold',fontSize=23,leading=32,spaceAfter=16),
      'h2':ParagraphStyle('H2',parent=body,fontName='CJK-Bold',fontSize=16,leading=23,spaceBefore=12,spaceAfter=10,keepWithNext=True),
      'h3':ParagraphStyle('H3',parent=body,fontName='CJK-Bold',fontSize=12,leading=18,spaceBefore=8,spaceAfter=6,keepWithNext=True),
      'cell':ParagraphStyle('Cell',parent=body,fontSize=8.8,leading=13,spaceAfter=0),
      'head':ParagraphStyle('TableHead',parent=body,fontName='CJK-Bold',fontSize=8.8,leading=13,spaceAfter=0),
      'code':ParagraphStyle('Code',fontName='Code' if Path('C:/Windows/Fonts/consola.ttf').exists() else 'Courier',fontSize=8,leading=11,spaceAfter=3,splitLongWords=1)}
    elements=[];lines=MD.read_text(encoding='utf-8').splitlines();i=0
    def para(text,style='body'): return Paragraph(inline(text),styles[style])
    def widths(headers):
        n=len(headers)
        if n==2: fractions=[.23,.77]
        elif n==3: fractions=[.26,.37,.37]
        elif n==4:
            fractions=[.28,.24,.24,.24] if headers[0]=='模型' else [.28,.16,.16,.40]
            if headers[0] in ('資料與觀察窗','階段'): fractions=[.30,.13,.13,.44]
            if headers[-1]=='95% 區間': fractions=[.13,.24,.25,.38]
            if headers[0]=='原單一模型情境': fractions=[.22,.15,.15,.48]
        elif n==5:
            fractions=[.20,.12,.25,.18,.25] if headers[0]=='50% 情境' else [.20,.15,.15,.25,.25]
            if headers[0]=='MA-GBT 整點配置': fractions=[.35,.13,.13,.195,.195]
            if headers[2].endswith(' 差'): fractions=[.12,.14,.13,.305,.305]
        elif n==6: fractions=[.25,.13,.155,.155,.155,.155]
        else: fractions=[1/n]*n
        return [content*f for f in fractions]
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1;continue
        if line=='<!-- PAGEBREAK -->': elements.append(PageBreak());i+=1;continue
        if line.startswith('<!--'): i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'): code.append(lines[i]);i+=1
            for row in code: elements.append(Paragraph(html.escape(row).replace(' ','&nbsp;'),styles['code']))
            elements.append(Spacer(1,6));i+=1;continue
        image=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
        if image:
            caption,target=image.groups();path=(REPORT/target).resolve()
            from PIL import Image as PILImage
            with PILImage.open(path) as picture: rw,rh=picture.size
            width=content;height=width*rh/rw
            elements.append(KeepTogether([Image(str(path),width=width,height=height),Spacer(1,6)]))
            i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells): rows.append(cells)
                i+=1
            assert all(len(r)==len(rows[0]) for r in rows)
            grid=[[para(c,'head' if ri==0 else 'cell') for c in row] for ri,row in enumerate(rows)]
            t=Table(grid,colWidths=widths(rows[0]),repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f5')),
              ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#637986')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d9e0e4')),
              ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),
              ('BOTTOMPADDING',(0,0),(-1,-1),5),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
            elements.extend([t,Spacer(1,10)]);continue
        if line.startswith('# '): elements.append(para(line[2:],'title'));i+=1;continue
        if line.startswith('## '): elements.append(para(line[3:],'h2'));i+=1;continue
        if line.startswith('### '): elements.append(para(line[4:],'h3'));i+=1;continue
        text=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','<!--','![')):
            text.append(lines[i].strip());i+=1
        elements.append(para(' '.join(text)))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('CJK',8);canvas.setFillColor(colors.HexColor('#596b77'))
        canvas.drawString(margin,H-27,'ICU 住院死亡預測完整研究報告')
        canvas.drawRightString(W-margin,H-27,'P0-P61  |  '+DATE)
        canvas.drawCentredString(W/2,25,f'第 {doc.page} 頁');canvas.restoreState()
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=margin,leftMargin=margin,topMargin=48,bottomMargin=44,
       title='ICU 住院死亡預測完整研究報告',author='ICU mortality research project',pageCompression=1)
    doc.build(elements,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    pages=len(PdfReader(PDF).pages)
    save(REPORT/'final_report_build.json',dict(report_date=DATE,page_count=pages,markdown_sha256=sha(MD),pdf_sha256=sha(PDF),
      packages={k:importlib.metadata.version(k) for k in ('reportlab','pypdf','pillow')},
      font_files=[Path(font).name,Path(bold).name],new_model_fits=0))
    print(f'Rendered {pages}-page PDF ({PDF.stat().st_size:,} bytes).')

def check():
    from pypdf import PdfReader
    manifest=read('final_report_sources.json');build=read('final_report_build.json')
    for path,digest in manifest['source_sha256'].items(): assert sha(ML/path)==digest,(path,'changed')
    assert sha(MD)==manifest['report_markdown_sha256']==build['markdown_sha256']
    assert sha(PDF)==build['pdf_sha256']
    reader=PdfReader(PDF);text='\n'.join(p.extract_text() or '' for p in reader.pages)
    compact=re.sub(r'\s+','',text)
    for token in ('0.60084','0.53111','0.48999','0.60290','0.57502','0.00884','限制','住院死亡','P61'):
        assert token in compact,('Missing PDF text',token)
    assert '\ufffd' not in text and len(reader.pages)==build['page_count']
    for path in [MD,ML/'reporting/README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if not target.startswith(('http://','https://','#')): assert (path.parent/target.split('#')[0]).exists(),(path,target)
    for name,n in [('P60_all_seed_results.csv',200),('P61_all_seed_mask_results.csv',1560)]:
        with (REPORT/name).open(encoding='utf-8-sig',newline='') as stream: rows=list(csv.DictReader(stream))
        assert len(rows)==n
    assert not subprocess.check_output(['git','ls-files','ML/artifacts/','ML/dataset/','data/'],cwd=ROOT,text=True).strip()
    print(f'Report integrity checked: {len(reader.pages)} pages, Chinese text, key values, links, source hashes and no tracked patient artifacts.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['compose','pdf','check'])
    parser.add_argument('--font',default='C:/Windows/Fonts/msjh.ttc');parser.add_argument('--bold-font',default='C:/Windows/Fonts/msjhbd.ttc')
    args=parser.parse_args()
    if args.mode=='compose': compose()
    elif args.mode=='pdf': render(args.font,args.bold_font)
    else: check()

"""Rebuild A/B only; this entry point audits only C archive metadata."""
from pathlib import Path
import csv, hashlib, json, collections, time, tarfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'ML/artifacts'
REPORT = ROOT / 'ML/reports'
DATA = ROOT / 'data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0'
SOURCE = 'https://physionet.org/content/challenge-2012/1.0.0/'
UNITS = {
 'Albumin':'g/dL','ALP':'IU/L','ALT':'IU/L','AST':'IU/L','Bilirubin':'mg/dL',
 'BUN':'mg/dL','Cholesterol':'mg/dL','Creatinine':'mg/dL','DiasABP':'mmHg',
 'FiO2':'fraction','GCS':'score','Glucose':'mg/dL','HCO3':'mmol/L','HCT':'%',
 'HR':'bpm','K':'mEq/L','Lactate':'mmol/L','Mg':'mmol/L','MAP':'mmHg',
 'MechVent':'binary','Na':'mEq/L','NIDiasABP':'mmHg','NIMAP':'mmHg','NISysABP':'mmHg',
 'PaCO2':'mmHg','PaO2':'mmHg','pH':'pH','Platelets':'cells/nL','RespRate':'bpm',
 'SaO2':'%','SysABP':'mmHg','Temp':'degC','TroponinI':'ug/L','TroponinT':'ug/L',
 'Urine':'mL per recorded observation','WBC':'cells/nL','Weight':'kg'}
STATIC = {'Age':'years','Gender':'category','Height':'cm','ICUType':'category','Weight':'kg'}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def clean(var,v):
 if not np.isfinite(v): return np.nan,'nonfinite'
 if v<0: return np.nan,'negative_sentinel'
 codes={'Gender':{0.,1.},'ICUType':{1.,2.,3.,4.},'MechVent':{0.,1.}}
 if var in codes and v not in codes[var]: return np.nan,'invalid_code'
 bounds={'FiO2':(0,1),'GCS':(3,15),'pH':(0,14),'SaO2':(0,100),'HCT':(0,100)}
 if var in bounds and not bounds[var][0]<=v<=bounds[var][1]: return np.nan,'outside_documented_domain'
 return v,'valid'

def stats_spec(var):
 if var=='MechVent': return {'ever':0,'last':0,'first':1,'observed_fraction':1}
 if var=='GCS': return {'min':0,'max':0,'last':0,'first':1,'delta':1}
 if var=='Urine': return {'min':0,'max':0,'mean':0,'last':0,'first':1,'std':1}
 return {'min':0,'max':0,'mean':0,'last':0,'first':1,'std':1,'delta':1,'slope':1}

def extract_record(content, rid):
 """Fixed encounter-local extraction, shared by development and C evaluation."""
 obs={v:[] for v in UNITS};adm={v:np.nan for v in STATIC};records=[]
 for seq,r in enumerate(csv.DictReader(content.decode().splitlines())):
  var=r.get('Parameter','').strip()
  if var=='RecordID':
   assert int(float(r['Value']))==rid
   continue
  try:
   hh,mm=map(int,r['Time'].split(':'));minute=hh*60+mm
   assert 0<=mm<60 and hh>=0
  except Exception:minute=-1
  try:raw=float(r['Value'])
  except Exception:raw=np.nan
  v,flag=clean(var,raw)
  if var not in UNITS and var not in STATIC:flag='blank_parameter' if not var else 'unknown_parameter';v=np.nan
  if minute<0 or minute>2880:flag='outside_window';v=np.nan
  records.append((rid,minute,var,raw,v,flag,seq))
  if var in STATIC and minute==0 and np.isnan(adm[var]):adm[var]=v
  if var in UNITS and flag=='valid':obs[var].append((minute/60,seq,v))
 feat={f'static_{v}':x for v,x in adm.items()};valid_dynamic=0
 for var,vals in obs.items():
  vals.sort();n=len(vals);valid_dynamic+=n
  t=np.array([v[0] for v in vals]);x=np.array([v[2] for v in vals]);spec=stats_spec(var)
  ss={s:np.nan for s in spec}
  if n:
   for s in spec:
    if s=='min':ss[s]=float(x.min())
    elif s in ('max','ever'):ss[s]=float(x.max())
    elif s in ('mean','observed_fraction'):ss[s]=float(x.mean())
    elif s=='last':ss[s]=float(x[-1])
    elif s=='first':ss[s]=float(x[0])
    elif s=='delta':ss[s]=float(x[-1]-x[0])
    elif s=='std' and n>=2:ss[s]=float(x.std(ddof=1))
    elif s=='slope' and n>=2 and np.ptp(t)>0:
     tc=t-t.mean();ss[s]=float(np.dot(tc,x-x.mean())/np.dot(tc,tc))
  for s,v in ss.items():feat[f'{var}__{s}']=v
  feat[f'{var}__count']=n;feat[f'{var}__missing']=int(n==0)
  for s in ('std','slope'):
   if s in spec:feat[f'{var}__{s}_defined']=int(np.isfinite(ss[s]))
 flags=dict(Age_under18=bool(adm['Age']<18),no_dynamic_observations=valid_dynamic==0,
            admission_review=bool(adm['Weight']<30 or adm['Weight']>250 or adm['Height']<130))
 return feat,records,flags

def main():
 OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 start=time.perf_counter(); rows=[]; labels=[]; dictionary=[]; quality=[]; alignment=[]
 raw_batch=[]; writer=None; counts=collections.Counter(); ranges={}; static_flags=[]
 for var,unit in STATIC.items():
  dictionary.append(dict(feature='static_'+var,variable=var,group='static_'+var,family=0,statistic='admission',unit=unit,kind='categorical' if var in ('Gender','ICUType') else 'numeric',meaning='first descriptor at elapsed 00:00'))
 for var,unit in UNITS.items():
  spec=stats_spec(var)
  for stat,fam in spec.items():
   dictionary.append(dict(feature=f'{var}__{stat}',variable=var,group=var,family=fam,statistic=stat,unit=unit+'/hour' if stat=='slope' else unit,kind='numeric',meaning='observed fraction, not ventilated time' if stat=='observed_fraction' else 'valid observations within inclusive 0-48h'))
  for stat in ['count','missing']+(['std_defined'] if 'std' in spec else [])+(['slope_defined'] if 'slope' in spec else []):
   dictionary.append(dict(feature=f'{var}__{stat}',variable=var,group=var,family=2,statistic=stat,unit='count' if stat=='count' else 'binary',kind='numeric',meaning='availability indicator; undefined summaries distinguished from no observation'))
 for setname in ['a','b']:
  outcomes=pd.read_csv(DATA/f'Outcomes-{setname}.txt').set_index('RecordID')
  seen=set()
  for ix,p in enumerate(sorted((DATA/f'set-{setname}').glob('*.txt'))):
   content=p.read_bytes();rid=int(p.stem);seen.add(rid)
   assert rid in outcomes.index
   alignment.append(dict(RecordID=rid,source_set=setname.upper(),sha256=hashlib.sha256(content).hexdigest()))
   feat,record_observations,flags=extract_record(content,rid)
   for _,minute,var,raw,v,flag,seq in record_observations:
    counts[(var,flag)]+=1
    if flag=='valid':
     rec=ranges.setdefault(var,[np.inf,-np.inf,0,0]);rec[0]=min(rec[0],v);rec[1]=max(rec[1],v);rec[2]+=1;rec[3]+=int(v==0)
   raw_batch.extend(record_observations)
   rows.append({'RecordID':rid,**feat})
   labels.append(dict(RecordID=rid,source_set=setname.upper(),In_hospital_death=int(outcomes.loc[rid,'In-hospital_death'])))
   static_flags.append(dict(RecordID=rid,source_set=setname.upper(),**flags))
   if (ix+1)%200==0 or ix==3999:
    frame=pd.DataFrame(raw_batch,columns=['RecordID','elapsed_minute','variable','raw_value','clean_value','quality_flag','source_order'])
    tab=pa.Table.from_pandas(frame,preserve_index=False)
    if writer is None: writer=pq.ParquetWriter(OUT/'observations_AB.parquet',tab.schema,compression='zstd')
    writer.write_table(tab);raw_batch=[]
   if (ix+1)%1000==0:print(f'prepared {setname.upper()} {ix+1}',flush=True)
  assert seen==set(outcomes.index)
 writer.close()
 X=pd.DataFrame(rows).set_index('RecordID');y=pd.DataFrame(labels).set_index('RecordID');fd=pd.DataFrame(dictionary).set_index('feature')
 assert X.index.is_unique and X.index.equals(y.index) and set(X.columns)==set(fd.index)
 assert not any(s in X.columns for s in ['RecordID','source_set','In-hospital_death','Length_of_stay','Survival','SAPS-I','SOFA'])
 X.to_parquet(OUT/'features_AB.parquet');y.to_csv(OUT/'labels_AB.csv');fd.to_csv(OUT/'feature_dictionary.csv',encoding='utf-8-sig')
 pd.DataFrame(alignment).to_csv(OUT/'record_alignment.csv',index=False)
 pd.DataFrame(static_flags).to_csv(OUT/'cohort_quality_flags.csv',index=False)
 pd.DataFrame([dict(variable=v,flag=f,count=n) for (v,f),n in counts.items()]).to_csv(OUT/'measurement_quality.csv',index=False)
 pd.DataFrame([dict(variable=v,min_nonnegative=r[0],max_nonnegative=r[1],valid_n=r[2],zeros=r[3]) for v,r in ranges.items()]).to_csv(OUT/'observed_ranges.csv',index=False)
 rules=[]
 for v in sorted(set(UNITS)|set(STATIC)):
  bounds={'FiO2':(0,1),'GCS':(3,15),'pH':(0,14),'SaO2':(0,100),'HCT':(0,100)}
  codes={'Gender':'0,1','ICUType':'1,2,3,4','MechVent':'0,1'}
  rules.append(dict(variable=v,unit=UNITS.get(v,STATIC.get(v)),lower=bounds.get(v,(0,None))[0],upper=bounds.get(v,(0,None))[1],valid_codes=codes.get(v,''),official_alias={'TroponinI':'TropI','TroponinT':'TropT'}.get(v,v),rule='negative/nonfinite -> missing; categorical/documented domain only; otherwise retain nonnegative values',source=SOURCE))
 pd.DataFrame(rules).to_csv(OUT/'cleaning_rules.csv',index=False)
 miss=X.isna().mean().sort_values(ascending=False).rename('missing_rate');miss.to_csv(OUT/'feature_missingness.csv')
 with tarfile.open(DATA/'set-c.tar.gz','r:gz') as t: c_count=sum(m.isfile() and m.name.endswith('.txt') for m in t.getmembers())
 manifest=dict(date='2026-10-13',source=SOURCE,development_n=len(X),deaths=int(y.In_hospital_death.sum()),mortality_rate=float(y.In_hospital_death.mean()),feature_n=X.shape[1],family_counts={str(i):int((fd.family<=i).sum()) for i in range(3)},sequence_rows=int(pq.ParquetFile(OUT/'observations_AB.parquet').metadata.num_rows),C_archived_n=c_count,C_models_evaluated=False,C_sha256=sha(DATA/'set-c.tar.gz'),input_label_sha256={s:sha(DATA/f'Outcomes-{s}.txt') for s in ['a','b']},age_under18=int(pd.DataFrame(static_flags).Age_under18.sum()),no_dynamic=int(pd.DataFrame(static_flags).no_dynamic_observations.sum()),elapsed_s=time.perf_counter()-start)
 (OUT/'preparation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 report=f'''# P0：資料與特徵重建報告

日期：2026-10-13。已完成 A/B 開發資料重建；C 只確認壓縮檔內紀錄數與雜湊，未讀取其模型特徵／標籤作探索或評估。

## 實際產出

- A/B 共 **{len(X):,}** 筆，死亡 **{manifest['deaths']:,}** 筆，占 **{100*manifest['mortality_rate']:.3f}%**。
- F0/F1/F2 分別 **{manifest['family_counts']['0']}／{manifest['family_counts']['1']}／{manifest['family_counts']['2']}** 個未編碼欄位。
- 保存 **{manifest['sequence_rows']:,}** 筆原始觀測及清理旗標，保留原始名稱與缺失觀測，支持序列重建。
- RecordID 與標籤完全對齊，未將結果檔的其他欄位作輸入。
- 年齡低於18有 **{manifest['age_under18']}** 筆；無有效動態觀測 **{manifest['no_dynamic']}** 筆；本輪均保留並標記。
- C 壓縮檔 **{c_count:,}** 筆保留不動。本輪原始資料未修改。

## 清理與先前 EDA 的差異

採官方缺失碼／非負值規則及明確編碼範圍：FiO2 0–1、GCS 3–15、pH 0–14、SaO2／HCT 0–100、Gender 0/1、ICUType 1–4、MechVent 0/1。其他非負極端值與零值先保留供稽核，未套用缺乏檔案依據的舊寬鬆界限。未自動把疑似單位錯誤轉換成新值。因此不可要求與先前全 A/B/C EDA 的每個缺失數完全相同。

GCS 不使用 mean/std/slope；MechVent 使用 ever/first/last 與「有記錄觀測中的比例」，不解讀為通氣時間比例；尿量只作每次記錄量的摘要，不加總為48小時尿量，不產生斜率或delta。官方文件的 TropI/TropT 縮寫對應原始檔的 TroponinI/TroponinT，本輪保留原始名稱並在規則表列出別名。

Weight 的00:00紀錄同時可能參與 admission 與序列摘要，已分開命名，後續相關性分析會呈現重複資訊。std 需至少兩筆，slope 需不同時間點。F2 才把觀測及可計算性指標作輸入。

## 驗證與限制

已檢查唯一鍵、標籤對齊、字典覆蓋及禁用欄位。重要的病人生理零值、單位與極端值仍需資料語意／領域查核；沒有宣稱全面修復臨床資料。未取得患者跨住院身分資訊，分析單位仍是 encounter。

重建耗時 {manifest['elapsed_s']:.1f} 秒。詳見 [manifest](../artifacts/preparation_manifest.json)、[清理統計](../artifacts/measurement_quality.csv)、[字典](../artifacts/feature_dictionary.csv)、[範圍稽核](../artifacts/observed_ranges.csv)。

官方字典與缺失碼：[PhysioNet Challenge 2012]({SOURCE})。
'''
 (REPORT/'P0_資料與特徵重建.md').write_text(report,encoding='utf-8')
 print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()

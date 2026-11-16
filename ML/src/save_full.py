"""Save the full A/B development model without using held-out C."""
import json,joblib,numpy as np,pandas as pd
from study import ART,FD,X,DATA_SHA,basic_cols,fitted
for p in (ART/'selectors').glob('*.joblib'):
 a=joblib.load(p)
 assert set(a['quality']).issubset(set(FD.index))
 assert not any(c.startswith('TropI__') or c.startswith('TropT__') for c in a['quality'])
 # All these caches were generated after the corrected matrix was created.
 a['feature_data_sha256']=DATA_SHA;joblib.dump(a,p)
folder=ART/'final_models';folder.mkdir(exist_ok=True)
b=json.loads((ART/'best_models.json').read_text());m=b['selected_boost']
p=folder/'Full.joblib'
if not p.exists():
 tr=np.arange(len(X));cols=basic_cols(tr);pipe,info=fitted(tr,cols,m,b[m]['params'],'final_development_refit');joblib.dump((pipe,info,cols),p)
else:pipe,info,cols=joblib.load(p)
meta={'Full':{'model':m,'columns':cols,'parameters':b[m]['params'],'trained_on':'A+B (8000 encounters)','test_C_evaluated':False,'data_sha256':DATA_SHA}}
(folder/'metadata.json').write_text(json.dumps(meta,indent=2))
pd.DataFrame({'encoded_column':pipe['pre'].get_feature_names_out()}).to_csv(folder/'Full_encoded_columns.csv',index=False)
before=pipe.predict_proba(X.iloc[:10][cols])[:,1]
assert np.allclose(before,joblib.load(p)[0].predict_proba(X.iloc[:10][cols])[:,1])
print('Full A/B model saved. Total fits:',sum(1 for _ in (ART/'fit_ledger.jsonl').open()))

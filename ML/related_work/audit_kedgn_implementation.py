"""Independent KEDGN provenance/data/numerical audit. Never replaces frozen fits."""
import argparse
import csv
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import numpy as np

ML=Path(__file__).resolve().parents[1]
UP=ML/'artifacts/related_work/KEDGN/upstream'
AUTHOR=ML/'artifacts/related_work/KEDGN/cpu_paper_01/code'
AB=ML/'artifacts/related_work/KEDGN/ab_dense_gpu_20261121'
P53=ML/'artifacts/related_work/KEDGN/frozen_transfer_20261121'
P63=ML/'artifacts/p63_report_five_seed'
ROOT=ML/'artifacts/p64_kedgn_audit'
COMMIT='c82bc004cd1d1305fc58ca7150369d78682fccad'
STATIC=('Age','Gender','Height','ICUType','Weight')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,value):
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def status(value):print(json.dumps(value,ensure_ascii=True),flush=True)

def sources():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip();assert commit==COMMIT
    changes=subprocess.check_output(['git','diff','--name-only',COMMIT,'--','model.py','train.py','utils.py','README.md'],cwd=UP,text=True)
    assert not changes
    differences={}
    for filename in ('model.py','train.py','utils.py'):
        original=subprocess.check_output(['git','show',f'{COMMIT}:{filename}'],cwd=UP).decode('utf-8').replace('\r\n','\n')
        actual=(AUTHOR/filename).read_text(encoding='utf-8').replace('\r\n','\n')
        expected=original
        if filename in ('train.py','utils.py'):expected=expected.replace('.cuda()','.cpu()')
        if filename=='train.py':expected=expected.replace("args.plm + suffix).to(device)","args.plm + suffix, map_location='cpu').to(device)")
        assert actual==expected,filename
        differences[filename]=dict(matches_pinned_author_with_only_declared_CPU_patches=True,
            original_sha256=hashlib.sha256(original.encode()).hexdigest(),working_sha256=sha(AUTHOR/filename),
            diff=list(difflib.unified_diff(original.splitlines(),actual.splitlines(),n=2)))
    for filename in ('ts_params.npy','static_params.npy','extended_static_params.npy'):
        assert np.array_equal(np.load(UP/'data/physionet/processed_data'/filename,allow_pickle=True),
                              np.load(AUTHOR/'data/physionet/processed_data'/filename,allow_pickle=True))
    embedding='data/physionet/physionet_bert_var_rep_gpt_source.pt'
    assert sha(UP/embedding)==sha(AUTHOR/embedding)
    assert sha(UP/'data/physionet/splits/phy12_split5.npy')==sha(AUTHOR/'data/physionet/splits/phy12_split5.npy')
    protocol=read(AB/'protocol.json');frozen=read(AB/'frozen.json')
    assert protocol['runner_sha256']==sha(ML/'related_work/run_kedgn_ab.py')
    assert protocol['dense_runtime_sha256']==sha(ML/'related_work/kedgn_dense_runtime.py')
    assert frozen['protocol_sha256']==sha(AB/'protocol.json')
    assert frozen['selection_sha256']==sha(AB/'selection.json')
    for seed,record in frozen['seeds'].items():
        for filename,field in (('weights.pt','weights_sha256'),('statistics.npz','statistics_sha256')):
            assert sha(AB/f'final_seed_{seed}'/filename)==record[field]
        assert record['seed']==int(seed) and record['epochs']==9 and record['training_stays']==8000
    configured=protocol['model']
    assert configured==dict(hidden_dim=8,num_of_variables=36,num_of_timestamps=215,d_static=9,n_class=2,
        rarity_alpha=1,query_vector_dim=5,node_emb_dim=9,plm_rep_dim=768)
    result=dict(author_commit=commit,author_code_worktree_clean=True,source_differences=differences,
        text_source='ChatGPT descriptions encoded by BERT',embedding_sha256=sha(AUTHOR/embedding),
        dynamic_order_matches_author=True,static_schema_matches_author=True,author_split5_identical=True,
        ab_frozen_sources_and_all_five_weights_verified=True,configuration=configured,
        original_requirements=(UP/'requirements.txt').read_text().splitlines(),ab_runtime=protocol,
        new_formal_fits=0)
    write('sources.json',result);status({'phase':'sources','status':'pass'})

def author_reference(content,variables):
    """Direct interpretation of pinned ParseData/IrregularSampling, without importing ours."""
    rows=list(csv.reader(io.StringIO(content.decode('utf-8-sig'))))[2:]
    fixed=[float(row[2]) for row in rows[:5]]
    age,gender,height,icu,weight=fixed
    static=np.asarray([age,int(gender==0),int(gender==1),height,
                       *[int(icu==i) for i in (1,2,3,4)],weight],dtype=float)
    lookup={name:i for i,name in enumerate(variables)};times=[];events=[]
    for clock,param,value in rows[5:]:
        if param not in lookup:continue
        t=60.*float(clock[:2])+float(clock[3:5])
        if t>=2880:continue
        if t not in times:times.append(t)
        events.append((t,lookup[param],float(value)))
    arr=np.zeros((max(215,len(times)),36));clock=np.zeros((len(arr),1));index={t:i for i,t in enumerate(times)}
    for t,col,value in events:arr[index[t],col]=value;clock[index[t],0]=t
    return arr,clock,static,len(times),tuple(row[1] for row in rows[:5])!=STATIC

def data():
    sys.path.insert(0,str(ML/'src'));sys.path.insert(0,str(ML/'related_work'))
    from prepare import DATA
    from kedgn_transfer_inputs import raw_record
    work=np.load(AUTHOR/'data/physionet/processed_data/PTdict_list.npy',allow_pickle=True)
    by_work={int(r['id']):r for r in work}
    variables=np.load(AUTHOR/'data/physionet/processed_data/ts_params.npy',allow_pickle=True).tolist()
    same={key:0 for key in ('arr','time','static','extended_static','length')}
    for key,row in by_work.items():
        content=(DATA/'set-a'/f'{key}.txt').read_bytes()
        ref,ref_clock,ref_static,ref_length,_=author_reference(content,variables)
        basic=[float(r[2]) for r in list(csv.reader(io.StringIO(content.decode('utf-8-sig'))))[2:7]]
        reference=dict(arr=ref,time=ref_clock,static=basic,extended_static=ref_static,length=ref_length)
        for field in same:same[field]+=int(np.array_equal(row[field],reference[field],equal_nan=True))
    labels_work=np.load(AUTHOR/'data/physionet/processed_data/arr_outcomes.npy',allow_pickle=True)
    labels_original=np.load(UP/'data/physionet/processed_data/arr_outcomes.npy',allow_pickle=True)
    assert np.array_equal(labels_work,labels_original)
    assert np.array_equal([int(row['id']) for row in work],labels_original[:,0])
    packed=np.load(AB/'inputs/ab.npz');ids=packed['ids'];offsets=packed['offsets'];arr=packed['arr'];clock=packed['time'];static=packed['static'];lengths=packed['lengths']
    index={int(stay):i for i,stay in enumerate(ids)}
    comparisons={};changed_private={}
    for part in ('a','b'):
        counts=dict(stays=0,raw_hashes_verified=0,cache_matches_current_parser=0,
            author_reference_dynamic_exact=0,author_reference_time_exact=0,author_reference_static_exact=0,
            author_reference_length_exact=0,nonstandard_first_five_rows=0,
            repaired_static_only=0,repaired_dynamic=0)
        examples=[]
        raw_hashes=read(AB/'inputs/manifest.json')['raw_files_sha256']
        for path in sorted((DATA/f'set-{part}').glob('*.txt')):
            stay=int(path.stem);j=index[stay];content=path.read_bytes();start,end=offsets[j:j+2]
            assert hashlib.sha256(content).hexdigest()==raw_hashes[f'set-{part}/{path.name}']
            counts['raw_hashes_verified']+=1;counts['stays']+=1
            current=raw_record(content,stay,variables)
            assert np.array_equal(current['arr'],arr[start:end])
            assert np.array_equal(current['time'],clock[start:end])
            assert np.array_equal(current['extended_static'],static[j]) and current['length']==lengths[j]
            counts['cache_matches_current_parser']+=1
            ref,ref_clock,ref_static,ref_length,repaired=author_reference(content,variables)
            flags=dict(dynamic=np.array_equal(ref,arr[start:end]),time=np.array_equal(ref_clock,clock[start:end]),
                       static=np.array_equal(ref_static,static[j]),length=(ref_length==lengths[j]))
            for label,target in [('dynamic','author_reference_dynamic_exact'),('time','author_reference_time_exact'),
                                 ('static','author_reference_static_exact'),('length','author_reference_length_exact')]:counts[target]+=int(flags[label])
            counts['nonstandard_first_five_rows']+=int(repaired)
            counts['repaired_static_only']+=int(not flags['static'] and all(flags[k] for k in ('dynamic','time','length')))
            counts['repaired_dynamic']+=int(not all(flags[k] for k in ('dynamic','time','length')))
            if not all(flags.values()):examples.append(dict(stay=stay,**{k:bool(v) for k,v in flags.items()}))
        comparisons[part]=counts;changed_private[part]=examples;status({'phase':'data','part':part,**counts})
    # Match the current raw parser across ALL original Set A rows, not only the original 30 probes.
    parser_work={key:0 for key in ('arr','time','extended_static','length')}
    cache_work={key:0 for key in parser_work}
    for stay,row in by_work.items():
        j=index[stay];start,end=offsets[j:j+2]
        current=raw_record((DATA/'set-a'/f'{stay}.txt').read_bytes(),stay,variables)
        for key in parser_work:parser_work[key]+=int(np.array_equal(current[key],row[key],equal_nan=True))
        for key,value in [('arr',arr[start:end]),('time',clock[start:end]),('extended_static',static[j]),('length',lengths[j])]:
            cache_work[key]+=int(np.array_equal(value,row[key],equal_nan=True))
    # C stored records and raw outcomes: only audit; no decisions or model selection.
    with np.load(P53/'set_c_portable.npz') as c:
        c_arr,c_time,c_static,c_length,c_offsets=[c[k] for k in ('arr','time','static','lengths','offsets')]
    with np.load(P53/'set_c_labels.npz') as c: c_ids,c_y=c['ids'],c['y']
    c_index={int(i):j for j,i in enumerate(c_ids)}
    c_counts=dict(stays=0,cache_matches_current_parser=0,author_reference_dynamic_exact=0,author_reference_static_exact=0,
                  author_reference_time_exact=0,author_reference_length_exact=0,nonstandard_first_five_rows=0)
    with tarfile.open(DATA/'set-c.tar.gz') as archive:
        for member in archive:
            if not member.isfile() or not Path(member.name).stem.isdigit():continue
            stay=int(Path(member.name).stem);j=c_index[stay];content=archive.extractfile(member).read();start,end=c_offsets[j:j+2]
            current=raw_record(content,stay,variables)
            assert np.array_equal(current['arr'],c_arr[start:end]) and np.array_equal(current['time'],c_time[start:end])
            assert np.array_equal(current['extended_static'],c_static[j]) and current['length']==c_length[j]
            c_counts['stays']+=1;c_counts['cache_matches_current_parser']+=1
            ref,ref_clock,ref_static,ref_length,repaired=author_reference(content,variables)
            for label,value in [('dynamic',np.array_equal(ref,c_arr[start:end])),('time',np.array_equal(ref_clock,c_time[start:end])),
                                ('static',np.array_equal(ref_static,c_static[j])),('length',ref_length==c_length[j])]:
                c_counts['author_reference_'+label+'_exact']+=int(value)
            c_counts['nonstandard_first_five_rows']+=int(repaired)
    result=dict(author_processed_stays=len(work),author_work_vs_independent_raw_reference=same,
        bundled_original_PTdict_available=(UP/'data/physionet/processed_data/PTdict_list.npy').exists(),
        labels_and_author_record_order_exact=True,all_Set_A_current_parser_vs_original_reproduction=parser_work,
        all_Set_A_cache_vs_original_reproduction=cache_work,AB_raw_comparison=comparisons,C_raw_comparison=c_counts,
        AB_stays=len(ids),AB_deaths=int(packed['y'].sum()),C_stays=len(c_ids),C_deaths=int(c_y.sum()),
        AB_C_overlap=len(set(ids)&set(c_ids)),all_AB_raw_and_cache_verified=True,new_formal_fits=0)
    assert result['AB_C_overlap']==0
    write('data.json',result);write('private_parser_differences.json',changed_private)
    status({'phase':'data','status':'complete','original_processed_exact_counts':same,'set_c':c_counts})

def runtime(device_name):
    import torch
    from sklearn.metrics import average_precision_score,roc_auc_score
    sys.path.insert(0,str(ML/'related_work'));import run_kedgn_ab as r
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(device_name)
    embeddings=torch.load(AUTHOR/'data/physionet/physionet_bert_var_rep_gpt_source.pt',map_location=device)
    rows,pack=r.records(AB/'inputs/ab.npz');y=pack['y'];length=pack['lengths']
    selected=set(np.linspace(0,len(rows)-1,12,dtype=int).tolist())
    selected.update(np.argsort(length)[-4:].tolist());selected.update(np.flatnonzero(length==0)[:4].tolist())
    selected.update(np.flatnonzero((length>0)&(length<20))[:4].tolist())
    selected=np.asarray(sorted(selected));probes=rows[selected];target=y[selected]
    with np.load(AB/'final_seed_0/statistics.npz') as a:stats=tuple(a[k] for k in ('mf','sf','ms','ss'))
    # Recompute statistics on all AB and prove the custom streaming implementation agrees with author getStats.
    actual_stats=r.statistics(rows)
    for a,b in zip(actual_stats,stats):np.testing.assert_array_equal(a,b)
    padded_stats=r.padded(rows)
    dynamic=np.stack([x['arr'] for x in padded_stats])
    expected_stats=r.getStats(dynamic)
    stat_error=[float(np.max(np.abs(a-b))) for a,b in zip(expected_stats,stats[:2])]
    for a,b in zip(expected_stats,stats[:2]):np.testing.assert_array_equal(a,b)
    del dynamic,padded_stats
    source_tensors=r.tensorize(probes,target,stats)
    direct=r.tensorize_normalize_extract_feature(r.padded(probes),target[:,None],*stats)
    for a,b in zip(source_tensors,direct):torch.testing.assert_close(a,b,rtol=0,atol=0)
    results=[]
    for seed in range(5):
        state=torch.load(AB/f'final_seed_{seed}/weights.pt',map_location=device)
        original=r.model_for(device,seed,dense=False);optimized=r.model_for(device,seed,dense=True)
        original.load_state_dict(state);optimized.load_state_dict(state);original.eval();optimized.eval()
        with torch.no_grad():
            out=r.forward(original,source_tensors,slice(None),embeddings,device)
            fast=r.forward(optimized,source_tensors,slice(None),embeddings,device)
        torch.testing.assert_close(out,fast,rtol=2e-4,atol=2e-5)
        p=torch.softmax(out.double(),dim=1)[:,1].cpu().numpy();q=torch.softmax(fast.double(),dim=1)[:,1].cpu().numpy()
        # Compare parameter gradients of TRAINED weights, including inactive parameters.
        original.zero_grad(set_to_none=True);optimized.zero_grad(set_to_none=True)
        original.train();optimized.train()
        torch.nn.functional.cross_entropy(r.forward(original,source_tensors,slice(None),embeddings,device),source_tensors[5].to(device)).backward()
        torch.nn.functional.cross_entropy(r.forward(optimized,source_tensors,slice(None),embeddings,device),source_tensors[5].to(device)).backward()
        error=0.;active=0;none=[]
        op=dict(original.named_parameters())
        for name,param in optimized.named_parameters():
            expected=op[name]
            assert (param.grad is None)==(expected.grad is None),name
            if param.grad is None:none.append(name);continue
            active+=1;error=max(error,float((param.grad-expected.grad).abs().max()))
            torch.testing.assert_close(param.grad,expected.grad,rtol=1e-3,atol=6e-5)
        results.append(dict(seed=seed,probe_cases=len(selected),logits_max_abs=float((out-fast).abs().max()),
            probability_max_abs=float(np.max(np.abs(p-q))),gradient_max_abs=error,active_parameters=active,
            inactive_parameter_names=none))
        status({'phase':'runtime','frozen_seed':seed,**{k:results[-1][k] for k in ('logits_max_abs','probability_max_abs','gradient_max_abs')}})
        del original,optimized,state
    # Three optimizer updates on a fixed real AB mini-batch, same initialization and indices.
    original=r.model_for(device,23,dense=False);optimized=r.model_for(device,23,dense=True)
    optimized.load_state_dict(original.state_dict())
    old_opt=torch.optim.Adam(original.parameters(),lr=.001);new_opt=torch.optim.Adam(optimized.parameters(),lr=.001)
    updates=[]
    for step in range(3):
        original.train();optimized.train();old_opt.zero_grad(set_to_none=True);new_opt.zero_grad(set_to_none=True)
        a=r.forward(original,source_tensors,slice(None),embeddings,device);b=r.forward(optimized,source_tensors,slice(None),embeddings,device)
        la=torch.nn.functional.cross_entropy(a,source_tensors[5].to(device));lb=torch.nn.functional.cross_entropy(b,source_tensors[5].to(device))
        la.backward();lb.backward();old_opt.step();new_opt.step()
        delta=max(float((x-y).abs().max()) for x,y in zip(original.parameters(),optimized.parameters()))
        with torch.no_grad():
            post_a=r.forward(original,source_tensors,slice(None),embeddings,device)
            post_b=r.forward(optimized,source_tensors,slice(None),embeddings,device)
        torch.testing.assert_close(post_a,post_b,rtol=1e-3,atol=1e-4)
        updates.append(dict(update=step+1,original_loss=float(la),dense_loss=float(lb),parameter_max_abs=delta,
                            post_update_logits_max_abs=float((post_a-post_b).abs().max())))
    # Fresh original-author inference of frozen AB weights over EVERY clean Set C patient.
    c_rows,c_pack=r.records(P53/'set_c_portable.npz')
    with np.load(P53/'set_c_labels.npz') as a:c_ids,c_y=a['ids'],a['y']
    with np.load(AB/'set_c_predictions.npz') as a:cached=a['probabilities'].copy();np.testing.assert_array_equal(a['ids'],c_ids);np.testing.assert_array_equal(a['y'],c_y)
    order=np.argsort([x['length'] for x in c_rows],kind='stable')
    replay=[]
    fresh=np.empty_like(cached)
    for seed in range(5):
        with np.load(AB/f'final_seed_{seed}/statistics.npz') as a:stats=tuple(a[k] for k in ('mf','sf','ms','ss'))
        original=r.model_for(device,seed,dense=False)
        original.load_state_dict(torch.load(AB/f'final_seed_{seed}/weights.pt',map_location=device));original.eval()
        for start in range(0,len(order),64):
            idx=order[start:start+64];tensors=r.tensorize(c_rows[idx],c_y[idx],stats)
            fresh[seed,idx]=r.predict(original,tensors,embeddings,device,batch=64)
            if start%1024==0:status({'phase':'runtime','original_Set_C_seed':seed,'completed':min(start+64,len(order)),'total':len(order)})
        difference=float(np.max(np.abs(fresh[seed]-cached[seed])))
        row=dict(seed=seed,ap_original=float(average_precision_score(c_y,fresh[seed])),ap_cached=float(average_precision_score(c_y,cached[seed])),
                 auroc_original=float(roc_auc_score(c_y,fresh[seed])),probability_max_abs=difference)
        assert difference<1e-4
        assert abs(row['ap_original']-row['ap_cached'])<1e-6
        replay.append(row);status({'phase':'runtime','Set_C_original_replay':row})
        del original
    np.savez_compressed(ROOT/'private_original_Set_C_replay.npz',ids=c_ids,y=c_y,probabilities=fresh)
    result=dict(device=str(device),torch=torch.__version__,numpy=np.__version__,AB_stats_exact=True,
        AB_direct_author_preprocessing_exact=True,AB_stats_max_error=stat_error,trained_model_checks=results,
        original_vs_dense_optimizer_updates=updates,Set_C_original_forward_all_five_models=replay,
        new_formal_fits=0,numerical_diagnostic_optimizers=2,numerical_updates_per_optimizer=3,
        frozen_weights_unchanged=True,complete=True)
    write('runtime.json',result);status({'phase':'runtime','status':'complete'})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('sources','data','runtime'))
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    {'sources':sources,'data':data,'runtime':lambda:runtime(args.device)}[args.phase]()

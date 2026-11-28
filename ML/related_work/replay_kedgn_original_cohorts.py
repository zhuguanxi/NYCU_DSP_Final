"""Replay frozen author-split and Demo weights with the untouched author cell."""
import json,os,sys,tarfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,recall_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_kedgn_ab as r
from audit_kedgn_implementation import ROOT,AB,sha,write,status,author_reference

def metrics(y,p):return dict(ap=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p)),
    brier=float(brier_score_loss(y,p)),sensitivity=float(recall_score(y,p>=.5)))

def load_stats(path):
    with np.load(path) as a:
        return tuple(a[k] for k in ('mf','sf' if 'sf' in a.files else 'stdf','ms','ss'))

def replay(rows,y,ids,stats,paths,embeddings,device,case,cached=None):
    order=np.argsort([row['length'] for row in rows],kind='stable')
    values=np.empty((5,len(rows)));results=[]
    for seed,path in enumerate(paths):
        model=r.model_for(device,seed,dense=False)
        model.load_state_dict(torch.load(path,map_location=device));model.eval()
        for start in range(0,len(rows),256):
            idx=order[start:start+256];tensors=r.tensorize(rows[idx],y[idx],stats)
            values[seed,idx]=r.predict(model,tensors,embeddings,device,batch=256)
        score=metrics(y,values[seed]);row=dict(cohort=case,seed=seed,stays=len(y),deaths=int(y.sum()),**score)
        if cached is not None:
            row['maximum_probability_difference_from_original_cache']=float(np.max(np.abs(values[seed]-cached[seed])))
            row['ap_difference_from_original_cache']=score['ap']-metrics(y,cached[seed])['ap']
            before=cached[seed,:,None]-cached[seed,None,:]
            after=values[seed,:,None]-values[seed,None,:]
            changed=np.triu(np.sign(before)!=np.sign(after),1)
            row['changed_score_order_or_tie_pairs']=int(changed.sum())
            row['maximum_cached_probability_gap_of_changed_pair']=float(np.max(np.abs(before[changed]))) if changed.any() else 0.
            row['changed_pairs_confined_to_numeric_error_interval']=bool(row['maximum_cached_probability_gap_of_changed_pair']<=2*row['maximum_probability_difference_from_original_cache']+1e-12)
            # AP is discontinuous at ties: report its actual difference rather than silently rounding it away.
            np.savez_compressed(ROOT/f'private_original_{case}_seed_{seed}.npz',ids=ids,y=y,probabilities=values[seed])
            write(f'{case}_seed_{seed}_numerics.json',row)
            assert row['maximum_probability_difference_from_original_cache']<1e-4
            assert row['changed_pairs_confined_to_numeric_error_interval']
        results.append(row);status(row);del model
    np.savez_compressed(ROOT/f'private_original_{case}_replay.npz',ids=ids,y=y,probabilities=values)
    return results

def parser_probe(embeddings,device):
    """Test-only paired parser sensitivity; does NOT recreate author-style AB training."""
    sys.path.insert(0,str(r.ML/'src'));from prepare import DATA
    from kedgn_transfer_inputs import raw_record
    with np.load(r.P53/'set_c_labels.npz') as a:ids,y=a['ids'],a['y']
    index={int(stay):j for j,stay in enumerate(ids)}
    variables=np.load(r.AUTHOR/'data/physionet/processed_data/ts_params.npy',allow_pickle=True).tolist()
    positions=[];rows=[]
    with tarfile.open(DATA/'set-c.tar.gz') as archive:
        for member in archive:
            if not member.isfile() or not Path(member.name).stem.isdigit():continue
            stay=int(Path(member.name).stem);content=archive.extractfile(member).read()
            ref,clock,static,length,_=author_reference(content,variables)
            corrected=raw_record(content,stay,variables)
            if all((np.array_equal(ref,corrected['arr']),np.array_equal(clock,corrected['time']),
                    np.array_equal(static,corrected['extended_static']),length==corrected['length'])):continue
            positions.append(index[stay]);rows.append(dict(arr=ref,time=clock,extended_static=static,length=length))
    positions=np.asarray(positions);rows=np.asarray(rows,dtype=object);result=[]
    for version in ('set_a','ab'):
        folder=r.P53 if version=='set_a' else AB
        with np.load(folder/'set_c_predictions.npz') as a:
            baseline=a['probabilities'].copy();np.testing.assert_array_equal(a['ids'],ids);np.testing.assert_array_equal(a['y'],y)
        alternative=baseline.copy();frozen=json.loads((folder/('inference.json' if version=='set_a' else 'frozen.json')).read_text())
        for seed in range(5):
            if version=='set_a':
                path=next(Path(p) for p in frozen['checkpoints_sha256'] if Path(p).stem.endswith('_'+str(seed)))
                statfile=r.P53/'training_statistics.npz'
            else:path=AB/f'final_seed_{seed}/weights.pt';statfile=AB/f'final_seed_{seed}/statistics.npz'
            stats=load_stats(statfile)
            model=r.model_for(device,seed,dense=False);model.load_state_dict(torch.load(path,map_location=device));model.eval()
            tensors=r.tensorize(rows,y[positions],stats)
            alternative[seed,positions]=r.predict(model,tensors,embeddings,device,batch=128)
            original_score=metrics(y,baseline[seed]);other_score=metrics(y,alternative[seed])
            result.append(dict(version=version,seed=seed,changed_stays=len(positions),
                corrected_parser=original_score,author_positional_test_parser=other_score,
                delta_ap=other_score['ap']-original_score['ap']))
            status({'phase':'parser_test_only','version':version,'seed':seed,'changed_stays':len(positions),'delta_ap':result[-1]['delta_ap']})
            del model
        np.savez_compressed(ROOT/f'private_positional_parser_{version}_C.npz',ids=ids,y=y,probabilities=alternative)
    write('parser_test_sensitivity.json',dict(rows=result,changed_stays=len(positions),new_fits=0,
        test_preprocessing_only=True,does_not_measure_changed_training_parser_effect=True,
        primary_scores_unchanged=True,selected_on_test=False))

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    embeddings=torch.load(r.AUTHOR/'data/physionet/physionet_bert_var_rep_gpt_source.pt',map_location=device)
    checkpoint_checks=[]
    frozen=json.loads((AB/'frozen.json').read_text())
    for seed in range(5):
        folder=AB/f'final_seed_{seed}';last=torch.load(folder/'last.pt',map_location='cpu')
        weight=torch.load(folder/'weights.pt',map_location='cpu');receipt=frozen['seeds'][str(seed)]
        assert last['seed']==seed and last['epoch']==receipt['epochs']==9
        assert set(last['model'])==set(weight)
        assert all(torch.equal(v,weight[k]) for k,v in last['model'].items())
        steps={int(v['step']) for v in last['optimizer']['state'].values()}
        assert steps=={receipt['epochs']*receipt['batches_per_epoch']}=={234}
        assert all(g['lr']==.001 for g in last['optimizer']['param_groups'])
        checkpoint_checks.append(dict(seed=seed,last_and_frozen_weight_exact=True,epochs=9,
            optimizer_update_steps=234,effective_batch=256,learning_rate=.001))
    write('checkpoint_checks.json',dict(rows=checkpoint_checks,complete=True,new_fits=0))
    inference=json.loads((r.P53/'inference.json').read_text())
    from utils import get_data_split
    train,valid,test,ytrain,yvalid,ytest=get_data_split(str(r.AUTHOR/'data/physionet'),'/splits/phy12_split5.npy',dataset='physionet')
    stats=r.statistics(train)
    for value,stored in zip(stats,load_stats(r.P53/'training_statistics.npz')):np.testing.assert_array_equal(value,stored)
    assert len(train)==3198 and len(valid)==400 and len(test)==399
    test_ids=np.asarray([int(row['id']) for row in test]);target=ytest[:,0]
    paths=[next(Path(path) for path in inference['checkpoints_sha256'] if Path(path).stem.endswith('_'+str(seed))) for seed in range(5)]
    for path in paths:assert sha(path)==inference['checkpoints_sha256'][str(path)]
    results=replay(test,target,test_ids,stats,paths,embeddings,device,'author_test')
    for row,old in zip(results,inference['author_replay']):
        row['difference_from_CPU_author_replay_ap']=row['ap']-old['ap_percent']/100
        assert abs(row['difference_from_CPU_author_replay_ap'])<1e-5
    write('author_test_replay.json',dict(rows=results,training_statistics_exact=True,original_cell=True,
        trained_on_author_training_split=True,five_seed_metric_mean=float(np.mean([x['ap'] for x in results])),
        sample_sd=float(np.std([x['ap'] for x in results],ddof=1)),population_sd=float(np.std([x['ap'] for x in results])),
        new_fits=0))
    del train,valid,test
    all_results=[]
    for case in ('eicu_hourly','eicu_native'):
        rows,pack=r.records(r.P53/f'{case}_portable.npz')
        with np.load(r.P53/f'{case}_labels.npz') as a:ids,y=a['ids'],a['y']
        with np.load(AB/f'{case}_predictions.npz') as a:
            cached=a['probabilities'].copy();np.testing.assert_array_equal(a['ids'],ids);np.testing.assert_array_equal(a['y'],y)
        with np.load(AB/'final_seed_0/statistics.npz') as a:stats=tuple(a[k] for k in ('mf','sf','ms','ss'))
        paths=[AB/f'final_seed_{seed}/weights.pt' for seed in range(5)]
        out=replay(rows,y,ids,stats,paths,embeddings,device,case,cached)
        all_results.extend(out)
        write('demo_replay_progress.json',dict(rows=all_results,new_fits=0,original_cell=True,complete=(case=='eicu_native')))
    write('demo_replay.json',dict(rows=all_results,new_fits=0,original_cell=True,complete=True))
    parser_probe(embeddings,device)
    status({'phase':'extra_original_replay','status':'complete','new_fits':0})

if __name__=='__main__':main()

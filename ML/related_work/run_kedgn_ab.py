"""Train KEDGN using A/B only, seal epoch choice, then evaluate frozen weights."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from kedgn_dense_runtime import install as install_dense

ML = Path(__file__).resolve().parents[1]
AUTHOR = ML / 'artifacts/related_work/KEDGN/cpu_paper_01/code'
P53 = ML / 'artifacts/related_work/KEDGN/frozen_transfer_20261121'
sys.path.insert(0, str(AUTHOR))
from model import KEDGN
from utils import getStats, getStats_static, tensorize_normalize_extract_feature

CONFIG = dict(hidden_dim=8, num_of_variables=36, num_of_timestamps=215,
              d_static=9, n_class=2, rarity_alpha=1, query_vector_dim=5,
              node_emb_dim=9, plm_rep_dim=768)
PROTOCOL = dict(max_epochs=10, learning_rate=0.001, effective_batch=256,
                development_seed=0, final_seeds=list(range(5)),
                selection='Maximum pooled three-fold OOF softmax AP; ties choose earlier epoch',
                hyperparameters='Fixed to completed author reproduction; only epochs selected on A/B')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def save(path, value):
    temporary = path.with_suffix('.pending.pt')
    torch.save(value, temporary)
    os.replace(temporary, path)


def records(path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    rows = np.asarray([dict(arr=arrays['arr'][start:end], time=arrays['time'][start:end],
                            extended_static=static, length=int(length))
                       for start,end,static,length in zip(arrays['offsets'][:-1], arrays['offsets'][1:],
                                                         arrays['static'], arrays['lengths'])], dtype=object)
    return rows, arrays


def padded(rows, size=None):
    size = size or max(215, max(row['length'] for row in rows))
    result=[]
    for source in rows:
        row = dict(source)
        row['arr'] = np.zeros((size,36), dtype=np.float64)
        row['time'] = np.zeros((size,1), dtype=np.float64)
        row['arr'][:source['length']] = source['arr'][:source['length']]
        row['time'][:source['length']] = source['time'][:source['length']]
        result.append(row)
    return result


def tensorize(rows, y, stats):
    size=max(215,max(row['length'] for row in rows))
    chunks=[tensorize_normalize_extract_feature(padded(rows[start:start+128],size),
             np.asarray(y[start:start+128])[:,None],*stats) for start in range(0,len(rows),128)]
    return tuple(torch.cat([part[k] for part in chunks]) for k in range(6))


def statistics(rows, require_finite=True):
    means=[]
    deviations=[]
    for channel in range(36):
        pieces=[row['arr'][:row['length'],channel] for row in rows]
        values=np.concatenate([part[part>0] for part in pieces])
        means.append(float(np.mean(values)) if len(values) else 0.)
        deviations.append(float(np.std(values)))
    mf,sf=np.asarray(means)[:,None],np.asarray(deviations)[:,None]
    ms, ss = getStats_static(np.asarray([row['extended_static'] for row in rows]), dataset='physionet')
    values = (mf,sf,ms,ss)
    if require_finite and not all(np.isfinite(x).all() for x in values):
        raise ValueError('Nonfinite training-only statistics')
    return values


def model_for(device, seed, dense=True):
    torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    model=KEDGN(DEVICE=device, **CONFIG)
    return install_dense(model) if dense else model


def forward(model, tensors, positions, embeddings, device):
    batch = [value[positions].to(device) for value in tensors[:5]]
    return model(*batch, embeddings)


def predict(model, tensors, embeddings, device, batch=256):
    result = np.empty(len(tensors[0]), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0,len(result),batch):
            logits = forward(model,tensors,slice(start,start+batch),embeddings,device)
            values = logits.detach().cpu().numpy().astype(np.float64)
            values -= values.max(axis=1,keepdims=True)
            values = np.exp(values)
            result[start:start+batch] = (values/values.sum(axis=1,keepdims=True))[:,1]
    if not (np.isfinite(result)&(result>=0)&(result<=1)).all():
        raise ValueError('Invalid prediction')
    return result


def train_job(root, name, rows, y, valid_rows, valid_y, device, microbatch, seed, epochs, embeddings):
    folder = root/name
    folder.mkdir(exist_ok=True)
    receipt_path = folder/'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['epochs'] != epochs or sha(folder/'weights.pt') != receipt['weights_sha256']:
            raise ValueError('Completed training receipt changed')
        print(f'Retaining completed {name}',flush=True)
        return receipt
    stats_path=folder/'statistics.npz'
    if stats_path.exists():
        with np.load(stats_path) as a: stats=tuple(a[k] for k in ('mf','sf','ms','ss'))
    else:
        stats=statistics(rows)
        np.savez_compressed(stats_path, **dict(zip(('mf','sf','ms','ss'),stats)))
    tensors=tensorize(rows,y,stats)
    valid=tensorize(valid_rows,valid_y,stats) if valid_rows is not None else None
    model=model_for(device,seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=PROTOCOL['learning_rate'])
    np.random.seed(seed)
    zeros=np.flatnonzero(y==0)
    ones=np.tile(np.flatnonzero(y==1),3)
    batches=min(len(zeros)//128,len(ones)//128)
    start_epoch=0
    history=[]
    last=folder/'last.pt'
    if last.exists():
        state=torch.load(last,map_location=device)
        if state['protocol']!=PROTOCOL or state['microbatch']!=microbatch or state['seed']!=seed:
            raise ValueError('Resume configuration differs')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        zeros,ones=state['zeros'],state['ones']
        start_epoch,history=state['epoch'],state['history']
        print(f'Resuming {name} after epoch {start_epoch}',flush=True)
        del state
    for epoch in range(start_epoch,epochs):
        started=time.monotonic()
        model.train()
        np.random.shuffle(ones)
        np.random.shuffle(zeros)
        loss_sum=0.
        for step in range(batches):
            positions=np.r_[zeros[step*128:(step+1)*128],ones[step*128:(step+1)*128]]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0,256,microbatch):
                idx=positions[offset:offset+microbatch]
                logits=forward(model,tensors,idx,embeddings,device)
                loss=torch.nn.functional.cross_entropy(logits,tensors[5][idx].to(device))
                weighted=loss*(len(idx)/256)
                weighted.backward()
                loss_sum+=float(weighted.detach().cpu())
            optimizer.step()
        record={'epoch':epoch+1,'training_loss':loss_sum/batches,'seconds':time.monotonic()-started}
        if valid is not None:
            values=predict(model,valid,embeddings,device)
            record.update(ap=float(average_precision_score(valid_y,values)),auroc=float(roc_auc_score(valid_y,values)))
            prediction_path=folder/f'epoch_{epoch+1:02d}_validation.npz'
            np.savez_compressed(prediction_path,probabilities=values)
        history.append(record)
        save(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,
                       numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                       zeros=zeros,ones=ones,history=history,protocol=PROTOCOL,microbatch=microbatch,seed=seed))
        write_json(folder/'history.json',history)
        print(json.dumps({'job':name,**record}),flush=True)
    save(folder/'weights.pt',model.state_dict())
    receipt=dict(epochs=epochs,seed=seed,training_stays=len(y),training_deaths=int(y.sum()),
                 weights_sha256=sha(folder/'weights.pt'),statistics_sha256=sha(stats_path),
                 batches_per_epoch=batches,effective_batch=256,microbatch=microbatch,history=history)
    write_json(receipt_path,receipt)
    del model,optimizer,tensors,valid
    if device.type=='cuda': torch.cuda.empty_cache()
    return receipt


def fit(args, device, embeddings):
    root=args.output
    manifest=json.loads((root/'inputs/manifest.json').read_text())
    if sha(root/'inputs/ab.npz')!=manifest['files_sha256']['ab.npz']:
        raise ValueError('A/B inputs changed')
    for path,digest in manifest['source_sha256'].items():
        if sha(path)!=digest: raise ValueError('A/B source changed')
    rows,data=records(root/'inputs/ab.npz')
    frozen_protocol={'model':CONFIG,'training':PROTOCOL,'input_manifest_sha256':sha(root/'inputs/manifest.json'),
                     'microbatch':args.microbatch,'torch':torch.__version__,'numpy':np.__version__,
                     'runner_sha256':sha(Path(__file__)),
                     'dense_runtime_sha256':sha(Path(__file__).with_name('kedgn_dense_runtime.py')),
                     'runtime':'Dense masked GPU cell; original architecture and parameter names',
                     'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}
    path=root/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=frozen_protocol: raise ValueError('Sealed protocol changed')
    else: write_json(path,frozen_protocol)
    for fold in range(3):
        tr,va=np.flatnonzero(data['folds']!=fold),np.flatnonzero(data['folds']==fold)
        train_job(root,f'fold_{fold}',rows[tr],data['y'][tr],rows[va],data['y'][va],device,
                  args.microbatch,0,PROTOCOL['max_epochs'],embeddings)
    selection_path=root/'selection.json'
    if not selection_path.exists():
        curve=[]
        for epoch in range(1,PROTOCOL['max_epochs']+1):
            pred=np.full(len(rows),np.nan)
            fold_ap=[]
            for fold in range(3):
                idx=np.flatnonzero(data['folds']==fold)
                with np.load(root/f'fold_{fold}/epoch_{epoch:02d}_validation.npz') as saved:
                    pred[idx]=saved['probabilities']
                fold_ap.append(float(average_precision_score(data['y'][idx],pred[idx])))
            curve.append(dict(epoch=epoch,pooled_oof_ap=float(average_precision_score(data['y'],pred)),
                              pooled_oof_auroc=float(roc_auc_score(data['y'],pred)),fold_ap=fold_ap))
        chosen=max(curve,key=lambda row:row['pooled_oof_ap'])
        write_json(selection_path,dict(selected_epoch=chosen['epoch'],curve=curve,training=PROTOCOL,
                                      protocol_sha256=sha(path),test_data_used=False))
    selection=json.loads(selection_path.read_text())
    if selection['protocol_sha256']!=sha(path): raise ValueError('Selection protocol changed')
    for seed in range(5):
        train_job(root,f'final_seed_{seed}',rows,data['y'],None,None,device,args.microbatch,
                  seed,selection['selected_epoch'],embeddings)
    receipts={str(seed):json.loads((root/f'final_seed_{seed}/receipt.json').read_text()) for seed in range(5)}
    write_json(root/'frozen.json',dict(selection_sha256=sha(selection_path),seeds=receipts,
                                    input_manifest_sha256=sha(root/'inputs/manifest.json'),
                                    fits=8,test_data_used=False,protocol_sha256=sha(path)))
    print('A/B-only training completed; all five final models frozen.',flush=True)


def infer(args, device, embeddings):
    root=args.output
    frozen=json.loads((root/'frozen.json').read_text())
    if frozen['selection_sha256']!=sha(root/'selection.json'): raise ValueError('Frozen selection changed')
    manifest=json.loads((P53/'inputs.json').read_text())
    for file,digest in manifest['files_sha256'].items():
        if sha(P53/file)!=digest: raise ValueError('P53 test input changed')
    with np.load(root/'inputs/ab.npz') as ab: training_ids=set(ab['ids'])
    for name in ('set_c','eicu_hourly','eicu_native'):
        destination=root/f'{name}_predictions.npz'
        if destination.exists():
            print(f'Retaining complete {name} prediction',flush=True)
            continue
        rows,unused=records(P53/f'{name}_portable.npz')
        del unused
        with np.load(P53/f'{name}_labels.npz') as labels: ids,y=labels['ids'],labels['y']
        if name=='set_c' and set(ids)&training_ids: raise ValueError('Training and Set C overlap')
        positions=np.argsort([row['length'] for row in rows],kind='stable')
        values=np.full((5,len(rows)),np.nan)
        for seed in range(5):
            folder=root/f'final_seed_{seed}'
            if sha(folder/'weights.pt')!=frozen['seeds'][str(seed)]['weights_sha256']:
                raise ValueError('Frozen weight changed')
            with np.load(folder/'statistics.npz') as saved: stats=tuple(saved[k] for k in ('mf','sf','ms','ss'))
            if sha(folder/'statistics.npz')!=frozen['seeds'][str(seed)]['statistics_sha256']:
                raise ValueError('Frozen preprocessing changed')
            model=model_for(device,seed)
            model.load_state_dict(torch.load(folder/'weights.pt',map_location=device))
            for start in range(0,len(rows),64):
                idx=positions[start:start+64]
                tensors=tensorize(rows[idx],y[idx],stats)
                values[seed,idx]=predict(model,tensors,embeddings,device)
            print(f'{name}: frozen A/B seed {seed} complete',flush=True)
            del model,tensors
        staged=destination.with_suffix('.pending.npz')
        np.savez_compressed(staged,ids=ids,y=y,probabilities=values)
        os.replace(staged,destination)
    write_json(root/'inference.json',dict(device=str(device),model_fits=0,frozen_sha256=sha(root/'frozen.json'),
                                        predictions={name:sha(root/f'{name}_predictions.npz') for name in ('set_c','eicu_hourly','eicu_native')},
                                        primary_prediction='Unweighted mean of five frozen A/B seed probabilities',
                                        p53_inputs_sha256=sha(P53/'inputs.json')))


def verify(args, device, embeddings):
    rows,data=records(args.output/'inputs/ab.npz')
    # Exercise real missingness and the full time window with a small gradient check.
    idx=np.linspace(0,len(rows)-1,8,dtype=int)
    empty=np.flatnonzero(data['lengths']==0)
    if len(empty): idx[0]=empty[0]
    checked=rows[idx]
    sample=statistics(checked,require_finite=False)
    original=getStats(np.stack([row['arr'] for row in padded(checked)]))
    for a,b in zip(sample[:2],original): np.testing.assert_array_equal(a,b)
    tensors=tensorize(rows[idx],data['y'][idx],statistics(rows))
    reference=model_for(torch.device('cpu'),17,dense=False)
    actual=model_for(device,17)
    actual.load_state_dict(reference.state_dict())
    reference.eval(); actual.eval()
    with torch.no_grad():
        expected=forward(reference,tensors,slice(None),embeddings.cpu(),torch.device('cpu'))
        got=forward(actual,tensors,slice(None),embeddings,device).cpu()
    torch.testing.assert_close(expected,got,rtol=2e-4,atol=2e-5)
    actual.train()
    out=forward(actual,tensors,slice(None),embeddings,device)
    torch.nn.functional.cross_entropy(out,tensors[5].to(device)).backward()
    full={k:p.grad.detach().clone() for k,p in actual.named_parameters() if p.grad is not None}
    reference.train()
    original_out=forward(reference,tensors,slice(None),embeddings.cpu(),torch.device('cpu'))
    torch.nn.functional.cross_entropy(original_out,tensors[5]).backward()
    gradient_error=0.
    for key,p in reference.named_parameters():
        if key in full:
            torch.testing.assert_close(p.grad,full[key].cpu(),rtol=8e-4,atol=5e-5)
            gradient_error=max(gradient_error,float((p.grad-full[key].cpu()).abs().max()))
    actual.zero_grad(set_to_none=True)
    for start in (0,4):
        out=forward(actual,tensors,slice(start,start+4),embeddings,device)
        (torch.nn.functional.cross_entropy(out,tensors[5][start:start+4].to(device))/2).backward()
    for key,p in actual.named_parameters():
        if key in full: torch.testing.assert_close(full[key],p.grad,rtol=5e-4,atol=3e-5)
    write_json(args.output/'runtime_verification.json',dict(cpu_gpu_logits_max_abs=float((expected-got).abs().max()),
                                                          dense_original_gradient_max_abs=gradient_error,
                                                          original_dense_gradients_verified=True,
                                                          gradient_accumulation_verified=True,device=str(device),
                                                          torch=torch.__version__,samples=8))
    print('CPU/GPU inference and gradient accumulation checks passed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('verify','fit','infer'))
    parser.add_argument('--output',type=Path,default=ML/'artifacts/related_work/KEDGN/ab_dense_gpu_20261121')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--microbatch',type=int,default=256)
    args=parser.parse_args()
    args.output=args.output.resolve()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('Requested GPU is unavailable')
    embeddings=torch.load(AUTHOR/'data/physionet/physionet_bert_var_rep_gpt_source.pt',map_location=device)
    {'verify':verify,'fit':fit,'infer':infer}[args.phase](args,device,embeddings)

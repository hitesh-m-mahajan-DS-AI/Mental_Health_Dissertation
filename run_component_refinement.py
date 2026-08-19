"""Execute attention/MLP refinement for explicitly selected causal layers."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any
import numpy as np

from src.causal_component_refinement import (capture_component_outputs, compact_refinement_report,
    induced_nodes, intervene_component_outputs, normalized_recovery, rank_candidate_nodes,
    validate_refinement_prerequisites, wang_style_subset_metrics, write_compact_refinement_report)
from src.causal_config import load_causal_config
from src.causal_pairs import build_pairs, validate_and_tokenize_pair
from src.causal_patching import _capture_compatible_config, _forward, _scores
from src.model_loader import load_model, validate_local_checkpoint

def score(model: Any, ids: list[list[int]], lengths: list[int], sid: int, nid: int, pad: int) -> list[float]:
    return _scores(_forward(model, ids, pad), lengths, sid, nid)

def run_dataset(model: Any, tokenizer: Any, cfg: Any, dataset: str, layers: tuple[int, ...], batch: int) -> dict[str, Any]:
    nodes=induced_nodes(layers); by_key={n.key:n for n in nodes}; universe=tuple(by_key)
    pairs=build_pairs(cfg,dataset); recoveries=[]; compact=[]; pad=int(tokenizer.eos_token_id)
    for start in range(0,200,batch):
        current=pairs[start:start+batch]
        enc=[validate_and_tokenize_pair(p,tokenizer,cfg) for p in current]
        clean_ids=[e['clean_input_ids'] for e in enc]; corrupt_ids=[e['corrupted_input_ids'] for e in enc]
        cl=[len(x) for x in clean_ids]; rl=[len(x) for x in corrupt_ids]
        pos=[e['patch_position'] for e in enc]; sid=enc[0]['supportive_target_id']; nid=enc[0]['neutral_target_id']
        with capture_component_outputs(model,nodes) as clean_cache: clean=score(model,clean_ids,cl,sid,nid,pad)
        corrupt=score(model,corrupt_ids,rl,sid,nid,pad)
        patched={}
        for key,node in by_key.items():
            with intervene_component_outputs(model,clean_cache,[node],pos,mode='restore'):
                patched[key]=score(model,corrupt_ids,rl,sid,nid,pad)
        with intervene_component_outputs(model,clean_cache,nodes,pos,mode='restore'):
            all_nodes=score(model,corrupt_ids,rl,sid,nid,pad)
        for i,pair in enumerate(current):
            row={key:normalized_recovery(values[i],clean[i],corrupt[i]) for key,values in patched.items()}
            recoveries.append(row); compact.append({'example_id':pair.example_id,'selection_ordinal':pair.selection_ordinal,
                'clean_score':clean[i],'corrupted_score':corrupt[i],'denominator_valid':abs(clean[i]-corrupt[i])>cfg.zero_denominator_epsilon,
                'node_recovery':row,'all_nodes_score':all_nodes[i]})
        print(f'{dataset}: {min(start+batch,200)}/200',flush=True)
    ranking=rank_candidate_nodes(recoveries)
    candidate=[r['node'] for r in ranking if r['mean_recovery']>0]
    if not candidate: candidate=[ranking[0]['node']]
    complement=[key for key in universe if key not in candidate]
    per_record_metrics=[]
    for row in compact:
        if not row['denominator_valid']: continue
        if set(candidate)==set(universe): candidate_score=row['all_nodes_score']
        else: candidate_score=row['corrupted_score']+(row['clean_score']-row['corrupted_score'])*sum(row['node_recovery'][k] for k in candidate)
        if not complement: complement_score=row['corrupted_score']
        elif set(complement)==set(universe): complement_score=row['all_nodes_score']
        else: complement_score=row['corrupted_score']+(row['clean_score']-row['corrupted_score'])*sum(row['node_recovery'][k] for k in complement)
        drop={}
        for key in candidate:
            remaining=[k for k in candidate if k!=key]
            drop[key]=(row['corrupted_score'] if not remaining else
                row['corrupted_score']+(row['clean_score']-row['corrupted_score'])*sum(row['node_recovery'][k] for k in remaining))
        per_record_metrics.append(wang_style_subset_metrics(candidate=candidate,universe=universe,
            clean_score=row['clean_score'],corrupted_score=row['corrupted_score'],
            candidate_restoration_score=candidate_score,complement_restoration_score=complement_score,
            drop_one_restoration_scores=drop))
    aggregate_metrics={key:float(np.mean([m[key] for m in per_record_metrics]))
        for key in ('faithfulness','completeness','minimality_mean')}
    return {'records':200,'valid_denominator_records':sum(r['denominator_valid'] for r in compact),
        'universe':list(universe),'ranking':ranking,'candidate':candidate,'records_compact':compact,
        'subset_protocol':'exact evaluation over the complete two-node induced universe',
        'aggregate_metrics':aggregate_metrics,'per_record_subset_metrics':per_record_metrics,
        'interpretation_boundary':'Whole attention/MLP outputs at the final prompt token; no head- or neuron-level claim.'}

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/causal_component_refinement.json'))
    p.add_argument('--causal-config',type=Path,default=Path('configs/causal_patching.json')); p.add_argument('--run-id',required=True)
    p.add_argument('--batch-size',type=int,default=4); a=p.parse_args(); raw=json.loads(a.config.read_text(encoding='utf-8'))
    pre=validate_refinement_prerequisites(Path(raw['corrected_causal_summary']),raw['selected_layers'])
    cfg=load_causal_config(a.causal_config); capture_cfg=_capture_compatible_config(cfg)
    _,tok,_=validate_local_checkpoint(capture_cfg); model=load_model(capture_cfg)
    results={d:run_dataset(model,tok,cfg,d,pre.selected_layers[d],a.batch_size) for d in ('motivation','empathy')}
    report=compact_refinement_report(pre,results); report['run_id']=a.run_id
    out=Path('results/causal_component_refinement')/f'component_refinement_{a.run_id}.json'
    write_compact_refinement_report(out,report); print(out)
if __name__=='__main__': main()

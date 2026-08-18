"""Convert task-separated SAE analyses into the circuit-atlas input contract."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--sae-run-id',required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); datasets={}
    for dataset in ('motivation','empathy'):
        source=Path('results/sae')/dataset/f'sae_analysis_{a.sae_run_id}.json'
        value=json.loads(source.read_text(encoding='utf-8')); layer=int(value['layer'])
        features=[]
        for aggregation,section in value['aggregations'].items():
            for row in section['features'][:50]:
                features.append({'layer':layer,'feature_id':int(row['feature_id']),
                    'score':abs(float(row['standardized_mean_difference'])),'aggregation':aggregation,
                    'rank':int(row['rank'])})
        datasets[dataset]={'status':'complete','layer_count':32,'selected_layers':[layer],
            'layer_scores':{str(layer):max(row['score'] for row in features)},
            'selection_rule':'Pinned Llama Scope SAE layer supported by corrected causal evidence in both tasks',
            'selected_features':features,'source':str(source)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'summary_kind':'sae_circuit_atlas_input','sae_run_id':a.sae_run_id,'datasets':datasets},indent=2),encoding='utf-8')
    print(a.output)
if __name__=='__main__': main()

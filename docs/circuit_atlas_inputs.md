# Circuit-atlas input contract

`run_circuit_atlas.py` consumes completed JSON summaries and never infers a
missing result. The existing linear-probe and combined causal-patching summary
formats are supported directly. A future SAE summary should use this compact
shape:

```json
{
  "datasets": {
    "motivation": {
      "status": "complete",
      "layer_count": 32,
      "selected_layers": [12, 15],
      "layer_scores": {"12": 0.72, "15": 0.81},
      "selection_rule": "prespecified rule",
      "selected_features": [
        {"layer": 15, "feature_id": 123, "score": 0.81}
      ]
    },
    "empathy": {}
  }
}
```

Layer scores may be a layer-keyed object or a full list indexed by layer.
Spearman correlation is emitted only when both methods provide at least two
shared, finite, nonconstant layer rankings. Jaccard is calculated only over the
shared layer coordinate. In particular, probe depth 32 is reported as excluded
when compared with the 32 residual-pre locations numbered 0 through 31.

SAE feature IDs are retained as provenance but are not compared with layer IDs.
An incomplete causal result (fewer than the configured 200 records), missing SAE
summary, empty selection union, or incompatible ranking is explicitly marked
unavailable in both JSON and Markdown outputs.

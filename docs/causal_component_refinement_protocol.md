# Post-layer causal component refinement

This phase may run only after a JSON summary explicitly declares
`"summary_kind": "corrected_causal_patching"`, contains exactly 200 motivation
and 200 empathy records, and the caller supplies a non-empty selected-layer list
for each dataset. The code refuses generic, partial, or unmarked summaries.

## Induced nodes

For each selected Llama layer, two nodes are induced:

- the whole self-attention output vector at the controlled response position;
- the whole MLP output vector at the controlled response position.

Restoration injects the clean node value into the corrupted pass. Knockout injects
the corrupted node value into the clean pass. The source cache is always explicit;
zero ablation is not silently substituted. Candidate nodes are ranked by mean
per-record normalized recovery. A semantic subset takes the top positive nodes, a
greedy subset uses measured forward-selection gains, and seeded size-matched random
subsets provide the baseline.

Faithfulness is candidate-subset normalized recovery. Completeness is one minus
the complement subset's normalized recovery. Minimality is the faithfulness loss
when each candidate node is removed in turn. Values are not clipped to `[0, 1]`,
so overshoot and sign reversal remain visible.

## Interpretation and computational limits

Completeness is only relative to the induced layer-by-component universe. It does
not establish completeness over unselected layers, token positions, attention
heads, MLP neurons, residual pathways, or SAE features. These nodes are causal
intervention units chosen for tractability, not uniquely identified biological-style
components.

Every subset needs additional model forwards. Exhaustive search grows as `2^N`
and is intentionally unsupported; semantic, greedy, and random size-matched subsets
bound the computation. Whole-vector caches also scale with batch size, sequence
length, hidden width, selected layers, and two component families. Use small batches
and checkpoint aggregate JSON rather than retaining activation tensors.

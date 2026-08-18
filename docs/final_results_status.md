# Final results and reproducibility status

The approved 200 motivation + 200 empathy mechanistic-analysis workflow is complete.
The two tasks were sampled and analysed independently throughout.

## Reproducibility IDs

- activation capture and linear probing: `20260817T181608Z`
- corrected causal patching: `20260818T030736Z`
- SAE analysis: `20260818T111200Z`
- component refinement: `20260818T112500Z`
- circuit atlas: `20260818T132800Z`

## Main outputs

- `results/probes/metadata/linear_probe_summary_20260817T181608Z.json`
- `results/causal_patching/metadata/causal_patching_summary_20260818T030736Z.json`
- `results/sae/{motivation,empathy}/sae_analysis_20260818T111200Z.json`
- `results/causal_component_refinement/component_refinement_20260818T112500Z.json`
- `results/circuit_atlas/circuit_atlas_20260818T132800Z.{json,md,png}`

## Result boundary

Corrected patching shows strong late-layer recovery and supports layer 16 for both
tasks. At layer 16, motivation ranks the whole MLP output above attention, whereas
empathy ranks the whole attention output above MLP. The atlas finds high overlap
between significant probing depths and causal candidate layers: Jaccard 0.818 for
motivation and 0.806 for empathy. The single available SAE layer overlaps both.

These experiments measure instruction-conditioned first-token behaviour under the
pre-registered `" I" - " Okay"` target-logit contrast. They do not directly measure
the quality of a complete counselling response. Missing-context placeholders affect
6 motivation and 18 empathy examples; 4 motivation prompts were truncated to 128
tokens. One motivation causal record has an undefined normalised ratio because its
clean and corrupted scores are identical. SAE interpretability scoring still
requires blinded human assessment. Component refinement uses whole attention/MLP
vectors at one token and one layer, so it must not be described as a complete
head/neuron circuit.

Large local-only inputs and caches are intentionally excluded from GitHub. The
repository contains code, configuration, compact JSON, charts, reports, and hashes
needed to audit the workflow without duplicating the 8B model or activation tensors.

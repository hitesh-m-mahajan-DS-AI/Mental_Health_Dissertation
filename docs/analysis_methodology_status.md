# Analysis methodology status

This file records the methodology supplied by the researcher. A `null` entry in
`configs/analysis_methodology.json` is an intentional execution block, not a
software default.

## Linear probing

Motivation and empathy remain independent binary tasks. Each uses the complete
200-example random sample at every residual-stream depth. The primary representation
is the final valid response token and the sensitivity representation is the mean of
response-token activations, excluding special tokens. Separate logistic-regression
probes use conversation-grouped 70/15/15 train/validation/test splits. Accuracy and
macro-F1 are reported with shuffled-label and layer-0 baselines, plus a label-
permutation significance test.

The approved probe preset defines layer 0 as the residual stream entering transformer
block 0, uses grouped split seed 42, training-only standardisation, L2 logistic
regression with `lbfgs` and a 5,000-iteration limit, and selects `C` from
`[0.01, 0.1, 1, 10, 100]` using validation macro-F1. Class weighting is disabled so
the natural random samples remain unchanged. The shuffled-label baseline uses 100
repetitions and the significance distribution uses 1,000 development-label
permutations, repeating model selection and refitting while keeping the grouped split
fixed.

This stage completed successfully for both independent tasks and both token
aggregations. Results, charts, permutation arrays, and the final report are stored in
`results/probes/`. Linear probing provides representational evidence only; no layer is
treated as a causal circuit component until patching and SAE evidence are available.

## Causal patching

The declared contrast is supportive versus neutral, using matched clean/corrupted
pairs. The behavioural score is supportive-target logit minus neutral-target logit
at the first generated response position. Faithfulness is:

```text
(logit_diff(patched) - logit_diff(corrupted))
-------------------------------------------------
(logit_diff(clean)   - logit_diff(corrupted))
```

The corrected layer-patching run `20260818T030736Z` completed 200 records for each
task. It patches only the final valid prompt position, uses task-specific
supportive/neutral instructions, and reports the fixed `" I" - " Okay"` first-token
logit contrast. One motivation record has a zero clean-minus-corrupted denominator
and is explicitly excluded from normalised faithfulness aggregation.

Layer 16 was retained for tractable component refinement because both corrected
causal confidence intervals exclude zero there and the pinned SAE is available at
that layer. Whole attention and MLP outputs were evaluated independently. This is
an induced component universe, not a head-, neuron-, position-, or feature-complete
circuit; no broader completeness claim is made.

## SAE and circuit atlas

The pinned Llama Scope `l16r_8x` residual-stream SAE (revision
`8dbc1d85edfced43081c03c38b05514dbab1368b`) was run through SAELens at layer 16.
Both tasks have 200 sparse records, ranked feature IDs, activation frequency/L0,
bootstrap intervals, and 1,000 matched random-neuron baseline samples.
Interpretability scores remain explicitly null pending blinded human review of top
activating examples; they are not silently inferred from feature magnitude.

The final atlas cross-references probing, patching, and SAE evidence. It reports
Jaccard overlap, Spearman correlation only where rankings are genuinely comparable,
and all disagreements. The completed atlas uses shared layer coordinates only and
keeps SAE feature IDs as provenance. Probe–patching Jaccard is 0.818 (motivation)
and 0.806 (empathy). Spearman is unavailable rather than fabricated because the
required shared ranked coordinates are absent.

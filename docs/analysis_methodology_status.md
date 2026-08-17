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

Faithfulness, completeness, and minimality are all required. Pair construction,
the exact supportive and neutral target token sequences, multi-token scoring (if
needed), and operational completeness/minimality definitions remain unresolved.

## SAE and circuit atlas

SAELens with a compatible Llama Scope residual-stream SAE is preferred for the
local Llama-3.1-8B Base model. The layer is selected only after probing and patching
evidence. Feature ID, sparsity/L0, interpretability score, and a random-neuron
baseline are required.

The final atlas cross-references probing, patching, and SAE evidence. It reports
Jaccard overlap, Spearman correlation only where rankings are genuinely comparable,
and all disagreements. The exact compatible SAE checkpoint and comparison-set
definitions remain to be established.

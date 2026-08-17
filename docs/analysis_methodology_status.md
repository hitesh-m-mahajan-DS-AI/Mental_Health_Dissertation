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

The run is blocked until the exact layer-0 definition, grouped split seed,
standardisation/regularisation choices, solver settings, and shuffle/permutation
procedures are confirmed. These choices affect the scientific result and are not
silently defaulted.

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

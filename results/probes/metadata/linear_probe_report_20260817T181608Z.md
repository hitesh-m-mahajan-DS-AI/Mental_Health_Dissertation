# Linear probing report

- Activation run: `20260817T181608Z`
- Tasks: motivation and empathy analysed independently
- Examples: 200 per task, uniformly sampled without replacement
- Sequence length: 128
- Split: grouped 70/15/15, seed 42
- Probe: training-standardised L2 logistic regression, validation-selected C
- Baselines: layer 0 and 100 shuffled-label repetitions
- Significance: 1,000 full development-label permutations

## Split audit

- Motivation: 140/30/30 examples; negatives 12/2/2; 52/11/14 transcript groups.
- Empathy: 140/30/30 examples; labels 92/48, 20/10, and 20/10; 132/28/27 dialogue groups.
- No transcript or dialogue appears in more than one split.

## Motivation

The final-token probe rises from layer-0 macro-F1
`0.483` to a descriptive
peak of `0.825`
at depth 12.
Depths [12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
pass both the shuffled-label/max-statistic and layer-0-improvement criteria.

The mean-token sensitivity probe has a strong layer-0 baseline of
`0.825` and a
descriptive peak of
`0.891` at
depth 15. Only depth
[15]
passes both criteria.

The motivation test set contains only two non-motivational responses because the approved
random 200-example sample is highly imbalanced. Its accuracy and macro-F1 therefore have
coarse resolution and must not be presented as a precise population estimate. No resampling,
class weighting, or label alteration was applied.

## Empathy

The final-token probe rises from layer-0 macro-F1
`0.400` to a descriptive peak
of `0.961` at depth
12. Every transformer-block
output depth 1-32 passes both permutation criteria.

The mean-token sensitivity probe is already strong at layer 0
(`0.841`) and reaches a
descriptive peak of
`0.961` at depth
8. It beats shuffled
labels, but no depth significantly improves over the strong layer-0 baseline under the declared
permutation criterion.

## Interpretation boundary

These probes show that motivation and empathy labels are linearly decodable from residual-
stream representations. They do not establish that the model causally uses those
representations. Candidate layers for the circuit atlas must also be supported by activation
patching and compatible SAE evidence. Descriptive test peaks are reported for orientation;
they are not treated as causal circuit selections.

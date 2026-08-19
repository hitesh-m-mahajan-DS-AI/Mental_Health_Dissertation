# Supervisor-ready project explanation

This dissertation treats motivation and empathy as two separate binary concepts.
They are never merged into a four-class task.

## Simple explanation

I have two labelled conversation datasets. The motivation dataset labels therapist
responses as motivational or non-motivational. The empathy dataset labels counselor
responses as empathetic or non-empathetic.

For the current experiment, the program selects 200 eligible responses uniformly at
random from each dataset. It does not force equal class counts or change any labels.
The exact random selections are recorded so the experiment can be reproduced.

Each selected response is tokenized to a maximum sequence length of 128 tokens and
passed through the locally stored Llama-3.1-8B model. A response is not separately fed into every layer. It passes
through the model once, while hooks record the required internal activations at all
32 transformer layers during that forward pass.

For every layer, the capture includes:

- the residual stream before and after the layer;
- attention query, key, and value projections;
- attention-head outputs before the output projection;
- the combined attention output;
- the gated MLP activation; and
- the MLP output.

The motivation and empathy activations are stored independently:

```text
results/activations/motivation/
results/activations/empathy/
```

Every stored tensor is linked to its dataset, source row, conversation, utterance,
binary label, layer, activation type, token position, model configuration, and run
identifier. Files are checksummed and reloaded after capture to verify their shapes
and values.

## Current completed result

The methodology-corrected 128-token run `20260817T181608Z` captured:

- 200 random motivation responses: 184 motivational and 16 non-motivational,
  covering 77 transcripts;
- 200 random empathy responses: 68 empathetic and 132 non-empathetic, covering
  187 dialogues;
- 115,200 activation tensors across the two independent experiments; and
- 15,732,572,160 verified finite activation values, occupying 29.31 GiB.

The selected empathy responses all fit within 128 tokens. Two selected motivation
responses were longer than 128 tokens and were explicitly truncated at the declared
sequence length. The capture manifest records those cases.

## Completed linear probing

The stored activations were used for two independent sets of linear probes:
one for motivation and one for empathy. A separate probe was trained for every
selected layer, using the same examples at each layer. Conversation-grouped 70/15/15
train/validation/test splits will prevent responses from the same transcript or
dialogue appearing in more than one split. The primary probe uses the final valid
response token; a sensitivity analysis uses the mean response-token activation.
Accuracy and macro-F1 will be reported alongside shuffled-label and layer-0
baselines and a label-permutation significance test. Probing shows where information
is represented; it does not prove that the model causally uses that information.

For motivation, the primary final-token probe rises from layer-0 macro-F1 0.483 to a
descriptive peak of 0.825 at depth 16. Later depths pass both the shuffled-label and
layer-0 permutation criteria. The mean-token sensitivity probe starts from a much
stronger layer-0 macro-F1 of 0.825 and peaks descriptively at 0.891; only depth 15
significantly improves over that strong baseline. The motivation test split contains
only two non-motivational responses because the approved random sample is naturally
imbalanced, so these scores have coarse resolution and must be interpreted cautiously.

For empathy, the primary final-token probe rises from layer-0 macro-F1 0.400 to a
descriptive peak of 0.961 at depth 16, with every transformer-block output depth 1-32
passing both permutation criteria. The mean-token sensitivity probe is already strong
at layer 0 (macro-F1 0.841) and peaks descriptively at 0.961; it beats shuffled labels
but does not significantly improve over its strong layer-0 baseline.

These findings establish linear decodability, not causality. Descriptive peak layers
are not yet final circuit selections.

## Later analysis stages

Activation patching will then rerun reproducible clean/corrupted response pairs and
intervene on candidate residual, attention, and MLP components. This tests causal
use and reports normalized logit-difference recovery, faithfulness, completeness,
and minimality.

Compatible sparse autoencoders will be applied to appropriate Llama-3.1-8B
activations to identify sparse motivation- and empathy-associated features. Feature
IDs, sparsity, top activating examples, interpretability scores, and random-feature
baselines will be recorded.

Finally, the circuit atlas will combine probing evidence, causal patching evidence,
and SAE features. JSON result files will support reproducibility, while charts and a
visual circuit diagram will summarize relevant layers, components, features, method
agreement, and disagreement. Jaccard overlap and, where rankings are genuinely
comparable, Spearman correlation will quantify cross-method agreement.

## Important status boundary

The 128-token 200+200 activation-capture and linear-probing stages are complete.
Activation patching, SAE analysis, cross-method agreement, and the final circuit atlas
remain subsequent stages. Their results must not be claimed until those experiments
have been implemented and run. Patching pair construction and target tokens,
completeness/minimality definitions, and a compatible SAE checkpoint still require
explicit methodological decisions.

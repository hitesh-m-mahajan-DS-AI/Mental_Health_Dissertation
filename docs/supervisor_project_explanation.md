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

## Later analysis stages

The stored activations provide the input for two independent sets of linear probes:
one for motivation and one for empathy. A separate probe will be trained for every
selected layer, using the same examples at each layer. Conversation-grouped 70/15/15
train/validation/test splits will prevent responses from the same transcript or
dialogue appearing in more than one split. The primary probe uses the final valid
response token; a sensitivity analysis uses the mean response-token activation.
Accuracy and macro-F1 will be reported alongside shuffled-label and layer-0
baselines and a label-permutation significance test. Probing shows where information
is represented; it does not prove that the model causally uses that information.

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

The 128-token 200+200 activation-capture stage is complete. Linear probing, activation
patching, SAE analysis, cross-method agreement, and the final circuit atlas are
subsequent stages. Their results must not be claimed until those experiments have
been implemented and run. Exact probe fitting/permutation settings, patching pair
construction and target tokens, completeness/minimality definitions, and a
compatible SAE checkpoint still require explicit methodological decisions.

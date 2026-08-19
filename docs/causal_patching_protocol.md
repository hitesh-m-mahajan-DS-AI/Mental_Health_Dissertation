# Causal activation-patching protocol

## Scope

This phase uses the exact local Meta Llama-3.1-8B Base checkpoint and the frozen random
selection from activation run `20260817T181608Z`: 200 motivation observations and 200 empathy
observations. The tasks remain separate.

## Controlled counterfactual

For each selected observation, the client context is held fixed and two equal-structure,
task-specific prompts are constructed:

- motivation clean/corrupted style: `motivational` / `neutral`
- empathy clean/corrupted style: `compassionate` / `neutral`

Both end with the same client context and `Counsellor:` prefix. Samples without an available
preceding client utterance are retained using an explicit missing-context sentence; this preserves
the approved random sample rather than silently replacing observations.

The behavioural score at the first generated response position is

`logit(" I") - logit(" Okay")`.

Both targets are single tokens under the checkpoint tokenizer. This fixed verbalizer makes the
score comparable across all pairs. It is an operational measure of instruction-conditioned
supportive opening behaviour, not a clinical assessment of a complete generated response.

## Layerwise intervention

The clean residual-stream input to each transformer block is cached. The paired prompts are
tokenizer-validated to differ at exactly one controlled style token while retaining aligned token
positions. For each layer in turn, the primary intervention restores only the clean residual vector
at the final valid prompt token (the `Counsellor:` position whose logits predict the first response
token) into the corrupted forward pass. All other positions remain corrupted and all later
computation is recomputed. Instruction-token restoration is reserved as a candidate-layer
sensitivity analysis. The primary normalized recovery is

`(patched logit difference - corrupted logit difference) /
 (clean logit difference - corrupted logit difference)`.

Pairs whose denominator has absolute magnitude at most `1e-6` are reported as undefined and
excluded from normalized aggregate estimates; they are never coerced to zero.

Layer means and 95% paired bootstrap intervals use 1,000 resamples with seed 42. Candidate layers
must have a positive mean raw patch effect and a normalized-recovery interval excluding zero.

## Completeness and minimality

Wang et al. define completeness and minimality on nodes of an induced computational circuit.
Residual-layer restoration identifies causal locations but does not, by itself, define such an
induced circuit. Consequently, the exact knockout tests will be run after causal layers are refined
into attention/MLP or SAE-feature nodes. Reporting a residual-layer subset as if it were a complete
circuit would overstate the evidence.

## Reproducibility

Pair manifests contain source identifiers, prompt hashes, token lengths, target IDs, truncation,
and missing-context status. Per-pair results are written atomically at the end of each batch and the
runner skips completed records when resumed.

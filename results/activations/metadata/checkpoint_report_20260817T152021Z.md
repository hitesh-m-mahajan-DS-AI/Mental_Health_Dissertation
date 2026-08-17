# Activation capture Checkpoints A–D report

Run `20260817T152021Z` completed successfully using the exact local Llama-3.1-8B
checkpoint. No quantization, alternate model, download, relabelling, deduplication,
or source-file modification was performed.

## Model and environment

- Model path: `U:\Mental_Health_Dissertation\model`
- Architecture: `LlamaForCausalLM`, base `meta-llama/Meta-Llama-3.1-8B`
- Weights: four local safetensors shards, 16,060,522,496 bytes indexed
- Revision: unavailable; no local VCS/Hugging Face revision metadata was present
- Configuration: 32 layers; width 4096; MLP width 14336; 32 query heads;
  8 KV heads; vocabulary 128256; Llama-3 rope scaling; configured bfloat16
- Tokenizer: local `PreTrainedTokenizerFast`; base vocabulary 128000, total 128256;
  BOS 128000; EOS 128009; no pad token
- Device: NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB; bfloat16 supported;
  driver 610.88
- Memory at run start: 7,035 MiB GPU free; 2,330,411,008 bytes system RAM
  available of 16,415,322,112 bytes total
- Placement: layers 0–10 on GPU, layers 11–12 on CPU, layers 13–31 plus
  final norm/lm head disk-offloaded. All remained bfloat16.

Software: Python 3.14.3; PyTorch 2.11.0+cu128; Transformers 4.57.6;
TransformerLens 3.5.1; Accelerate 1.14.0; safetensors 0.8.0;
scikit-learn 1.8.0; pandas 2.3.3; NumPy 2.3.4. SAELens is not installed.

TransformerLens successfully converted the local configuration as
`Llama-3.1-8B` with the same dimensions. Direct Hugging Face/PyTorch hooks were
used to avoid constructing a second converted 8B weight copy under the available
memory constraints.

## Dataset audit

- Motivation: 4,882 source rows. The binary scientific population is 2,246 rows
  (2,096 motivational, 150 non-motivational); 2,636 `review` rows are excluded.
  There are 131 binary-population transcripts. Group key: `transcript_id`.
- Empathy: 27,844 source rows. The counselor-response scientific population is
  14,408 rows (10,091 non-empathetic, 4,317 empathetic), across 1,004 dialogues.
  Eighteen `counselor ` values were included through logged whitespace trimming;
  two labeled rows with missing roles were excluded rather than inferred. Group
  key: `dialogueId`.
- Duplicates were retained. The audits record six duplicate motivation response
  texts, 10,817 duplicate empathy response texts, and four duplicate empathy
  `(dialogueId, utteranceNo)` keys in the retained counselor population.

Source SHA-256:

- Motivation: `94c2233997f0a493e9d129dfd98599a9ef49fb61106569b2b93ff0ea6a6c0f05`
- Empathy: `c26d6a2ea0311a0bd59f80d9b4ff4b1c2599d57b2560f330949db235030a702c`

## Hook and tensor validation

The runner found 288 hook points: nine activation types at each of 32 layers.
It captured one source-ordered example from each dataset, every token, with a
64-token safety cap.

- Motivation sample: 44 tokens, 288 tensors, 115,370,312 bytes
- Empathy sample: 20 tokens, 288 tensors, 52,455,536 bytes
- Residual pre/post and attention/MLP outputs: `[tokens, 4096]`
- Query and pre-output head concatenation: `[tokens, 4096]`; the latter is also
  stored as `[tokens, 32, 128]` for individual-head access
- Key/value: `[tokens, 1024]` (8 KV heads × 128)
- Gated MLP activation: `[tokens, 14336]`

All six output files passed SHA-256 checking and safetensors reload validation;
all 576 tensor keys and shapes matched their manifests. All 83,886,080 activation
values were finite, and no tensor was entirely zero.

## Capacity boundary before Checkpoint E

The configured comprehensive all-token payload costs 2,621,440 bytes per token.
At `max_length=64`, the full binary populations would require approximately
140.93 GiB for motivation and 1,003.07 GiB for empathy—1,144.00 GiB total before
metadata/filesystem overhead. Drive U had about 332.93 GiB free. The complete run
therefore must not start under this configuration.

This is not resolved by silently changing token aggregation. The final prompt/input
construction and token-position or aggregation policy require scientific approval;
alternatively, substantially more output storage is required.

## Outputs

- Motivation manifest: `U:\Mental_Health_Dissertation\results\activations\motivation\manifest_20260817T152021Z.json`
- Empathy manifest: `U:\Mental_Health_Dissertation\results\activations\empathy\manifest_20260817T152021Z.json`
- Run metadata: `U:\Mental_Health_Dissertation\results\activations\metadata\run_20260817T152021Z.json`
- Storage estimate: `U:\Mental_Health_Dissertation\results\activations\metadata\storage_estimate_20260817T152021Z.json`
- Tensor validation: `U:\Mental_Health_Dissertation\results\activations\metadata\tensor_validation_20260817T152021Z.json`
- Motivation population IDs: `U:\Mental_Health_Dissertation\results\activations\motivation\metadata\population_manifest.jsonl`
- Empathy population IDs: `U:\Mental_Health_Dissertation\results\activations\empathy\metadata\population_manifest.jsonl`

## Unresolved scientific choices

The smoke run used the raw response with BOS and all token positions to validate
mechanics only. It does not approve the final prompt construction, token aggregation,
grouped train/validation/test ratios, causal corruption construction, behavioural
logit definition, or SAE checkpoint/source. Checkpoint E and later scientific stages
remain intentionally unstarted.

The handover also names `AnnoMI-full.csv`, `annomi_binary_corrected.csv`, and
`annomi_manual_review_corrected.csv` as provenance/reference files. None is present
in the current workspace; this does not block Checkpoints A–D but must be resolved
before a complete provenance audit.

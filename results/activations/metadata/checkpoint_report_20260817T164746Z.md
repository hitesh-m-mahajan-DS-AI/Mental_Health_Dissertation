# Activation capture checkpoint report

- Run ID: `20260817T164746Z`
- Mode: `configured capture`
- Resumed after interruption: `True`
- Model: `U:\Mental_Health_Dissertation\model` (`LlamaForCausalLM`)
- Shape: 32 layers, hidden size 4096, 32 attention heads, 8 KV heads
- Configured dtype: `bfloat16`
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- Instrumentation: direct Hugging Face module hooks; TransformerLens config compatibility is `True`
- Hook points discovered: 288

## Random selections

- motivation: 200 examples; labels {'0': 16, '1': 184}; 77 unique conversations; seed 42
- empathy: 200 examples; labels {'0': 132, '1': 68}; 187 unique conversations; seed 43

## Capture and verification

- Estimated tensor payload: 27.30 GiB
- motivation: 200 examples; 600 files; 12.86 GiB; 57600 verified tensors
  - Manifest: `U:\Mental_Health_Dissertation\results\activations\motivation\manifest_20260817T164746Z.json`
- empathy: 200 examples; 600 files; 14.45 GiB; 57600 verified tensors
  - Manifest: `U:\Mental_Health_Dissertation\results\activations\empathy\manifest_20260817T164746Z.json`

## Methodological boundary

This run captures the configured raw response with BOS and every token up to 64. It does not decide the later grouped train/validation/test ratios, causal corruption construction, behavioural logit definition, or SAE checkpoint/source.

# Activation capture checkpoint report

- Run ID: `20260817T181608Z`
- Mode: `configured capture`
- Resumed after interruption: `False`
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

- Estimated tensor payload: 29.30 GiB
- motivation: 200 examples; 600 files; 14.48 GiB; 57600 verified tensors
  - Manifest: `U:\Mental_Health_Dissertation\results\activations\motivation\manifest_20260817T181608Z.json`
- empathy: 200 examples; 600 files; 14.84 GiB; 57600 verified tensors
  - Manifest: `U:\Mental_Health_Dissertation\results\activations\empathy\manifest_20260817T181608Z.json`

## Methodological boundary

This run captures the configured raw response with BOS and every token up to 128. The downstream analyses use their separately declared aggregation and experimental settings.

# Sparse-autoencoder analysis protocol

The SAE phase uses the pinned Llama Scope residual-stream SAE for Llama-3.1-8B
Base layer 16 (`l16r_8x`, revision
`8dbc1d85edfced43081c03c38b05514dbab1368b`). It is loaded only from the local
`.hf-sae` folder through SAELens with offline mode enabled.

Motivation and empathy are encoded, analysed, checkpointed, and reported
independently. For each captured `layer_16.residual_post` sequence, the encoder
processes small token batches. Dense token-by-feature tensors are discarded
immediately. A resumable JSONL cache retains only non-zero feature IDs and
values for the final token and token mean, plus sparse activation-count data.

Feature selection uses only the grouped training split. Examples are first
collapsed into equally weighted conversation-label cells, then ranked by the
absolute pooled standardized mean difference between labels. Uncertainty is a
1,000-resample conversation bootstrap. Each selected set is compared with
1,000 same-size samples of raw residual neurons from the same layer and
aggregation. Outputs include compact JSON, a chart, top activating example IDs,
feature activation frequency, and mean token L0. Interpretability scores remain
explicitly null until blinded human assessment of those examples is performed.

Example commands (using the dedicated SAE environment):

```powershell
.\.venv-sae\Scripts\python.exe -m src.sae_analysis --run-id <RUN_ID> --phase encode
.\.venv-sae\Scripts\python.exe -m src.sae_analysis --run-id <RUN_ID> --phase analyze
```

The encode phase resumes by skipping example IDs already present in each
task's cache. The analysis phase requires all 200 records and never downloads
weights.

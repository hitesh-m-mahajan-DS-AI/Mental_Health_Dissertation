# Mental-health Llama-3.1-8B mechanistic pipeline

The first implemented stage is the single-entry-point activation capture workflow:

```powershell
python run_activation_capture.py
```

It validates and loads the local `model/` checkpoint only, processes motivation and
empathy independently, captures all configured internal activation types, writes
chunk-safe `safetensors` files, and verifies every saved tensor by checksum and reload.

The checked-in configuration is intentionally a one-example-per-dataset smoke test.
It must remain a smoke test until the final input construction and token-position
policy are approved. `--inspect-only` performs the full environment/dataset/hook
inspection without loading the 8B weights.

The full-population payload estimate for the current all-token, nine-activation
configuration is about 1.144 TiB at 64 tokens, so it cannot fit on the currently
available workspace drive. Do not remove the smoke limit as a storage workaround;
the scientific token policy must be decided explicitly first.

Outputs are separated under `results/activations/motivation/` and
`results/activations/empathy/`. Original CSV and model files are read-only inputs.

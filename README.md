# Mental-health Llama-3.1-8B mechanistic pipeline

The first implemented stage is the single-entry-point activation capture workflow:

```powershell
python run_activation_capture.py
```

It validates and loads the local `model/` checkpoint only, processes motivation and
empathy independently, captures all configured internal activation types, writes
chunk-safe `safetensors` files, and verifies every saved tensor by checksum and reload.

The checked-in configuration selects 200 eligible binary examples uniformly at
random without replacement from each dataset. Motivation Review rows are not
eligible; empathy uses counselor responses only. Dataset-specific seeds are
recorded solely to make the random selections exactly reproducible. The sampler
does not force class balance or alter labels. `--inspect-only` performs the full
environment/dataset/hook inspection without loading the 8B weights.

Interrupted captures can be resumed without deleting or recapturing completed
examples:

```powershell
python run_activation_capture.py --resume-run-id <RUN_ID>
```

The configured 200+200 experiment fits the local storage budget while retaining
all tokens up to 64 and all nine activation types at every layer. A full-population
capture would still be about 1.144 TiB and must not be started accidentally.

Outputs are separated under `results/activations/motivation/` and
`results/activations/empathy/`. Original CSV and model files are read-only inputs.

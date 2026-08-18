# Mental-health Llama-3.1-8B mechanistic pipeline

The activation-capture workflow is:

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
all tokens up to 128 and all nine activation types at every layer. A full-population
capture would still be about 1.144 TiB and must not be started accidentally.

Outputs are separated under `results/activations/motivation/` and
`results/activations/empathy/`. Original CSV and model files are read-only inputs.

The supplied probing, causal-patching, SAE, component-refinement, and circuit-atlas
methodology is recorded in `configs/` and `docs/`. The approved 200+200 workflow is
complete; compact results and reports are checked in while the model, datasets,
activation tensors, caches, and SAE checkpoint remain local and ignored.

The approved independent linear-probing stage is now complete for activation run
`20260817T181608Z`:

```powershell
python run_linear_probes.py
python run_probe_reporting.py
```

It uses group-disjoint 70/15/15 splits, final-token primary representations,
mean-response-token sensitivity representations, validation-selected L2 logistic
regression, 100 shuffled-label baselines, and 1,000 full label-permutation pipelines.
Compact JSON, permutation arrays, charts, checksums, and the human-readable report are
stored under `results/probes/`.
## Causal activation patching

The project now includes reproducible activation capture, grouped linear probing, and controlled
causal residual-stream patching for 200 randomly selected motivation examples and 200 randomly
selected empathy examples using the exact local Llama-3.1-8B Base checkpoint.

The causal-patching protocol and its interpretation limits are documented in
`docs/causal_patching_protocol.md`. Run an audit-only preflight with:

```powershell
python run_causal_patching.py --inspect-only
```

The valid corrected run is `20260818T030736Z`. Run or resume it with:

```powershell
python run_causal_patching.py --resume-run-id 20260818T030736Z
```

Run `20260818T020527Z` is retained only as an invalidated diagnostic: it restored
the whole prompt residual tensor and therefore could not localise causal layers.

## Completed downstream analyses

- Corrected causal patching: 200 motivation + 200 empathy records, 32 layers.
- SAE: pinned Llama Scope `l16r_8x` at layer 16, 200+200 sparse encodings,
  50 ranked features per task/aggregation, and 1,000-draw random-neuron baselines.
- Component refinement: whole attention and MLP outputs at layer 16. Motivation
  ranks MLP first; empathy ranks attention first.
- Circuit atlas: probe–patching Jaccard is 0.818 for motivation and 0.806 for
  empathy. SAE overlaps at layer 16. Spearman is explicitly unavailable because
  the probe summary has no complete layer ranking and the SAE covers one layer.

See `docs/final_results_status.md` for artifact locations, reproducibility IDs,
scientific limitations, and the remaining dissertation-writing boundary.

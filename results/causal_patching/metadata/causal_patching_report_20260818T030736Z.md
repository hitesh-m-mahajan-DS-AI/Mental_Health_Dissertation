# Causal activation-patching report

- Run: `20260818T030736Z`
- Activation selection: `20260817T181608Z`
- Samples: 200 motivation + 200 empathy
- Intervention: clean residual-pre restoration into the neutral-instruction run, one layer at a time
- Score: `logit(" I") - logit(" Okay")` at the first generated response position

## Motivation

- Valid denominators: 199/200
- Descriptive peak layer: 22
- Peak mean normalized recovery: 1.0910
- Candidate layers under the prespecified rule: [1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

## Empathy

- Valid denominators: 200/200
- Descriptive peak layer: 30
- Peak mean normalized recovery: 0.9911
- Candidate layers under the prespecified rule: [2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

## Interpretation boundary

This experiment localizes model computation for an explicit supportive-versus-neutral style instruction and fixed opening-token contrast. It does not establish the clinical quality of a full response. Residual-layer locations are not yet an induced circuit; Wang-style completeness and minimality tests follow component/SAE refinement.

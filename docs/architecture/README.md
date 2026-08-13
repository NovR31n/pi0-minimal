# Architecture specification

P1 specification index:

1. `01_system_architecture.md`: end-to-end modules, data flow, and attention
   dependencies.
2. `02_tensor_spec.md`: shapes, dtypes, ranges, padding, masks, and runtime
   assertions.
3. `03_flow_matching.md`: interpolation, target velocity, loss, time sampling,
   Euler inference, and analytical tests.
4. `04_reference_mapping.md`: paper/OpenPI mapping and backend decision.
5. `05_simplification_decisions.md`: explicit deviations, fair-comparison
   boundary, allowed claims, and resource gates.
6. `06_autoregressive.md`: continuous action likelihood, shifted teacher
   forcing, causal generation, and matched-comparison boundary.

The machine-readable primary configurations are
`configs/model_flow_tiny.toml` and `configs/model_ar_tiny.toml`. Validate them
with:

```bash
uv run python scripts/validate_model_spec.py
```

These files are implementation requirements. Any behavior-changing deviation
must update the specification, tests, and experiment registry in the same
commit.

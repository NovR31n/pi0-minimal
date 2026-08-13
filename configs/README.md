# Configurations

Store one versioned configuration per experiment family. Every formal result
must reference the exact configuration file and Git commit used to produce it.

- `model_flow_tiny.toml`: compact flow-matching policy.
- `model_ar_tiny.toml`: matched continuous autoregressive policy.
- `data_libero_spatial.toml`: LIBERO split and normalization protocol.
- `libero_spatial_tasks.toml`: exact LeRobot dataset-index to LIBERO benchmark
  task-ID mapping, frozen by language string.
- `experiments.toml`: immutable experiment registry.

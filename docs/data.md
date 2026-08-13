# Data contract and LIBERO conversion

The model-facing API is NumPy-only and independent of OpenPI. Images enter as
`uint8 [B,V,C,H,W]`; token IDs and masks are explicit; states and fixed-horizon
actions are `float32`.

The source LIBERO-Spatial LeRobot v2.1 dataset stores an 8D proprioceptive
vector: 6D end-effector pose and two signed finger positions. This compact
project deliberately converts it to 7D by retaining the pose and using
`left_finger - right_finger` as the physical opening width. Raw actions are
7D: the first six components use LIBERO's continuous action convention, while
the LeRobot gripper label is binary (`1` open, `0` closed). At the simulator
boundary this is converted to LIBERO's bipolar command (`-1` open, `+1`
closed).

At the end of an episode, action chunks are zero-padded to the configured
horizon and the corresponding scalar validity mask is false. Padded actions
therefore do not contribute to normalization statistics or training loss.

Train/validation splitting happens at the episode level, stratified by task.
The saved split fingerprint is also stored in the normalization cache so that
statistics from a different split cannot be silently presented as the current
training statistics.

`scripts/create_libero_smoke_subset.py` extracts a few real frames and action
chunks into a compressed NPZ. The artifact belongs under ignored `data/` and
is evidence for data smoke tests, not a replacement for the full training
dataset.

The committed `configs/data_libero_spatial.toml` freezes split seed 7,
validation fraction 0.1, the conversion rules, quantiles, and smoke episode
selection. `scripts/prepare_libero_data.py` creates the episode manifest and
fits normalization statistics from training episodes only.

## Official-teacher distillation records

Teacher data is collected through the official OpenPI websocket policy API
without loading OpenPI weights into this project. Each episode directory holds
preprocessed base/wrist images, the compact 7D state, the official 8D state,
raw LIBERO-space teacher chunks, fixed-horizon 7D student targets, executed
actions, query steps, and validity masks. A `_COMPLETE` marker is written last;
resume logic ignores partial directories. Successful teacher episodes are the
primary Student-v1 source, while completed failures are retained for analysis
and infrastructure failures abort the collector without becoming policy
failures.

Official teacher outputs are retained verbatim for audit and official-protocol
execution. Student targets are clipped to LIBERO's `[-1,1]` action domain before
the bipolar gripper command is converted to the binary training convention.

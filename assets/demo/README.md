# Public Student-v3 demo

[`student_v3_demo.mp4`](student_v3_demo.mp4) is the 640 × 640, 15-second H.264
version. [`student_v3_demo.gif`](student_v3_demo.gif) is a 480 × 480 README
preview.

The montage uses four successful episodes from the frozen formal evaluation:

| Panel | Formal episode | Task description |
| --- | --- | --- |
| Top left | task 0 / init 43 | bowl between the plate and ramekin |
| Top right | task 3 / init 11 | bowl on the cookie box |
| Bottom left | task 4 / init 12 | bowl in the cabinet's top drawer |
| Bottom right | task 7 / init 24 | bowl on the stove |

Each source is a complete `success=true`, exception-free rollout from
`STUDENT-V3-SUCCESS39-10K-S7-R5-500-018`. The four clips are retimed to finish
together in the ten-second grid; no control steps are removed. The title and
end cards report the full 319/500 result and identify the footage as LIBERO
simulation. The montage is illustrative and is not itself evaluation evidence.

Raw rollout videos remain in the private artifact store. Their corresponding
`result.json` records are covered by the integrity checks described in the
[formal report](../../reports/2026-08-13_student_v3_formal_500.md).

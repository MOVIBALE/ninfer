# Research notes

Records for implementation candidates that were built, measured, and then **not** routed. An entry
exists so a later round can see what was already tried and on what evidence, instead of rebuilding the
same candidate. Every number here is a complete public Op measurement (median, GPU-side, captured CUDA
graph, one point per process, the two builds alternating inside one window) unless the line says
otherwise. The reports under `docs/performance/` own published results; this file does not replace them.

## Q5 row-block small-T shape (`ops/linear/q5/q5_rowsplit_rowblock_small_t.cuh`, removed)

The shape staged one activation slab per block in shared memory and let `kRowsPerBlock` warps read it, so
one warp still owned one output row but the repeated activation loads were divided by the row count. It was
introduced to serve the Q5 parent of the two Q4/Q5 input projections at `T=7..9`, where a warp-per-row
kernel re-reads the activation tile once per row. It is not routed at any column count, and the file was
removed from the production tree.

What decided it:

| Measurement | row-block | routed shape | reading |
|---|---|---|---|
| GDN Q5 parent alone, `T=7/8/9` (us) | 62.7 / 62.7 / 79.1 | 52.5 / 60.6 / 70.9 (split4) | row-block loses at every count |
| GDN complete Snapshot, `T=10/12` (us) | 91.4 / 102.3 | 81.3 / 82.2 (c4 SIMT tile) | row-block loses where it was hoped to win |
| attention complete projection, `T=7/8/9` | - | - | split4 is 18.8% / 14.6% / 5.4% faster than what was routed (row-block at 7/8, c4 at 9) |

The `T=10/12` row is the important one: a **parent-only** probe ranked the row-block *ahead* of the c4 tile
at those two counts, while the complete Op, the projection op's own benchmark, and kernel-level attribution
all ranked it behind. Parent-only timing is therefore not used to move a route boundary in these Ops. The
`T=7/8/9` rows also record that the comparison that first chose the row-block was row-block vs row-split
SIMT, and the split4 shape that is routed now had not been in it.

The shape's mechanism was correct, including the `__syncwarp()` fence added to its pipeline. Both the
mechanism and that fix remain available in git history.

## Q5 split4 band ends, per parent

The split4 shape (one warp per output row, four warps splitting K, exact compile-time column count) is the
Q5 parent mechanism for the low column counts of both fused projections. Its band ends where the c4
narrow-column SIMT tile takes over, and the two parents end at different counts because their row counts
differ (12288 against 7168):

- GDN parent: split4 through `T=10`. At `T=10` split4 beats the c4 tile in the complete Op for every
  organisation that exposes 10 aggregate columns - `B=1/W=10` (105.7 against 110.3 us cold), `B=2/W=5`
  (104.9 against 109.7) and `B=5/W=2` (104.2 against 108.1), in both forms and under both cache policies,
  and by more warm (about 93 against 104 us). At `T=11` and `T=12` it loses (118.0 against 111.9 and 140.5
  against 113.9 us cold), so the band ends at 10.
- attention parent: split4 through `T=9`. `T=10` ties with the c4 tile (77.06 us both) and `T=11`/`T=12`
  lose (85.0 against 79.1 and 95.5 against 81.2 us), so the band ends at 9.

Both ends are crossovers between two legal shapes, not limits of either shape: both are correct at every
count in `[2,15]` for the GDN parent and `[2,12]` for the attention parent.

## Fused projection + convolution for the Q4/Q5 GDN input projections

The op either projects into a workspace and then runs the convolution as its own step, or fuses the
convolution into the projection epilogue (which is what the resolutions `T=1,2,3,5,6` already do). Three
attempts to widen the fused route were built and measured; all three were slower, so the materialized route
stays for the rest:

| Candidate | routed form (us, cold) | candidate (us, cold) | note |
|---|---|---|---|
| `T=4`, `B=1`, Snapshot / Record | 52.2 / 50.9 | 54.5 / 53.0 | 5 -> 4 graph nodes, 81920 -> 0 bytes of workspace |
| `T=7`, `B=1`, Snapshot | 69.4 | 80.9 | Q4 side becomes the row-split SIMT kernel |
| `T=8`, `B=1`, Snapshot | 76.3 | 91.4 | same |
| aggregate 8, `B=2/W=4`, Snapshot | 75.0 | 95.5 | batched fused organisation |
| aggregate 8, `B=8/W=1`, Snapshot | 73.7 | 105.7 | same |

The `T=4` candidate is the cleanest illustration: it is structurally simpler (no workspace, one fewer node)
and still loses, so the loss is not about the number of launches. It comes from which kernels the fused
template can reach - the row-split SIMT Q4 kernel rather than the K-split MMA kernel - and from running the
convolution inside the projection epilogue, where the history chain costs the projection kernel rather than
a separate one that can be shaped for it. The batched variant at aggregate 8 (one launch per side over the
flattened `(request, token)` column axis, with the convolution epilogue reading each token's request index
from that axis) reuses exactly those shapes and loses by the same mechanism.

Not implemented: fusing the K-split MMA Q4 side and the Q5 split4 side with a sequence-collecting
convolution. Every measurement above says the loss is concentrated there, so that variant is the one that
could still be interesting; it needs a genuinely different kernel rather than a new instantiation of the
existing template, and it was not built.

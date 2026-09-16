# Connectome Graph Report (Phase 1 / Gate G1)

## Neuron and edge counts
- Proofread root IDs (raw, `proofread_root_ids_783.npy`): 139,255
- Proofread neurons with an annotation row (final node set): 139,241
- Nodes in graph.npz: 139,241
- Edges (after `min_syn=5` threshold, summed across neuropils): 2,700,429
- Type pairs: 410,768
- Descending neurons (DN): 1,303

## Input layer choice
- Photoreceptor-matched neurons (R1-6/R7/R8, both sides): 11,118
- Lamina-matched neurons (L1-L3, both sides): 4,730
- **Chosen input layer: photoreceptor** (11,118 neurons; gate floor is 500).
- T4/T5 neurons matched: 12,246
- Photoreceptor counts per cell_type per side:
  - R1-6_left: 4,423
  - R1-6_right: 4,029
  - R7_left: 672
  - R7_right: 670
  - R8_left: 670
  - R8_right: 654

## Photoreceptor positions: soma vs. anchor (D15)
- All neurons: 118,100 from `soma_x/y/z`, 21,141 fell back to the `pos_x/y/z` anchor point, 0 had neither.
- Photoreceptors only: 10 from soma, **11,108 from the anchor point** (expected: photoreceptor somata sit in the retina, outside the FAFB volume, so `soma_x/y/z` is null for almost all of them and `pos_x/y/z` is used instead), 0 had neither.

## Cell-type assignment (D14: cell_type -> hemibrain_type -> cell_class -> singleton)
- cell_type: 137,714 neurons
- hemibrain_type: 2 neurons
- cell_class: 1,358 neurons
- singleton: 167 neurons
- Total distinct types: 9,028

## Neurotransmitter / edge sign assignment (D11-D13)
- Confidence threshold for using the predicted `top_nt`: 0.5
- Predicted `top_nt` used directly: 114,935 neurons
- Fallback (synapse-weighted mean of outgoing edges) used: 23,387 neurons
- No data available for either predicted or fallback NT (defaulted to ACh/+1): 919 neurons
- Histamine override applied (photoreceptors, D12): 11,118 neurons
  - What `top_nt` originally predicted for these neurons before the override:
    - ach: 6,958
    - ser: 377
    - glut: 2,863
    - gaba: 769
    - oct: 116
    - da: 35

- Per-transmitter neuron counts (post-override, sign in parentheses):
  - his (-1): 11,118
  - glut (-1): 21,994
  - da (+1): 5,487
  - oct (+1): 117
  - ser (+1): 1,877
  - ach (+1): 80,318
  - gaba (-1): 18,330

- Per-transmitter edge counts (D13):
  - ach (+1): 1,569,262
  - da (+1): 43,940
  - gaba (-1): 615,741
  - glut (-1): 429,291
  - his (-1): 18,657
  - oct (+1): 5,166
  - ser (+1): 18,372

## Positions (D15)
- `pos_voxel`: raw FlyWire soma/anchor voxel coordinates, resolution 4x4x40 nm/voxel
- `pos_nm`: pos_voxel scaled by (4, 4, 40) to nanometers

## Edge sign share (post-alias-fix)
- sign -1 (inhibitory): 1,063,689 / 2,700,429 (39.39%)
- sign +1 (excitatory): 1,636,740 / 2,700,429 (60.61%)

## Type-pair size distribution (for D16 gain-sharing)
| min edges per pair | pairs at/above | share of pairs | edges covered | share of edges |
|---|---|---|---|---|
| 1 | 410,768 | 100.00% | 2,700,429 | 100.00% |
| 2 | 244,824 | 59.60% | 2,534,485 | 93.85% |
| 5 | 55,770 | 13.58% | 2,032,709 | 75.27% |
| 10 | 19,599 | 4.77% | 1,800,715 | 66.68% |
| 20 | 9,347 | 2.28% | 1,667,719 | 61.76% |

- D16 hybrid: smallest-parameter-count k with fine-pair edge coverage >= 90% is **k=2** (coverage 93.85%; k=3 would drop to 85.88%).
- Hybrid (k=2) gain parameter count: 244,824 fine pairs + 18,925 coarse `(pre_type, post_super_class)` buckets = **263,749 total**.
- For reference: pure fine (pre_type, post_type) = 410,768 params; pure coarse (pre_type, post_super_class), used by ES = 22,212 params.

## Gate G1 checklist
- Proofread neurons ~138.6k? Actual final node count: 139,241
- Input neurons >= 500: PASS (11,118)
- DNs >= 100: PASS (1,303)
- Type pairs > 300k fallback needed?: YES - apply fallback

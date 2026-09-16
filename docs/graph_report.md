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
    - gaba: 769
    - ach: 6,958
    - da: 35
    - ser: 377
    - oct: 116
    - glut: 2,863

- Per-transmitter neuron counts (post-override, sign in parentheses):
  - ser (+1): 1,877
  - da (+1): 5,487
  - gaba (-1): 18,330
  - his (-1): 11,118
  - ach (+1): 80,318
  - glut (-1): 21,994
  - oct (+1): 117

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

## Gate G1 checklist
- Proofread neurons ~138.6k? Actual final node count: 139,241
- Input neurons >= 500: PASS (11,118)
- DNs >= 100: PASS (1,303)
- Type pairs > 300k fallback needed?: YES - apply fallback

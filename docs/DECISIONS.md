# Decisions (append-only; change only with the user's approval)

- D1: Connectome is FlyWire FAFB v783 (Zenodo 10676866 + github.com/flyconnectome/flywire_annotations). Not BANC or MaleCNS.
- D2: Rate model, not spiking. Euler integration.
- D3: Parameters shared per cell type and per (pre type, post type) pair. No per-edge weights.
- D4: Edge signs from predicted neurotransmitter: ACh +1, GABA −1, Glu −1. Others decided in Phase 1 and recorded here.
- D5: Training is behavior cloning from chart-derived labels. PPO is out of scope. ES fine-tuning is a stretch goal.
- D6: Training data is the user's local Clone Hero song library (guitar parts, Easy/Medium/Hard/Expert). The curriculum is the four difficulties. No procedural song generation, except tiny hand-written test fixtures.
- D7: Out of scope: the real Clone Hero game, screen capture, CGEvents.
- D8: The web viewer plays precomputed replays; no live inference in the browser.
- D9 (default; user to confirm before Phase 7): the public site hosts no song audio and no original chart files. Replays show title/artist/charter credit, the highway, and brain activity, with synthesized per-lane hit sounds. A local-only "load your own audio file" option may sync real audio in the browser without uploading or hosting it.
- D10: shared, cross-phase utility code (YAML config loader, run-dir helper) lives in `flyhero/utils/` (`config.py`, `run.py`). Added to the repo layout in PLAN.md during Phase 0 since the original layout diagram had no home for code used by every later package.
- D11: Per-neuron NT sign source (Phase 1) is `top_nt` from the `flywire_annotations` v3.1.0 TSV. Fallback, when `top_nt` is missing or `top_nt_conf < 0.5`: synapse-weighted mean of that neuron's outgoing per-edge NT columns (`gaba_avg`/`ach_avg`/`glut_avg`/`oct_avg`/`ser_avg`/`da_avg`, weighted by `syn_count`) from `proofread_connections_783.feather`, still collapsed to one sign per neuron. Fallback usage counts reported in `docs/graph_report.md`.
- D12: Histamine override — photoreceptor cell types (R1–6, R7, R8; exact `cell_type` spellings confirmed against the live annotations TSV during Phase 1) are forced to histamine, sign −1, regardless of predicted NT. What `top_nt` originally predicted for these neurons is reported in `docs/graph_report.md`.
- D13: Modulatory NTs for v1 — dopamine, serotonin, octopamine all map to sign +1 (not inhibitory). Per-transmitter neuron and edge counts reported in `docs/graph_report.md`.
- D14: Per-neuron type-ID fallback chain (Phase 1 `build_graph.py`): `cell_type` → `hemibrain_type` → `cell_class` → singleton (unique type per unlabeled neuron). Neuron counts landing at each level of the chain reported in `docs/graph_report.md`.
- D15: Neuron positions are stored in `graph.npz` two ways: `pos_nm` (soma/anchor coords converted to nanometers via the FlyWire 4×4×40 nm/voxel scaling) and `pos_voxel` (the raw voxel-index coords as published). Both documented in `docs/graph_report.md` alongside which one is raw vs. derived.

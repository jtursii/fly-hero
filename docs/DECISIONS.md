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

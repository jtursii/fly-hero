# Chaos check (Phase 3b diagnostic)

gain0=10.0, bias_init=0.005

## (a) Reliability
Target song: Steve Ouimette - The Devil Went Down to Georgia

Burn-in songs: ['System of a Down - Toxicity', 'No Doubt - Excuse Me Mr.']

Mean per-DN Pearson correlation over the last 3s: **1.0000** (threshold 0.5, 1303 DNs, 0.7114 fraction with near-zero variance in a run, excluded from the mean)

## (b) Perturbation
Typical photoreceptor input magnitude (measured on Steve Ouimette - The Devil Went Down to Georgia): 0.156949; noise std (1%): 0.001569

Relative DN divergence at 3s: **0.0387** (no fixed numeric gate given by the user for this check)

## (c) gain0 sweep (existing, from Phase 3a)
- gain0=0.1: FAIL
- gain0=0.3: FAIL
- gain0=1.0: FAIL
- gain0=3.0: FAIL
- gain0=10.0: PASS
- gain0=30.0: PASS

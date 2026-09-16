# gain0 stability sweep (Phase 3a, D24)

Song: Scorpions (Steve Ouimette) - Rock You Like A Hurricane (Medium, train split), 10.0s.

"Blank highway" = same rendered background/lanes/strikeline, zero notes (D24). active_r_threshold=0.01, reach_threshold=0.01, active_frac_max (saturation bar)=0.3, reach_frac_min (hard-stop bar)=0.1.


| gain0 | active_frac (last 1s) | max\|v\| | NaN | reach_frac | median latency (ms) | saturated | pass |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.0065 | 1.4606 | False | 0.0000 | nan | False | False |
| 0.3 | 0.0065 | 1.4606 | False | 0.0000 | nan | False | False |
| 1.0 | 0.0179 | 1.4606 | False | 0.0000 | nan | False | False |
| 3.0 | 0.2053 | 3.0050 | False | 0.0000 | nan | False | False |
| 10.0 | 0.2470 | 11.3179 | False | 0.3031 | 3800.00 | False | True |
| 30.0 | 0.2278 | 31.3839 | False | 0.4682 | 3966.67 | False | True |

**At least one gain0 clears reach_frac > 0.1 without saturation or NaN.**

## Context: bias_init

This sweep uses `bias_init=0.005` (`configs/brain.yaml`, D25). The first
version of this sweep (`bias_init=0` per PLAN.md's original default) found
`reach_frac` exactly 0 at every gain0 up to 10,000 -- a provable structural
consequence of D12 (photoreceptor output sign forced -1) with no tonic
baseline for that inhibition to modulate. A first fix attempt
(`bias_init=0.2`) restored signal reach but pushed `active_frac` to
94-99% at low gain0 -- a uniform bias well above `active_r_threshold=0.01`
makes nearly the whole population read as trivially "active" at rest,
regardless of gain0, saturating this sweep's own saturation check even
though real signal reach existed underneath it. `bias_init=0.005` (just
below the active threshold) resolves both: **gain0=10.0 and gain0=30.0 both
pass** (chose **gain0=10.0** in `configs/brain.yaml` as the more
conservative of the two -- smaller `max|v|` for the same low-in-degree
outlier neurons that scale roughly linearly with gain0; gain0=30.0 gives a
larger reach_frac margin, 0.468 vs 0.303, if that's preferred later).

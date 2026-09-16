# gain0 stability sweep (Phase 3a, D24)

Song: Scorpions (Steve Ouimette) - Rock You Like A Hurricane (Medium, train split), 10.0s.

"Blank highway" = same rendered background/lanes/strikeline, zero notes (D24). active_r_threshold=0.01, reach_threshold=0.01, active_frac_max (saturation bar)=0.3, reach_frac_min (hard-stop bar)=0.1.


| gain0 | active_frac (last 1s) | max\|v\| | NaN | reach_frac | median latency (ms) | saturated | pass |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.0064 | 1.4556 | False | 0.0000 | nan | False | False |
| 0.3 | 0.0064 | 1.4556 | False | 0.0000 | nan | False | False |
| 1.0 | 0.0064 | 1.4556 | False | 0.0000 | nan | False | False |
| 3.0 | 0.0063 | 2.4971 | False | 0.0000 | nan | False | False |
| 10.0 | 0.0062 | 8.3237 | False | 0.0000 | nan | False | False |
| 30.0 | 0.0062 | 24.9712 | False | 0.0000 | nan | False | False |

**STOPPED (invariant 7): no gain0 achieves DN reach > 0.1 without saturation.** No fallback list exists for stability in docs/PLAN.md (unlike the throughput gate) -- this needs the user's input.

## Root cause (diagnosed, not just "try more gain0")

This is not a gain0-tuning problem. A direct diagnostic (constant input held for
2 game-seconds, `brain.g` set directly up to 10,000) shows:

- `max|v|` scales exactly linearly with gain0 (== gain0 exactly, at every value
  from 30 to 10,000) -- but this growth is confined entirely to a fixed ~7.7%
  of neurons that turns out to be almost exactly the photoreceptor population
  itself (11,118 / 139,241 = 7.98%).
- **Zero of the 128,123 non-photoreceptor neurons ever reach `r > 0`, at any
  gain0 from 30 to 10,000.** Their `v` is always <= 0 (observed range down to
  exactly `-gain0`, never positive).

The mechanism is a direct consequence of two facts, together:
1. **D12** forces every photoreceptor's *outgoing* synapse sign to -1
   (histamine, biologically correct -- fly photoreceptors depolarize to light
   and histamine-inhibit their targets). Photoreceptors are the *only* nodes
   that receive external current (`I_in`, at invariant 1's designated input
   layer).
2. **PLAN.md's stated init has `b = 0`** everywhere, and `v` starts at 0
   (`init_state`). With no tonic/resting drive anywhere and the sole external
   input entering through an inhibitory-only relay, `syn[post]` for any
   non-photoreceptor node is a sum of `sign * r[pre]` terms that are provably
   never positive: every currently-r=0 presynaptic partner contributes 0, and
   the one population that *isn't* r=0 (photoreceptors) contributes only
   `-softplus(g) * (...)` (sign forced -1). By induction this holds for every
   node transitively downstream of photoreceptors, at any gain0 -- so `r`
   stays at exactly 0 everywhere except the photoreceptors themselves, for
   the entire simulation, regardless of gain0's magnitude.

In real fly vision this pathway signals via *disinhibition* against a tonic
baseline firing rate -- exactly what `bias` (trainable, but initialized to 0
per PLAN.md) would need to encode. At initialization, with `bias=0`, there is
no baseline for inhibition to modulate, so the sign-inverting first synapse
is architecturally invisible to any r-based readout, no matter how strong
gain0 is. This is a provable property of this exact configuration (D12 +
`v0=0` + `b0=0`), not a search-range problem -- confirmed by testing gain0 up
to 10,000 (300x the top of the original sweep) with identically zero
downstream activity.

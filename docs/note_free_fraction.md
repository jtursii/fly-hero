# Note-free fraction report (Phase 3b)

For every train-split (song, difficulty) pair, the fraction of valid 2.0s gradient-window start times (burn_in=1.0s, fps=60) with 0 notes in the window, restricted to windows with either 0 or >=2 notes (matching what the sampler actually draws from; 1-note windows are excluded from the ratio).


**n = 1143 (song, difficulty) pairs**

**Library-wide median note-free fraction: 0.0521**

Mean: 0.0813


| difficulty | n songs | median note-free frac |
|---|---|---|
| Easy | 230 | 0.0638 |
| Medium | 231 | 0.0603 |
| Hard | 231 | 0.0566 |
| Expert | 451 | 0.0410 |

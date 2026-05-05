# Aurora

Aurora is a leverage-uniform polar-factor optimizer. It generalizes Muon's
polar-factor update for non-square weight matrices: instead of `polar(G)`,
which inherits non-uniform left-singular row norms (the *leverage scores*),
Aurora computes `polar(D · G)` for tall G or `polar(G · D)` for wide G,
where the positive diagonal `D` is chosen so the result has uniformly
distributed row norms equal to `sqrt(min(m, n) / max(m, n))`. For square
matrices Aurora reduces to the standard polar.

### Code structure

```text
src/
├── main.py    # Entry point: training loop and CLI
├── msign.py   # Matrix sign function via simple-quintic Newton-Schulz
└── aurora.py  # Aurora update rule
```

### Usage

```python
from aurora import aurora

# Inside the training loop, for each weight tensor W with gradient G
# and a caller-managed momentum buffer m (zeros at init):
aurora(W, G, m, eta=lr, weight_decay=0.025)
```

### Hyperparameters

- `pp_iterations` (default 2): number of D-update refinement iterations.
  `pp_iterations=1` is the "Muon EQ" baseline (single per-row L2 normalize
  then polar). Higher values refine D toward the row-uniform fixed point
  at the cost of one extra polar call per parameter per iteration.
- `pp_beta` (default 0.5): damping exponent for the D update, in `(0, 1]`.
  Default 0.5 gives undamped square-root steps; lower values damp
  oscillation between odd/even D iterates.
- `mu` (default 0.95), `nesterov` (default True), `weight_decay` (default
  0.025): standard Muon / SGD-momentum hyperparameters.

### Numerical parity with the Modded-NanoGPT track-3 baseline

`msign.py` uses simple-quintic 12-iteration Newton-Schulz with coefficients
`(2, -1.5, 0.5)` — byte-identical to the
[Modded-NanoGPT track-3 baseline](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/train_gpt_simple.py)
`zeropower_via_newtonschulz5`. Aurora's full `aurora()` step (Nesterov
momentum → leverage-uniform polar → spectral aspect-ratio scale →
decoupled weight decay) reproduces the leaderboard submission's
`muon_update + Muon.step` byte-for-byte. Anyone using this code can
reproduce that benchmark's numerical trajectory exactly (modulo random
seed and hardware nondeterminism).

### Reference

Aurora was previously named **Harold** in the
[kitchen](https://github.com/tilde-research/kitchen) repository
(`feature/harold` by Pai et al.). This release renames the public interface
to Aurora.

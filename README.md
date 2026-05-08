# Aurora

Aurora is an optimizer for non-square matrices that achieves more effective utilization of MLP neurons. Instead of `polar(G)`, which inherits non-uniform left-singular row norms, Aurora iteratively approximates a projection onto the intersection of the row oblique and Steifel manifolds, giving more balanced updates without sacrificing polar factor precision. For square matrices Aurora reduces to the standard Muon update.

See the blog for more information:
https://blog.tilderesearch.com/blog/aurora

### Code structure

```text
src/
├── main.py               # Entry point: training loop and CLI
├── polar.py              # Polar factor via simple-quintic Newton-Schulz
├── aurora.py             # Aurora update rule
└── riemannian_aurora.py  # Riemannian Aurora: Riemannian gradient ascent on the balanced Stiefel manifold

```

### Usage

```python
from aurora import aurora

# Inside the training loop, for each weight tensor W with gradient G
# and a caller-managed momentum buffer m (zeros at init):
aurora(W, G, m, eta=lr, weight_decay=0.025)
```

### Hyperparameters

- `pp_iterations` (default 2): number of update refinement iterations.
  Higher values refine the update toward the row-uniform fixed point
  at the cost of one extra polar call per parameter per iteration.
- `pp_beta` (default 0.5): damping exponent for the row normalization step, in `(0, 1]`.
  Default 0.5 gives undamped square-root steps; lower values damp
  oscillation between odd/even D iterates.
- `mu` (default 0.95), `nesterov` (default True), `weight_decay` (default
  0.025): standard Muon / SGD-momentum hyperparameters.

### Utilities

`polar.py` uses simple-quintic 12-iteration Newton-Schulz with coefficients. Aurora's full `aurora()` step follows: Nesterov momentum → leverage-uniform polar → spectral aspect-ratio scale → decoupled weight decay. Different Newton-Schultz iterations can be added as a drop-in replacement to our `polar` function.

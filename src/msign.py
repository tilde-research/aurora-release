"""Matrix sign function (polar factor) via simple-quintic Newton-Schulz.

For a matrix G with SVD G = U Σ V^T, this returns U V^T (the polar factor /
matrix sign of G). All non-zero singular values are mapped to 1.

Implementation: 12 iterations of the simple-quintic polynomial
    p(σ) = 2σ - 1.5σ³ + 0.5σ⁵
which has fixed points at σ ∈ {0, 1, √2}, with σ=1 super-attracting
(p'(1) = 0). After 12 iterations of cubic-rate convergence, all input
singular values in (0, √2) are driven to 1 to bf16 precision.

This is the polar method used by the Modded-NanoGPT track-3 baseline at
https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/train_gpt_simple.py
"not optimizing for wallclock speed". We match it byte-for-byte so that
optimizers built on this `msign` reproduce leaderboard val_loss curves.
"""

import torch


@torch.no_grad()
def msign(G: torch.Tensor) -> torch.Tensor:
    """Matrix sign / polar factor via 12-step simple-quintic Newton-Schulz.

    Args:
        G: input matrix of shape [..., m, n].

    Returns:
        polar(G) of the same shape, in bfloat16. All non-zero singular values
        of G are mapped to 1.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm <= 1 so the iteration converges to polar.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Simple-quintic coefficients: p(σ) = aσ + bσ³ + cσ⁵ with σ=1 super-attracting.
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

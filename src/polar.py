"""Polar factor via CANS-12 Newton-Schulz.

For a matrix G with SVD G = U Σ V^T, this approximates U V^T (the polar
factor of G) by mapping non-zero singular values toward 1.

The first nine cubic iterations use the Chebyshev-optimized coefficients
from CANS (https://arxiv.org/abs/2506.10935), followed by three classic
Newton-Schulz iterations. This matches Poutine's CANS-12 coefficient schedule.
"""

import torch


@torch.no_grad()
def polar(G: torch.Tensor) -> torch.Tensor:
    """Approximate the polar factor with CANS-12 Newton-Schulz.

    Args:
        G: input matrix of shape [..., m, n].

    Returns:
        An FP32 approximation to polar(G) with the same shape.
    """
    assert G.ndim >= 2
    # Run CANS in FP32
    X = G.to(torch.float32)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm <= 1 so the iteration converges to polar.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    coefficients = (
        (5.182503604966906, -5.178098480082684),
        (2.586120737395915, -0.6479542005271643),
        (2.567364126726186, -0.6454968804392178),
        (2.520560084348265, -0.6393528082067044),
        (2.410759275435182, -0.6248683598710716),
        (2.1883348130094173, -0.5952022073798908),
        (1.8595760874873613, -0.5504490972723968),
        (1.589020160467417, -0.5126569802066718),
        (1.5051653981684994, -0.5007377068751799),
        (1.5, -0.5),
        (1.5, -0.5),
        (1.5, -0.5),
    )
    for a, b in coefficients:
        A = X @ X.mT
        X = a * X + b * A @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

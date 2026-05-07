import torch

from polar import polar


@torch.no_grad()
def aurora(
    W,
    G,
    momentum,
    eta=0.05,
    weight_decay=0.025,
    mu=0.95,
    nesterov=True,
    pp_iterations=2,
    pp_beta=0.5,
    eps=1e-7,
):
    if W.ndim != 2:
        raise ValueError(f"aurora expects 2D weight tensors, got shape {tuple(W.shape)}")
    if G.shape != W.shape:
        raise ValueError(f"G shape {tuple(G.shape)} must match W shape {tuple(W.shape)}")
    if momentum.shape != W.shape:
        raise ValueError(f"momentum shape {tuple(momentum.shape)} must match W shape {tuple(W.shape)}")
    if not (0.0 < mu < 1.0):
        raise ValueError(f"mu must be in (0, 1), got {mu}")
    if eta <= 0.0:
        raise ValueError(f"eta must be positive, got {eta}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if pp_iterations < 1:
        raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
    if pp_beta <= 0.0:
        raise ValueError(f"pp_beta must be positive, got {pp_beta}")

    # SGD-momentum (Nesterov by default).
    momentum.lerp_(G, 1 - mu)
    # Clone when not using Nesterov to avoid scaling the momentum buffer in-place below.
    update = G.lerp_(momentum, mu) if nesterov else momentum.clone()
    # Aurora's leverage-uniform polar via diagonal preconditioning.
    m, n = update.size(-2), update.size(-1)
    if m == n:
        # Square: standard polar (no leverage freedom to exploit).
        update = polar(update)
    else:
        # For wide G, transpose to tall, apply, transpose back.
        # polar(G * D) = polar(D * G^T)^T
        transposed = m < n
        if transposed:
            update = update.mT
            m, n = n, m
        G32 = update.to(torch.float32)
        target_row_sq = n / m
        row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
        D = 1.0 / row_norm
        for k in range(pp_iterations):
            U = polar(D * G32)
            if k < pp_iterations - 1:
                row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
                D = D * (target_row_sq / row_sq).pow(pp_beta)
        update = U.mT if transposed else U
    # Spectral aspect-ratio scaling (Muon convention).
    update *= max(1, G.size(-2) / G.size(-1)) ** 0.5
    if not update.isfinite().all():
        raise RuntimeError(
            f"aurora produced non-finite update for parameter of shape {tuple(W.shape)}. "
            "Check for NaN/Inf in gradients or an ill-conditioned weight matrix."
        )
    # Decoupled weight decay then apply.
    W.mul_(1 - eta * weight_decay)
    W.add_(update, alpha=-eta)
    return W

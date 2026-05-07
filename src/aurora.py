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
    # SGD-momentum (Nesterov by default).
    momentum.lerp_(G, 1 - mu)
    update = G.lerp_(momentum, mu) if nesterov else momentum
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
    # Decoupled weight decay then apply.
    W.mul_(1 - eta * weight_decay)
    W.add_(update, alpha=-eta)
    return W

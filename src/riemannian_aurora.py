import math
import torch

from polar import polar


@torch.no_grad()
def _solve_row_norm_multipliers(U, r, b, max_iter=20, eps=1e-7):
    """Approximately solve (r I - P ∘ P) λ = b, where P = U U^T."""
    # Stabilize the CG operator. This is a regularized approximate solve.
    h_sq = U.pow(2).sum(dim=-1).pow(2)
    reg = (h_sq.max() - r + 1e-3).clamp_min(0.0).item()
    r_eff = r + reg

    def matvec(v):
        # (P ∘ P) v without explicitly forming P.
        # Equivalent to: ((U U^T) ∘ (U U^T)) v
        T = U.mT @ (v.unsqueeze(-1) * U)
        return r_eff * v - (U @ T * U).sum(dim=-1)

    x = torch.zeros_like(b)
    res = b.clone()
    p = res.clone()
    rs_old = (res * res).sum()
    b_norm = b.norm().clamp_min(1e-12)

    for _ in range(max_iter):
        Ap = matvec(p)
        denom = (p * Ap).sum()
        if denom < 1e-30:
            break

        alpha = rs_old / denom
        x = x + alpha * p
        res = res - alpha * Ap

        rs_new = (res * res).sum()
        if not rs_new.isfinite():
            break
        if rs_new.sqrt() < 1e-8 * b_norm:
            break

        p = res + (rs_new / rs_old.clamp_min(1e-30)) * p
        rs_old = rs_new

    return x if x.isfinite().all() else torch.zeros_like(b)


@torch.no_grad()
def _riemannian_balanced_polar(
    G,
    outer_steps=3,
    cg_steps=20,
    riemannian_eta=0.1,
    retraction_steps=2,
    eps=1e-7,
):
    """Approximate balanced-Stiefel update aligned with G.

    Solves approximately:

        max_U <G, U>
        s.t. U^T U = I_n,
             ||U_i||^2 = n/m.

    For wide matrices, transposes internally and returns to original shape.
    """
    if G.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {tuple(G.shape)}")

    transposed = G.size(-2) < G.size(-1)
    if transposed:
        G = G.mT.contiguous()

    G32 = G.to(torch.float32)
    m, n = G32.shape
    r = n / m
    target_row_norm = math.sqrt(r)

    # Initial point: Muon/polar update.
    U = polar(G32).to(torch.float32)

    for _ in range(outer_steps):
        # Stiefel correction.
        UtG = U.mT @ G32
        B = 0.5 * (UtG + UtG.mT)

        # Row-norm correction RHS.
        q = (G32 * U).sum(dim=-1) - (U @ B * U).sum(dim=-1)
        q = q - q.mean()

        # Solve for row-norm Lagrange multipliers.
        lam = _solve_row_norm_multipliers(U, r, q, max_iter=cg_steps, eps=eps)
        lam = lam - lam.mean()

        # Tangent projection: Z = G - U S - D U.
        S = B - U.mT @ (lam.unsqueeze(-1) * U)
        Z = G32 - U @ S - lam.unsqueeze(-1) * U

        if not Z.isfinite().all():
            break

        # Riemannian ascent step.
        Y = U + riemannian_eta * Z

        # Approximate retraction by alternating row normalization and polar.
        for _ in range(retraction_steps):
            row_norm = Y.norm(dim=-1, keepdim=True).clamp_min(eps)
            Y = Y * (target_row_norm / row_norm)
            Y = polar(Y).to(torch.float32)

        U = Y

    return U.mT.contiguous() if transposed else U


@torch.no_grad()
def riemannian_aurora(
    W,
    G,
    momentum,
    eta=0.05,
    weight_decay=0.025,
    mu=0.95,
    nesterov=True,
    outer_steps=3,
    cg_steps=20,
    riemannian_eta=0.1,
    retraction_steps=2,
    eps=1e-7,
):
    # SGD-momentum (Nesterov by default).
    momentum.lerp_(G, 1 - mu)
    update = G.lerp_(momentum, mu) if nesterov else momentum

    # Riemannian-Aurora balanced polar update.
    m, n = update.size(-2), update.size(-1)
    if m == n:
        # Square: no leverage freedom to exploit.
        update = polar(update)
    else:
        update = _riemannian_balanced_polar(
            update,
            outer_steps=outer_steps,
            cg_steps=cg_steps,
            riemannian_eta=riemannian_eta,
            retraction_steps=retraction_steps,
            eps=eps,
        )

    # Spectral aspect-ratio scaling (Muon convention).
    update *= max(1, G.size(-2) / G.size(-1)) ** 0.5

    # Decoupled weight decay then apply.
    W.mul_(1 - eta * weight_decay)
    W.add_(update, alpha=-eta)
    return W
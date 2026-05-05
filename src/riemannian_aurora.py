# Copyright 2026 Tilde Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Riemannian Aurora optimizer.

Riemannian ascent inner solver for balanced Muon-style updates, applied to
2-D weight matrices.  All other parameters (biases, norms, embeddings) are
handled by an inner AdamW instance.

Public API
----------
polar              — Polar Express Newton-Schulz polar factor
solve_coupled_update — find the balanced-Stiefel update direction for a gradient
RiemannianAurora   — the optimizer
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = [
    "polar",
    "solve_coupled_update",
    "RiemannianAurora",
]

# ---------------------------------------------------------------------------
# Polar Express Newton-Schulz
# ---------------------------------------------------------------------------
# Coefficients from: https://arxiv.org/abs/2505.16932
# Each tuple (a, b, c) defines the quintic step  X ← a·X + (b·A + c·A²)·X
# where A = X·Xᵀ  (the c·A² term makes it degree-5 in singular values).
# Steps beyond the listed 8 coefficients fall back to the classic (1.5, -0.5)
# iteration which converges to the polar factor from inside the unit ball.
_POLAR_EXPRESS_COEFFS: list[tuple[float, float, float]] = [
    (8.2051, -22.9019, 16.4607),
    (4.0664,  -2.8612,  0.5184),
    (3.9096,  -2.8234,  0.5250),
    (3.2856,  -2.4153,  0.4853),
    (2.2779,  -1.6198,  0.3985),
    (1.8726,  -1.2307,  0.3585),
    (1.8564,  -1.2132,  0.3568),
    (1.8750,  -1.2500,  0.3750),
]
_CLASSIC_COEFF: tuple[float, float, float] = (1.5, -0.5, 0.0)


def _ns_step(X: Tensor, a: float, b: float, c: float) -> Tensor:
    """Single quintic Newton-Schulz step: X ← a·X + (b·A + c·A²)·X, A = X·Xᵀ.

    The c·A²·X term is degree-5 in the singular values, hence "quintic."
    When c == 0 this reduces to the classic degree-3 (cubic) step.
    """
    A = X @ X.mT
    B = b * A if c == 0.0 else b * A + c * (A @ A)
    return torch.addmm(X, B, X, beta=a, alpha=1.0)


def polar(A: Tensor, steps: int = 8, eps: float = 1e-7) -> Tensor:
    """Return the polar factor of *A* via Polar Express Newton-Schulz iteration.

    The iteration works on the tall (m ≥ n) form; *A* is transposed internally
    when necessary and un-transposed before returning.

    Args:
        A:     2-D float tensor.
        steps: Number of Newton-Schulz iterations (≥ 1).
        eps:   Frobenius-norm floor used during normalisation.

    Returns:
        Semi-orthogonal float32 tensor of the same shape as *A*
        (columns orthonormal when m ≥ n, rows orthonormal otherwise).
    """
    if A.ndim != 2:
        raise ValueError(f"polar expects a 2-D tensor, got shape {tuple(A.shape)}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    A = A.float()
    transposed = A.shape[0] < A.shape[1]
    if transposed:
        A = A.mT.contiguous()

    X = A / A.norm().clamp_min(eps)

    for i in range(steps):
        a, b, c = _POLAR_EXPRESS_COEFFS[i] if i < len(_POLAR_EXPRESS_COEFFS) else _CLASSIC_COEFF
        X = _ns_step(X, a, b, c)

    return X.mT.contiguous() if transposed else X


# ---------------------------------------------------------------------------
# Row-norm Lagrange multiplier solver
# ---------------------------------------------------------------------------

def _solve_row_norm_multipliers(
    U: Tensor, r: float, b: Tensor, max_iter: int
) -> Tensor:
    """Approximately solve (r·I − P∘P) λ ≈ b via CG, where P = UUᵀ.

    The ideal operator (r·I − P∘P) is singular on the Stiefel manifold (the
    constant vector is in its null space, handled by the caller's mean-centering
    of both b and λ).  Away from the manifold the operator may not be positive
    definite, so we add a diagonal stabilization:

        ε = max(0, maxᵢ ‖uᵢ‖⁴ − r + δ)

    This is a heuristic that makes the diagonal positive and works well in
    practice when U is near-Stiefel, but does not guarantee global PD-ness.
    The resulting solve is therefore a regularized approximation, not an exact
    tangent projection.

    Args:
        U:        [m, n] current iterate (tall, m ≥ n).
        r:        target squared row-norm = n / m.
        b:        [m] right-hand side (mean-centred by caller).
        max_iter: CG iteration budget.

    Returns:
        Approximate solution λ.
    """
    with torch.no_grad():
        h_sq = (U * U).sum(dim=-1).pow(2)
        eps = (h_sq.max() - r + 1e-3).clamp_min_(0.0).item()
    r_eff = r + eps

    def matvec(v: Tensor) -> Tensor:
        T = U.mT @ (v.unsqueeze(-1) * U)
        return r_eff * v - (U @ T * U).sum(dim=-1)

    x = torch.zeros_like(b)
    res = b.clone()
    p = res.clone()
    rs_old = (res * res).sum()
    b_norm = b.norm().clamp_min(1e-12)

    for i in range(max_iter):
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


# ---------------------------------------------------------------------------
# solve_coupled_update
# ---------------------------------------------------------------------------

def solve_coupled_update(
    G: Tensor,
    outer_steps: int = 3,
    cg_steps: int = 20,
    eta: float = 0.1,
    proj_steps: int = 2,
) -> Tensor:
    """Find the balanced-Stiefel update direction for gradient *G*.

    Runs Riemannian ascent on the objective max_U <G, U> subject to U lying
    on the equal-row-norm Stiefel manifold:

        M = { U ∈ ℝ^{m×n} : UᵀU = Iₙ,  ‖uᵢ‖² = n/m  ∀i }

    The result is the approximate maximiser U ∈ M, suitable for use as a
    scaled weight update (cf. Muon).

    Wide matrices (m < n) are handled by transposing internally; the output
    has the same shape as the input.  For wide inputs the constraint is applied
    after transposition, so the returned update has approximately orthonormal
    rows and uniform column norms in the original orientation.

    Algorithm (T outer Riemannian ascent steps):

        U = polar(G)
        for t = 1 .. T:
            B = sym(UᵀG)
            q = diag(GUᵀ) − diag(UBUᵀ);   q −= mean(q)
            λ ≈ CG((rI − P∘P) λ = q, K steps);  λ −= mean(λ)
            S = B − Uᵀ diag(λ) U
            Z = G − US − diag(λ)U           # Riemannian ascent direction
            Y = U + η·Z
            for j = 1 .. J:                 # alternating projection onto M
                normalise rows of Y to ‖yᵢ‖ = √r
                Y = polar(Y)
            U = Y

    Args:
        G:           2-D gradient tensor (any shape; transposed internally if wide).
        outer_steps: Riemannian ascent steps T.
        cg_steps:    CG budget K per outer step.
        eta:         Riemannian ascent step size η.
        proj_steps:  Alternating projection steps J (row-norm ↔ Stiefel).

    Returns:
        Float32 tensor of the same shape as *G*, approximately on M.
    """
    if G.ndim != 2:
        raise ValueError(f"G must be 2-D, got shape {tuple(G.shape)}")

    G = G.float()
    transposed = G.shape[0] < G.shape[1]
    if transposed:
        G = G.mT.contiguous()

    m, n = G.shape
    r = n / m
    target_norm = math.sqrt(r)

    U = polar(G)

    for _ in range(outer_steps):
        UtG = U.mT @ G
        B = (UtG + UtG.mT) * 0.5
        q = (G * U).sum(dim=-1) - (U @ B * U).sum(dim=-1)
        q = q - q.mean()

        lam = _solve_row_norm_multipliers(U, r, q, cg_steps)
        lam = lam - lam.mean()

        S = B - U.mT @ (lam.unsqueeze(-1) * U)
        Z = G - U @ S - lam.unsqueeze(-1) * U

        if not Z.isfinite().all():
            break

        Y = U + eta * Z
        for _ in range(proj_steps):     # alternating projection onto M
            Y = Y * (target_norm / Y.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            Y = polar(Y)
        U = Y

    if transposed:
        U = U.mT.contiguous()

    return U


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class RiemannianAurora(Optimizer):
    """Riemannian ascent inner solver for balanced Muon-style updates.

    Applies ``solve_coupled_update`` to every matrix-shaped parameter
    (flattening all non-leading dimensions) and falls back to AdamW for all
    other parameters (biases, layer-norm scales, etc.).

    Update pipeline for each matrix-shaped weight W with gradient G:

        1. Nesterov momentum:  M ← β·M + G;   G_eff = G + β·M
           (Lookahead convention: G_eff = current grad + β × accumulated buffer.
           Matches the Keller-style Muon reference implementation.)
        2. U = solve_coupled_update(G_eff)   [balanced-Stiefel update direction]
        3. Weight update: W ← W − lr · scale · U,  scale = max(1, m/n)^½

    Args:
        matrix_params: Iterable of matrix-shaped weight tensors (non-leading
                       dimensions are flattened before the update is computed).
        scalar_params: Iterable of all remaining parameters (may be empty).
        lr:            Learning rate for matrix params.
        momentum:      Nesterov momentum β ∈ [0, 1).
        weight_decay:  Decoupled weight decay for matrix params.
        outer_steps:   Riemannian ascent steps T.
        cg_steps:      CG budget K per outer step.
        riemannian_lr: Riemannian ascent step size η.
        proj_steps:    Alternating projection steps J (row-norm ↔ Stiefel).
        scalar_lr:     Learning rate for scalar params (AdamW).
        scalar_betas:  AdamW β₁, β₂ for scalar params.
        scalar_wd:     Weight decay for scalar params.
    """

    def __init__(
        self,
        matrix_params: Iterable[Tensor],
        scalar_params: Iterable[Tensor],
        lr: float = 2e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        outer_steps: int = 3,
        cg_steps: int = 20,
        riemannian_lr: float = 0.1,
        proj_steps: int = 2,
        scalar_lr: float = 1e-3,
        scalar_betas: tuple[float, float] = (0.9, 0.95),
        scalar_wd: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if outer_steps < 1:
            raise ValueError(f"outer_steps must be >= 1, got {outer_steps}")

        self._outer_steps = outer_steps
        self._cg_steps = cg_steps
        self._riemannian_lr = riemannian_lr
        self._proj_steps = proj_steps

        scalar_params = list(scalar_params)
        self._scalar_opt: Optimizer | None = (
            torch.optim.AdamW(
                scalar_params,
                lr=scalar_lr,
                betas=scalar_betas,
                weight_decay=scalar_wd,
            )
            if scalar_params
            else None
        )

        defaults: dict[str, Any] = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(matrix_params, defaults)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        if self._scalar_opt is not None:
            self._scalar_opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:  # type: ignore[override]
        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._scalar_opt is not None:
            self._scalar_opt.step()
        self._step_matrix_params()
        return loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_matrix_params(self) -> None:
        for group in self.param_groups:
            lr: float   = group["lr"]
            beta: float = group["momentum"]
            wd: float   = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.float()
                state = self.state[p]
                if not state:
                    state["buf"] = torch.zeros_like(g)

                buf: Tensor = state["buf"]
                buf.mul_(beta).add_(g)
                g_eff = g.add(buf, alpha=beta)      # Nesterov: G + β·M

                orig_shape = p.shape
                g2d = g_eff.reshape(orig_shape[0], -1)

                U = solve_coupled_update(
                    g2d,
                    outer_steps=self._outer_steps,
                    cg_steps=self._cg_steps,
                    eta=self._riemannian_lr,
                    proj_steps=self._proj_steps,
                )

                m_orig = orig_shape[0]
                n_orig = math.prod(orig_shape[1:])
                scale = max(1.0, m_orig / n_orig) ** 0.5

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(U.reshape(orig_shape).to(p.dtype), alpha=-lr * scale)

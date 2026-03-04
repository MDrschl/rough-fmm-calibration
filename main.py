import numpy as np
import torch
import torch.nn as nn
import pickle
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.special import hyp2f1 as scipy_hyp2f1
import py_lets_be_rational as _lbr

def _stable_key_hash(key: tuple) -> int:
    """Deterministic hash for swaption keys (expiry, tenor)."""
    return int(key[0] * 1000) * 100 + int(key[1])

ATM_TOL = 1e-6

def black_price_np(S0, K, T, sigma, annuity, is_call=True):
    """Black (1976) swaption price (numpy, scalar or vectorized)."""
    S0, K, T, sigma = np.asarray(S0), np.asarray(K), np.asarray(T), np.asarray(sigma)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if is_call:
        return annuity * (S0 * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return annuity * (K * norm.cdf(-d2) - S0 * norm.cdf(-d1))

def bachelier_price_np(S0, K, T, sigma_n, annuity, is_call=True):
    """Bachelier (normal) swaption price (numpy)."""
    sqrt_T = np.sqrt(T)
    d = (S0 - K) / (sigma_n * sqrt_T)
    undiscounted = sigma_n * sqrt_T * (d * norm.cdf(d) + norm.pdf(d))
    if not is_call:
        undiscounted = undiscounted - (S0 - K)
    return annuity * undiscounted

def black_iv(price, forward, strike, T, annuity=1.0, is_call=True):
    """Black implied volatility via Jäckel's *Let's Be Rational* (2015)."""
    if price <= 0 or forward <= 0 or strike <= 0 or T <= 0 or annuity <= 0:
        return np.nan

    p = price / annuity
    q = 1.0 if is_call else -1.0

    try:
        sigma = _lbr.implied_volatility_from_a_transformed_rational_guess(
            p, forward, strike, T, q,
        )
        return float(sigma) if sigma > 0 and np.isfinite(sigma) else np.nan
    except Exception:
        return np.nan

def bachelier_to_black_iv(S0, K, T, sigma_n, annuity, is_call=True):
    """Convert Bachelier (normal) IV to Black (lognormal) IV."""
    if K <= 0 or S0 <= 0 or T <= 0 or sigma_n <= 0:
        return np.nan
    target_price = bachelier_price_np(S0, K, T, sigma_n, annuity, is_call)
    return black_iv(target_price, S0, K, T, annuity, is_call)

@dataclass
class SwaptionData:
    """Market data for a single swaption (expiry, tenor) pair."""
    I: int
    J: int
    expiry_years: float
    tenor_years: float
    S0: torch.Tensor
    A0: torch.Tensor
    Pi: torch.Tensor
    pi: torch.Tensor
    strikes: torch.Tensor
    ivs_black: torch.Tensor
    ivs_normal: torch.Tensor
    is_call: torch.Tensor
    target_prices: torch.Tensor
    vegas: torch.Tensor
    n_strikes: int

@dataclass
class MarketData:
    """Complete market data for calibration."""
    P: torch.Tensor
    R: torch.Tensor
    N: int
    theta: float

    swaptions: dict

    groups: dict = field(default_factory=dict)

    subset_keys: dict = field(default_factory=dict)

    device: str = "cpu"

def load_market_data(
    pkl_path: str,
    subset: str = "joint_all_smiles",
    date: str = "2024-12-09",
    device: str = "cpu",
) -> 'MarketData':
    """Load preprocessed market data and convert to torch tensors."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    if "dates" in raw and date in raw:
        date_data = raw[date]
        N = date_data["T_N"]
        theta = date_data["theta"]
        P_np = date_data["discount_factors"][:N + 1]
        R_np = date_data["forward_term_rates"][:N + 1]
        swaptions_raw = date_data["swaptions"]
        subset_keys_all = date_data["calibration_subsets"]
    elif "metadata" in raw and "swaptions" in raw:
        N = raw["metadata"]["T_N"]
        theta = raw["metadata"]["theta"]
        P_np = raw["discount_factors"][:N + 1]
        R_np = raw["forward_term_rates"][:N + 1]
        swaptions_raw = raw["swaptions"]
        subset_keys_all = raw.get("calibration_subsets", {})
    else:
        raise ValueError(f"Unknown pickle format. Available keys: {list(raw.keys())}")

    P = torch.tensor(P_np, dtype=torch.float64, device=device)
    R = torch.tensor(R_np, dtype=torch.float64, device=device)

    active_keys = subset_keys_all.get(subset, list(swaptions_raw.keys()))

    swaptions = {}
    for key in active_keys:
        swn_raw = swaptions_raw[key]
        I, J = swn_raw["I"], swn_raw["J"]
        S0 = swn_raw["S0"]
        A0 = swn_raw["A0"]
        T_exp = swn_raw["expiry_years"]

        strikes_np = swn_raw["strikes"]
        is_call_np = swn_raw["is_call"]
        n_strikes = len(strikes_np)

        if "ivs_black" in swn_raw and "ivs_normal" in swn_raw:
            ivs_black_np = swn_raw["ivs_black"]
            ivs_normal_np = swn_raw["ivs_normal"]
        elif "ivs_normal" in swn_raw:
            ivs_normal_np = swn_raw["ivs_normal"]
            ivs_black_np = np.array([
                bachelier_to_black_iv(S0, K, T_exp, sig_n, A0, bool(ic))
                for K, sig_n, ic in zip(strikes_np, ivs_normal_np, is_call_np)
            ])
        else:
            ivs_raw = swn_raw["ivs"]
            atm_mask = np.abs(strikes_np - S0) < ATM_TOL
            otm_mask = ~atm_mask
            ivs_black_np = np.copy(ivs_raw)

            if np.any(otm_mask):
                for i in np.where(otm_mask)[0]:
                    sigma_n = ivs_raw[i] / 100.0
                    iv_b = bachelier_to_black_iv(
                        S0, strikes_np[i], T_exp,
                        sigma_n, A0, bool(is_call_np[i])
                    )
                    ivs_black_np[i] = iv_b if not np.isnan(iv_b) else ivs_raw[i]

            ivs_normal_np = np.array([
                sig_b * S0 for sig_b in ivs_black_np
            ])

        prices_np = np.array([
            float(black_price_np(S0, K, T_exp, sig, A0, ic))
            for K, sig, ic in zip(strikes_np, ivs_black_np, is_call_np)
        ])

        sqrt_T = np.sqrt(T_exp)
        vegas_np = np.array([
            float(A0 * S0 * sqrt_T * norm.pdf(
                (np.log(S0 / K) + 0.5 * sig**2 * T_exp) / (sig * sqrt_T)
            )) if sig > 0 else 0.0
            for K, sig in zip(strikes_np, ivs_black_np)
        ])

        swn = SwaptionData(
            I=I, J=J,
            expiry_years=swn_raw["expiry_years"],
            tenor_years=swn_raw["tenor_years"],
            S0=torch.tensor(S0, dtype=torch.float64, device=device),
            A0=torch.tensor(A0, dtype=torch.float64, device=device),
            Pi=torch.tensor(swn_raw["frozen_weights_Pi"], dtype=torch.float64, device=device),
            pi=torch.tensor(swn_raw["normalized_weights_pi"], dtype=torch.float64, device=device),
            strikes=torch.tensor(strikes_np, dtype=torch.float64, device=device),
            ivs_black=torch.tensor(ivs_black_np, dtype=torch.float64, device=device),
            ivs_normal=torch.tensor(ivs_normal_np, dtype=torch.float64, device=device),
            is_call=torch.tensor(is_call_np, dtype=torch.bool, device=device),
            target_prices=torch.tensor(prices_np, dtype=torch.float64, device=device),
            vegas=torch.tensor(vegas_np, dtype=torch.float64, device=device),
            n_strikes=n_strikes,
        )
        swaptions[key] = swn

    groups = {}
    for key, swn in swaptions.items():
        ij = (swn.I, swn.J)
        if ij not in groups:
            groups[ij] = []
        groups[ij].append(swn)

    return MarketData(
        P=P, R=R, N=N, theta=theta,
        swaptions=swaptions,
        groups=groups,
        subset_keys=subset_keys_all,
        device=device,
    )

class MappedRoughSABRParams(nn.Module):
    """Layer 0: Learnable parameters for the Mapped Rough SABR FMM."""

    def __init__(self, N: int, device: str = "cpu"):
        """Initialize with N forward rates."""
        super().__init__()
        self.N = N
        self._device = device

        self.H_tilde = nn.Parameter(
            torch.tensor(-0.405, dtype=torch.float64, device=device)
        )

        self.eta_tilde = nn.Parameter(
            torch.tensor(2.194, dtype=torch.float64, device=device)
        )

        self.alpha_tilde = nn.Parameter(
            torch.full((N,), -0.853, dtype=torch.float64, device=device)
        )

        init_omega = torch.zeros(N + 1, N + 1, dtype=torch.float64, device=device)
        init_omega[1:, 0] = np.log(0.5)
        self.omega_tilde = nn.Parameter(init_omega)

    def get_H(self) -> torch.Tensor:
        """Hurst exponent H ∈ (0, 0.5)."""
        return 0.5 * torch.sigmoid(self.H_tilde)

    def get_eta(self) -> torch.Tensor:
        """Vol-of-vol eta ∈ (0, ∞) — rBergomi convention."""
        return nn.functional.softplus(self.eta_tilde)

    def get_alpha(self) -> torch.Tensor:
        """Variance levels alpha_j ∈ (0, ∞), shape (N,)."""
        return nn.functional.softplus(self.alpha_tilde)

    def get_full_correlation_matrix(self) -> torch.Tensor:
        """Full (N+1)×(N+1) correlation matrix Σ for (W⁰, W¹, ..., Wᴺ)."""
        Np1 = self.N + 1

        omega = torch.zeros(Np1, Np1, dtype=torch.float64, device=self._device)

        omega[1:, 0] = (
            torch.pi / 2 + (torch.pi / 2) * torch.sigmoid(self.omega_tilde[1:, 0])
        )

        omega[1:, 1:] = torch.pi * torch.sigmoid(self.omega_tilde[1:, 1:])

        B = torch.zeros(Np1, Np1, dtype=torch.float64, device=self._device)

        for i in range(Np1):
            sin_prod = torch.ones(1, dtype=torch.float64, device=self._device)
            for l in range(i):
                B[i, l] = torch.cos(omega[i, l]) * sin_prod
                sin_prod = sin_prod * torch.sin(omega[i, l])
            B[i, i] = sin_prod

        return B @ B.T

    def get_rho0(self) -> torch.Tensor:
        """Spot-vol correlations ρ_{0,j} for j = 1,...,N."""
        Sigma = self.get_full_correlation_matrix()
        return Sigma[0, 1:]

    def get_correlation_matrix(self) -> torch.Tensor:
        """Forward-rate correlation matrix ρ_{ij} for i,j = 1,...,N."""
        Sigma = self.get_full_correlation_matrix()
        return Sigma[1:, 1:]

    def forward(self) -> dict:
        """Compute all constrained parameters from unconstrained ones."""
        Sigma = self.get_full_correlation_matrix()
        return {
            "H": self.get_H(),
            "eta": self.get_eta(),
            "alpha": self.get_alpha(),
            "rho0": Sigma[0, 1:],
            "rho": Sigma[1:, 1:],
            "Sigma": Sigma,
        }

    def summary(self) -> str:
        """Print current parameter values."""
        with torch.no_grad():
            p = self.forward()
            H_val = p['H'].item()
            eta_val = p['eta'].item()
            kappa_adachi = eta_val * np.sqrt(2.0 * H_val)
            lines = [
                "=== Mapped Rough SABR FMM Parameters ===",
                f"H     = {H_val:.4f}",
                f"eta   = {eta_val:.4f}",
                f"kappa = {kappa_adachi:.4f}  (κ = η√(2H), Adachi convention)",
                f"alpha = {p['alpha'].numpy()}",
                f"rho0  = {p['rho0'].numpy()}",
                f"rho   = (N×N matrix, diag check: {torch.diag(p['rho']).numpy()})",
                f"Sigma = ({self.N+1}×{self.N+1} matrix, PSD, "
                f"diag check: {torch.diag(p['Sigma']).numpy()})",
            ]
            return "\n".join(lines)

    def fix_H(self):
        """Freeze H (for Stage 2 calibration with exact Cholesky scheme)."""
        self.H_tilde.requires_grad_(False)

    def unfix_H(self):
        """Unfreeze H (for Stage 1 with approximate scheme)."""
        self.H_tilde.requires_grad_(True)

    def set_H(self, H_value: float):
        """Set H to a specific value (e.g., from grid search or Stage 1)."""
        with torch.no_grad():
            x = 2.0 * H_value
            self.H_tilde.fill_(np.log(x / (1.0 - x)))

    def set_eta(self, eta_value: float):
        """Set eta to a specific value."""
        with torch.no_grad():
            self.eta_tilde.fill_(np.log(np.exp(eta_value) - 1.0))

    def set_alpha(self, alpha_values: torch.Tensor):
        """Set alpha to specific values by inverting softplus."""
        with torch.no_grad():
            for j in range(self.N):
                a = float(alpha_values[j])
                if a > 20.0:
                    self.alpha_tilde.data[j] = a
                else:
                    self.alpha_tilde.data[j] = a + np.log(-np.expm1(-a))

    def fix_alpha(self):
        """Freeze alpha (for level-shape separation calibration)."""
        self.alpha_tilde.requires_grad_(False)

    def unfix_alpha(self):
        """Unfreeze alpha (restore standard joint calibration)."""
        self.alpha_tilde.requires_grad_(True)

    def fix_eta(self):
        """Freeze eta."""
        self.eta_tilde.requires_grad_(False)

    def unfix_eta(self):
        """Unfreeze eta."""
        self.eta_tilde.requires_grad_(True)

    def fix_omega(self):
        """Freeze correlation matrix (ω angles)."""
        self.omega_tilde.requires_grad_(False)

    def unfix_omega(self):
        """Unfreeze correlation matrix (ω angles)."""
        self.omega_tilde.requires_grad_(True)

    def register_freeze_rho_hook(self):
        """Freeze inter-rate correlations ρ_{ij} while keeping ρ₀ free.

        Zeros gradients for omega_tilde[:, 1:] (columns that control ρ).
        Returns hook handle — call handle.remove() when done.
        """
        def _hook(grad):
            mask = grad.clone()
            mask[:, 1:] = 0.0
            return mask
        return self.omega_tilde.register_hook(_hook)

    def register_freeze_rho0_hook(self):
        """Freeze spot-vol correlations ρ₀ while keeping ρ_{ij} free.

        Zeros gradients for omega_tilde[:, 0] (first column that controls ρ₀).
        Returns hook handle — call handle.remove() when done.
        """
        def _hook(grad):
            mask = grad.clone()
            mask[:, 0] = 0.0
            return mask
        return self.omega_tilde.register_hook(_hook)

def volterra_gamma_integral(
    t: torch.Tensor,
    H: torch.Tensor,
    T_i_minus_1: float,
    T_i: float,
) -> torch.Tensor:
    """Closed-form ∫₀ᵗ (t−s)^{H−1/2} γ_i(s) ds."""
    Hp12 = H + 0.5
    Hp32 = H + 1.5
    delta_T = T_i - T_i_minus_1

    result = t ** Hp12 / Hp12

    a = torch.clamp(t - T_i_minus_1, min=0.0) ** Hp32
    b = torch.clamp(t - T_i, min=0.0) ** Hp32
    result = result - (a - b) / (Hp12 * Hp32 * delta_T)

    return result

def compute_xi_full(
    t_grid: torch.Tensor,
    j: int,
    H: torch.Tensor,
    eta: torch.Tensor,
    alpha: torch.Tensor,
    rho0: torch.Tensor,
    R: torch.Tensor,
    theta: float,
    N: int,
    precomputed_integrals: Optional[dict] = None,
) -> torch.Tensor:
    """Layer 2 (full): Compute ξ_j(t) on a time grid using Adachi eq. (p.22)."""
    alpha_j = alpha[j - 1]
    two_H = 2.0 * H
    sqrt_2H = torch.sqrt(two_H)

    exponent = eta ** 2 * t_grid ** two_H / 4.0

    for i in range(j + 1, N + 1):
        R_i = R[i]
        c_i = (theta * R_i / (1.0 + theta * R_i)
               * alpha[i - 1]
               * rho0[i - 1]
               * eta * sqrt_2H)

        if precomputed_integrals is not None and i in precomputed_integrals:
            integral = precomputed_integrals[i]
        else:
            T_i_minus_1 = float(i - 1) * theta
            T_i = float(i) * theta
            integral = volterra_gamma_integral(t_grid, H, T_i_minus_1, T_i)

        exponent = exponent - c_i * integral

    return alpha_j ** 2 * torch.exp(exponent)

def compute_effective_params(
    alpha: torch.Tensor,
    rho0: torch.Tensor,
    rho_matrix: torch.Tensor,
    swn: "SwaptionData",
) -> dict:
    """Layer 3 (core): Compute v(0) and ρ_eff from frozen weights."""
    I, J = swn.I, swn.J

    pi = swn.pi

    alpha_sub = alpha[I:J]
    w = pi * alpha_sub

    rho_sub = rho_matrix[I:J, I:J]
    v0 = w @ rho_sub @ w

    rho0_sub = rho0[I:J]
    rho_eff = (rho0_sub * w).sum() / torch.sqrt(v0 + 1e-30)

    rho_eff = torch.clamp(rho_eff, -1.0 + 1e-7, 1.0 - 1e-7)

    return {"v0": v0, "rho_eff": rho_eff}

def compute_v_curve(
    time_grid: torch.Tensor,
    swn: "SwaptionData",
    mkt: "MarketData",
    H: torch.Tensor,
    eta: torch.Tensor,
    alpha: torch.Tensor,
    rho0: torch.Tensor,
    rho_matrix: torch.Tensor,
    v0: torch.Tensor,
    mode: str = "full",
) -> torch.Tensor:
    """Layer 3 (variance curve): Compute v(t)/v(0) on the simulation grid."""
    M = time_grid.shape[0]

    if mode == "constant":
        return torch.ones(M, dtype=torch.float64, device=time_grid.device)

    if mode == "simplified":
        two_H = 2.0 * H
        return torch.exp(eta ** 2 * time_grid ** two_H / 4.0)

    if mode == "full":
        I, J = swn.I, swn.J
        pi = swn.pi
        n_rates = J - I

        precomputed_integrals = {}
        for i in range(1, mkt.N + 1):
            T_i_minus_1 = float(i - 1) * mkt.theta
            T_i = float(i) * mkt.theta
            precomputed_integrals[i] = volterra_gamma_integral(
                time_grid, H, T_i_minus_1, T_i,
            )

        xi_all = torch.stack([
            compute_xi_full(
                time_grid, j=I + 1 + k, H=H, eta=eta,
                alpha=alpha, rho0=rho0, R=mkt.R, theta=mkt.theta, N=mkt.N,
                precomputed_integrals=precomputed_integrals,
            )
            for k in range(n_rates)
        ], dim=0)

        rho_sub = rho_matrix[I:J, I:J]
        sqrt_xi = torch.sqrt(xi_all)

        w_t = pi.unsqueeze(1) * sqrt_xi

        v_t = (w_t * (rho_sub @ w_t)).sum(dim=0)

        return v_t / (v0 + 1e-30)

    raise ValueError(f"Unknown variance curve mode: {mode}")

def covariance_matrix_rBergomi(H: float, M: int, T: float) -> np.ndarray:
    """Compute the 2M×2M covariance matrix for exact rough Bergomi simulation."""
    h = T / M
    mat = np.zeros((2 * M, 2 * M))
    gamma_aux = 0.5 - H
    frac_aux = 1.0 / (1.0 - gamma_aux)

    for i in range(M):
        for j in range(M):
            mm = min(i, j) + 1
            mm_x = max(i, j) + 1

            mat[2*i, 2*j+1] = (
                ((i+1)*h)**(H+0.5) - ((i+1)*h - mm*h)**(H+0.5)
            ) / (H + 0.5)

            mat[2*i, 2*j] = (
                ((mm*h)**(2*H))
                * float(scipy_hyp2f1(1.0, gamma_aux, 2 - gamma_aux, mm / mm_x))
                * frac_aux
                * (mm / mm_x)**gamma_aux
            )

            mat[2*i+1, 2*j+1] = mm * h

            mat[2*i+1, 2*j] = (
                ((j+1)*h)**(H+0.5) - ((j+1)*h - mm*h)**(H+0.5)
            ) / (H + 0.5)

    return mat

def build_cholesky(H: float, M: int, T: float, device: str = "cpu") -> torch.Tensor:
    """Build and Cholesky-decompose the fBM-BM covariance matrix."""
    cov = covariance_matrix_rBergomi(H, M, T)
    L = np.linalg.cholesky(cov)
    return torch.tensor(L, dtype=torch.float64, device=torch.device(device))

def simulate_exact(
    S0: torch.Tensor,
    v0: torch.Tensor,
    eta: torch.Tensor,
    H: torch.Tensor,
    rho_eff: torch.Tensor,
    T: float,
    M: int,
    N_paths: int,
    cholesky_L: torch.Tensor,
    v_curve: Optional[torch.Tensor] = None,
    antithetic: bool = False,
) -> torch.Tensor:
    """Layer 4 (exact scheme): Simulate S*_T via Cholesky-based exact fBM sampling."""
    h = T / M
    device = S0.device
    sqrt_rho = torch.sqrt(1.0 - rho_eff**2)

    if antithetic:
        N_half = N_paths // 2
        Z_corr_half = torch.randn(2 * M, N_half, dtype=torch.float64, device=device)
        Z_indep_half = torch.randn(N_half, M, dtype=torch.float64, device=device) * np.sqrt(h)
        Z_corr = torch.cat([Z_corr_half, -Z_corr_half], dim=1)
        Z_indep = torch.cat([Z_indep_half, -Z_indep_half], dim=0)
    else:
        Z_corr = torch.randn(2 * M, N_paths, dtype=torch.float64, device=device)
        Z_indep = torch.randn(N_paths, M, dtype=torch.float64, device=device) * np.sqrt(h)

    prod_mat = (cholesky_L @ Z_corr).T

    log_S = torch.log(S0) * torch.ones(N_paths, dtype=torch.float64, device=device)

    two_H = 2.0 * H
    sqrt_2H = torch.sqrt(two_H)
    eta_sq = eta ** 2

    V_current = v0 * torch.ones(N_paths, dtype=torch.float64, device=device)

    prev_W = torch.zeros(N_paths, dtype=torch.float64, device=device)

    for i in range(M):
        t_next = (i + 1) * h

        fBm_next = prod_mat[:, 2 * i]

        W0_next = prod_mat[:, 2 * i + 1]

        dW0_i = W0_next - prev_W
        prev_W = W0_next

        brownian_incr = rho_eff * dW0_i + sqrt_rho * Z_indep[:, i]
        log_S = log_S - 0.5 * V_current * h + torch.sqrt(V_current) * brownian_incr

        correction = eta_sq / 2.0 * t_next ** two_H
        V_next = v0 * torch.exp(eta * sqrt_2H * fBm_next - correction)

        if v_curve is not None:
            V_next = V_next * v_curve[i]

        V_next = torch.clamp(V_next, min=0.0)
        V_current = V_next

    return torch.exp(log_S)

def simulate_approx(
    S0: torch.Tensor,
    v0: torch.Tensor,
    eta: torch.Tensor,
    H: torch.Tensor,
    rho_eff: torch.Tensor,
    T: float,
    M: int,
    N_paths: int,
    v_curve: Optional[torch.Tensor] = None,
    antithetic: bool = False,
) -> torch.Tensor:
    """Layer 4 (approximate scheme): Simulate S*_T via kernel discretization."""
    h = T / M
    device = S0.device
    sqrt_h = np.sqrt(h)
    sqrt_rho = torch.sqrt(1.0 - rho_eff**2)

    if antithetic:
        N_half = N_paths // 2
        dW0_half = torch.randn(N_half, M, dtype=torch.float64, device=device) * sqrt_h
        dW_perp_half = torch.randn(N_half, M, dtype=torch.float64, device=device) * sqrt_h
        dW0 = torch.cat([dW0_half, -dW0_half], dim=0)
        dW_perp = torch.cat([dW_perp_half, -dW_perp_half], dim=0)
    else:
        dW0 = torch.randn(N_paths, M, dtype=torch.float64, device=device) * sqrt_h
        dW_perp = torch.randn(N_paths, M, dtype=torch.float64, device=device) * sqrt_h

    log_S = torch.log(S0) * torch.ones(N_paths, dtype=torch.float64, device=device)
    two_H = 2.0 * H
    sqrt_2H = torch.sqrt(two_H)
    eta_sq = eta ** 2

    V_current = v0 * torch.ones(N_paths, dtype=torch.float64, device=device)

    for i in range(M):
        t_next = (i + 1) * h

        brownian_incr = rho_eff * dW0[:, i] + sqrt_rho * dW_perp[:, i]
        log_S = log_S - 0.5 * V_current * h + torch.sqrt(V_current) * brownian_incr

        lags = h * (i + 1 - torch.arange(0, i + 1, dtype=torch.float64, device=device))
        kernels = lags ** (H - 0.5)
        fBm_next = dW0[:, 0:i+1] @ kernels

        correction = eta_sq / 2.0 * t_next ** two_H
        V_next = v0 * torch.exp(eta * sqrt_2H * fBm_next - correction)

        if v_curve is not None:
            V_next = V_next * v_curve[i]

        V_next = torch.clamp(V_next, min=0.0)
        V_current = V_next

    return torch.exp(log_S)

def _gauss_legendre_nodes_weights(n_quad: int = 64):
    """Return Gauss-Legendre nodes and weights on [0, 1] (numpy, float64)."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_quad)
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    return nodes, weights

_GL64_NODES, _GL64_WEIGHTS = _gauss_legendre_nodes_weights(64)

def build_hybrid_covariance(
    alpha: torch.Tensor,
    h: float,
    kappa: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Build the (κ+1)×(κ+1) covariance matrix Σ for the BLP hybrid scheme."""
    dim = kappa + 1
    Sigma = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    u = torch.tensor(_GL64_NODES, dtype=torch.float64, device=device)
    w = torch.tensor(_GL64_WEIGHTS, dtype=torch.float64, device=device)

    alpha1 = alpha + 1.0
    two_alpha1 = 2.0 * alpha + 1.0
    h_t = torch.tensor(h, dtype=torch.float64, device=device)

    Sigma[0, 0] = h_t

    for j in range(1, dim):
        jf = float(j)

        j_t = torch.tensor(jf, dtype=torch.float64, device=device)
        jm1_t = torch.tensor(jf - 1.0, dtype=torch.float64, device=device)

        cov_0j = (h_t ** alpha1 / alpha1) * (j_t ** alpha1 - jm1_t ** alpha1)
        Sigma[0, j] = cov_0j
        Sigma[j, 0] = cov_0j

        for k in range(j, dim):
            kf = float(k)
            k_t = torch.tensor(kf, dtype=torch.float64, device=device)
            integrand = (j_t - u) ** alpha * (k_t - u) ** alpha
            integral = (w * integrand).sum()
            cov_jk = h_t ** two_alpha1 * integral
            Sigma[j, k] = cov_jk
            if k != j:
                Sigma[k, j] = cov_jk

    return Sigma

def compute_bstar_weights(
    alpha: torch.Tensor,
    h: float,
    kappa: int,
    M: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute optimal far-field kernel weights for the BLP hybrid scheme."""
    h_t = torch.tensor(h, dtype=torch.float64, device=device)
    alpha1 = alpha + 1.0
    weights = torch.zeros(M, dtype=torch.float64, device=device)

    k_vals = torch.arange(kappa + 1, M + 1, dtype=torch.float64, device=device)
    km1_vals = k_vals - 1.0

    bstar_alpha = (k_vals ** alpha1 - km1_vals ** alpha1) / alpha1

    far_w = bstar_alpha * h_t ** alpha
    weights[kappa:] = far_w

    return weights

def simulate_hybrid(
    S0: torch.Tensor,
    v0: torch.Tensor,
    eta: torch.Tensor,
    H: torch.Tensor,
    rho_eff: torch.Tensor,
    T: float,
    M: int,
    N_paths: int,
    v_curve: Optional[torch.Tensor] = None,
    kappa: int = 2,
    antithetic: bool = False,
) -> torch.Tensor:
    """Layer 4 (BLP hybrid scheme): Simulate S*_T with H fully differentiable."""
    h = T / M
    device = S0.device
    alpha = H - 0.5
    two_H = 2.0 * H
    sqrt_2H = torch.sqrt(two_H)
    eta_sq = eta ** 2
    sqrt_rho = torch.sqrt(1.0 - rho_eff ** 2)

    Sigma = build_hybrid_covariance(alpha, h, kappa, device=device)
    jitter = 1e-12
    L_near = None
    for _attempt in range(5):
        try:
            Sigma_reg = Sigma + jitter * torch.eye(kappa + 1, dtype=torch.float64, device=device)
            L_near = torch.linalg.cholesky(Sigma_reg)
            break
        except RuntimeError:
            jitter *= 100
    if L_near is None:
        raise RuntimeError(
            f"BLP hybrid Cholesky failed after 5 attempts "
            f"(α={alpha.item():.4f}, h={h:.6f}, κ={kappa})"
        )

    far_weights = compute_bstar_weights(alpha, h, kappa, M, device=device)

    if antithetic:
        N_half = N_paths // 2
        Z_half = torch.randn(N_half, M, kappa + 1, dtype=torch.float64, device=device)
        Z = torch.cat([Z_half, -Z_half], dim=0)
        dW_perp_half = torch.randn(N_half, M, dtype=torch.float64, device=device) * np.sqrt(h)
        dW_perp = torch.cat([dW_perp_half, -dW_perp_half], dim=0)
    else:
        Z = torch.randn(N_paths, M, kappa + 1, dtype=torch.float64, device=device)
        dW_perp = torch.randn(N_paths, M, dtype=torch.float64, device=device) * np.sqrt(h)

    W_vec = Z @ L_near.T

    dW0 = W_vec[:, :, 0]
    W_near = W_vec[:, :, 1:]

    log_S = torch.log(S0) * torch.ones(N_paths, dtype=torch.float64, device=device)
    V_current = v0 * torch.ones(N_paths, dtype=torch.float64, device=device)

    for i in range(M):
        t_next = (i + 1) * h

        brownian_incr = rho_eff * dW0[:, i] + sqrt_rho * dW_perp[:, i]
        log_S = log_S - 0.5 * V_current * h + torch.sqrt(V_current) * brownian_incr

        fBm_val = torch.zeros(N_paths, dtype=torch.float64, device=device)

        n_near = min(i + 1, kappa)
        for k in range(1, n_near + 1):
            fBm_val = fBm_val + W_near[:, i + 1 - k, k - 1]

        n_far = i + 1 - kappa
        if n_far > 0:

            fw = far_weights[kappa: kappa + n_far]
            fBm_val = fBm_val + dW0[:, 0:n_far] @ fw.flip(0)

        correction = eta_sq / 2.0 * t_next ** two_H
        V_next = v0 * torch.exp(eta * sqrt_2H * fBm_val - correction)

        if v_curve is not None:
            V_next = V_next * v_curve[i]

        V_next = torch.clamp(V_next, min=0.0)
        V_current = V_next

    return torch.exp(log_S)

def simulate_swaption(
    params: "MappedRoughSABRParams",
    swn: "SwaptionData",
    mkt: "MarketData",
    N_paths: int = 10000,
    M: int = 50,
    use_exact: bool = True,
    variance_curve_mode: str = "simplified",
    cholesky_cache: Optional[dict] = None,
    scheme: Optional[str] = None,
    hybrid_kappa: int = 2,
    antithetic: bool = False,
) -> torch.Tensor:
    """End-to-end simulation: Layers 0-4 combined."""
    if scheme is None:
        scheme = "exact" if use_exact else "approx"

    p = params()
    T = swn.expiry_years

    eff = compute_effective_params(p["alpha"], p["rho0"], p["rho"], swn)

    h = T / M
    time_grid = torch.arange(1, M + 1, dtype=torch.float64,
                             device=params._device) * h
    v_curve = compute_v_curve(
        time_grid, swn, mkt, H=p["H"], eta=p["eta"],
        alpha=p["alpha"], rho0=p["rho0"], rho_matrix=p["rho"],
        v0=eff["v0"], mode=variance_curve_mode,
    )

    if scheme == "exact":
        H_val = p["H"].detach().item()
        cache_key = (round(H_val, 8), T, M)

        if cholesky_cache is not None and cache_key in cholesky_cache:
            L = cholesky_cache[cache_key]
        else:
            L = build_cholesky(H_val, M, T, device=params._device)
            if cholesky_cache is not None:
                cholesky_cache[cache_key] = L

        S_T = simulate_exact(
            S0=swn.S0, v0=eff["v0"], eta=p["eta"],
            H=p["H"], rho_eff=eff["rho_eff"],
            T=T, M=M, N_paths=N_paths, cholesky_L=L,
            v_curve=v_curve, antithetic=antithetic,
        )
    elif scheme == "hybrid":
        S_T = simulate_hybrid(
            S0=swn.S0, v0=eff["v0"], eta=p["eta"],
            H=p["H"], rho_eff=eff["rho_eff"],
            T=T, M=M, N_paths=N_paths,
            v_curve=v_curve, kappa=hybrid_kappa,
            antithetic=antithetic,
        )
    else:
        S_T = simulate_approx(
            S0=swn.S0, v0=eff["v0"], eta=p["eta"],
            H=p["H"], rho_eff=eff["rho_eff"],
            T=T, M=M, N_paths=N_paths,
            v_curve=v_curve, antithetic=antithetic,
        )

    return S_T

def compute_swaption_prices(
    S_T: torch.Tensor,
    swn: "SwaptionData",
) -> torch.Tensor:
    """Layer 5: Compute Monte Carlo swaption prices for all strikes."""
    S_T_col = S_T.unsqueeze(1)
    K = swn.strikes.unsqueeze(0)

    diff = S_T_col - K

    sign = torch.where(swn.is_call.unsqueeze(0), torch.ones_like(diff), -torch.ones_like(diff))
    payoff = torch.clamp(sign * diff, min=0.0)

    mc_price = swn.A0 * payoff.mean(dim=0)

    return mc_price

def compute_vbar(
    T: float,
    v0: torch.Tensor,
    H: torch.Tensor,
    eta: torch.Tensor,
    mode: str = "simplified",
    swn: "SwaptionData" = None,
    mkt: "MarketData" = None,
    alpha: torch.Tensor = None,
    rho0: torch.Tensor = None,
    rho_matrix: torch.Tensor = None,
    n_quad: int = 50,
) -> torch.Tensor:
    """Compute the time-averaged forward variance v̄(T) = ∫₀¹ v(Ts) ds."""
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)
    s_np = 0.5 * (nodes_np + 1.0)
    w_np = 0.5 * weights_np

    device = v0.device if hasattr(v0, 'device') else 'cpu'
    s = torch.tensor(s_np, dtype=torch.float64, device=device)
    w = torch.tensor(w_np, dtype=torch.float64, device=device)
    t_points = T * s

    if mode == "simplified":
        two_H = 2.0 * H
        ratios = torch.exp(eta ** 2 * t_points ** two_H / 4.0)
        vbar = v0 * (w * ratios).sum()
    elif mode == "full":
        assert swn is not None and mkt is not None
        v_curve = compute_v_curve(
            t_points, swn, mkt, H=H, eta=eta,
            alpha=alpha, rho0=rho0, rho_matrix=rho_matrix,
            v0=v0, mode="full",
        )
        vbar = v0 * (w * v_curve).sum()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return vbar

def match_alpha_atm(
    swn: "SwaptionData",
    mkt: "MarketData",
    H: torch.Tensor,
    eta: torch.Tensor,
    rho0: torch.Tensor,
    rho_matrix: torch.Tensor,
    alpha_other: torch.Tensor,
    variance_curve_mode: str = "simplified",
    method: str = "formula",
    N_paths: int = 100_000,
    M: int = 100,
    seed: int = 42,
) -> torch.Tensor:
    """Find α_j for rate j such that model ATM IV = market ATM IV."""
    I, J = swn.I, swn.J
    n_rates = J - I

    atm_mask = (swn.strikes - swn.S0).abs() < ATM_TOL
    if atm_mask.sum() == 0:
        atm_idx = (swn.strikes - swn.S0).abs().argmin()
        atm_mask = torch.zeros(swn.n_strikes, dtype=torch.bool)
        atm_mask[atm_idx] = True
    sigma_atm_mkt = swn.ivs_black[atm_mask][0]

    pi = swn.pi

    if method == "formula":

        if n_rates == 1:
            j_idx = I
            pi_j = pi[0]

            n_quad = 50
            nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)
            s_np = 0.5 * (nodes_np + 1.0)
            w_np = 0.5 * weights_np
            s = torch.tensor(s_np, dtype=torch.float64)
            w = torch.tensor(w_np, dtype=torch.float64)

            T = swn.expiry_years
            two_H = 2.0 * H

            if variance_curve_mode == "simplified":
                G = (w * torch.exp(eta**2 * (T * s)**two_H / 4.0)).sum()
            else:
                c = _match_alpha_brent(
                    swn, mkt, H, eta, rho0, rho_matrix,
                    alpha_other, sigma_atm_mkt, variance_curve_mode,
                    method="formula",
                )
                return torch.clamp(alpha_other[j_idx] * c, min=1e-6)

            alpha_j = sigma_atm_mkt / (pi_j * torch.sqrt(G) + 1e-30)
            return torch.clamp(alpha_j, min=1e-6)

        else:
            c = _match_alpha_brent(
                swn, mkt, H, eta, rho0, rho_matrix,
                alpha_other, sigma_atm_mkt, variance_curve_mode,
                method="formula",
            )
            return c

    elif method == "mc":
        c = _match_alpha_brent(
            swn, mkt, H, eta, rho0, rho_matrix,
            alpha_other, sigma_atm_mkt, variance_curve_mode,
            method="mc", N_paths=N_paths, M=M, seed=seed,
        )
        if n_rates == 1:
            return torch.clamp(alpha_other[I] * c, min=1e-6)
        return c

    else:
        raise ValueError(f"Unknown ATM matching method: {method}")

def _match_alpha_brent(
    swn, mkt, H, eta, rho0, rho_matrix,
    alpha_other, sigma_atm_mkt, variance_curve_mode,
    method="formula", N_paths=100_000, M=100, seed=42,
):
    """Internal: Brent root-finding for α matching."""
    I, J = swn.I, swn.J
    T = swn.expiry_years
    pi = swn.pi

    alpha_base = alpha_other[I:J].detach().clone()

    cholesky_L = None
    if method == "mc":
        H_val = H.detach().item()
        device = str(swn.S0.device)
        cholesky_L = build_cholesky(H_val, M, T, device=device)

    def objective(log_c):
        c = np.exp(log_c)
        alpha_trial = alpha_other.detach().clone()
        alpha_trial[I:J] = alpha_base * c

        if method == "formula":
            w = pi * alpha_trial[I:J]
            rho_sub = rho_matrix[I:J, I:J].detach()
            v0_trial = (w @ rho_sub @ w).item()

            vbar_trial = compute_vbar(
                T, torch.tensor(v0_trial), H.detach(), eta.detach(),
                mode=variance_curve_mode,
                swn=swn, mkt=mkt,
                alpha=alpha_trial, rho0=rho0.detach(),
                rho_matrix=rho_matrix.detach(),
            ).item()

            sigma_model = np.sqrt(max(vbar_trial, 1e-30))

        elif method == "mc":
            alpha_t = alpha_trial.requires_grad_(False)
            torch.manual_seed(seed)

            w = pi * alpha_t[I:J]
            rho_sub = rho_matrix[I:J, I:J].detach()
            v0_t = w @ rho_sub @ w
            rho0_sub = rho0[I:J].detach()
            rho_eff_t = (rho0_sub * w).sum() / torch.sqrt(v0_t + 1e-30)
            rho_eff_t = torch.clamp(rho_eff_t, -1.0 + 1e-7, 1.0 - 1e-7)

            h = T / M
            time_grid = torch.arange(1, M + 1, dtype=torch.float64) * h
            v_curve = compute_v_curve(
                time_grid, swn, mkt, H=H.detach(), eta=eta.detach(),
                alpha=alpha_t, rho0=rho0.detach(), rho_matrix=rho_matrix.detach(),
                v0=v0_t, mode=variance_curve_mode,
            )
            S_T = simulate_exact(
                S0=swn.S0, v0=v0_t, eta=eta.detach(),
                H=H.detach(), rho_eff=rho_eff_t,
                T=T, M=M, N_paths=N_paths, cholesky_L=cholesky_L,
                v_curve=v_curve,
            )
            atm_payoff = torch.clamp(S_T - swn.S0, min=0.0).mean()
            atm_price = (swn.A0 * atm_payoff).item()
            sigma_model = black_iv(
                atm_price, swn.S0.item(), swn.S0.item(), T,
                annuity=swn.A0.item(), is_call=True,
            )
            if np.isnan(sigma_model):
                sigma_model = 0.3

        return sigma_model - sigma_atm_mkt.item()

    try:
        log_c_opt = brentq(objective, -3.0, 3.0, xtol=1e-8, maxiter=100)
        c_opt = np.exp(log_c_opt)
    except ValueError:
        c_opt = 1.0

    return torch.tensor(c_opt, dtype=torch.float64)

def match_all_alphas(
    mkt: "MarketData",
    H: torch.Tensor,
    eta: torch.Tensor,
    rho0: torch.Tensor,
    rho_matrix: torch.Tensor,
    alpha_init: torch.Tensor,
    smile_keys: Optional[list] = None,
    variance_curve_mode: str = "simplified",
    method: str = "formula",
    N_paths: int = 100_000,
    M: int = 100,
    seed: int = 42,
) -> torch.Tensor:
    """Match α_i for all smile tenors via ATM root-finding."""
    alpha = alpha_init.detach().clone()

    if smile_keys is None:
        smile_keys = sorted([k for k in mkt.swaptions.keys() if k[1] == 1])

    for key in smile_keys:
        if key not in mkt.swaptions:
            continue
        swn = mkt.swaptions[key]
        I, J = swn.I, swn.J

        alpha_j = match_alpha_atm(
            swn, mkt, H, eta, rho0, rho_matrix, alpha,
            variance_curve_mode=variance_curve_mode,
            method=method,
            N_paths=N_paths,
            M=M,
            seed=seed,
        )

        if J - I == 1:
            alpha[I] = alpha_j
        else:
            alpha[I:J] = alpha[I:J] * alpha_j

    return alpha

def _interpolate_alpha(alpha, matched_indices):
    """Fill unmatched alpha by linear interpolation, flat extrapolation."""
    alpha = alpha.clone()
    N = alpha.shape[0]
    anchors = sorted(matched_indices)

    for j in range(N):
        if j in anchors:
            continue
        if j <= anchors[0]:
            alpha[j] = alpha[anchors[0]]
        elif j >= anchors[-1]:
            alpha[j] = alpha[anchors[-1]]
        else:
            lo = max(a for a in anchors if a < j)
            hi = min(a for a in anchors if a > j)
            frac = (j - lo) / (hi - lo)
            alpha[j] = (1 - frac) * alpha[lo] + frac * alpha[hi]

    return alpha

def rematch_alpha_to_atm(
    params,
    mkt,
    variance_curve_mode="simplified",
):
    """Re-match alpha to pin model ATM IVs to market ATM IVs."""
    with torch.no_grad():
        p = params()

        smile_keys_1y = sorted([k for k in mkt.swaptions.keys() if k[1] == 1])

        alpha_matched = match_all_alphas(
            mkt, p["H"], p["eta"], p["rho0"], p["rho"],
            alpha_init=p["alpha"],
            smile_keys=smile_keys_1y,
            variance_curve_mode=variance_curve_mode,
            method="formula",
        )

        matched_indices = []
        for key in smile_keys_1y:
            if key in mkt.swaptions:
                swn = mkt.swaptions[key]
                if swn.J - swn.I == 1:
                    matched_indices.append(swn.I)

        alpha_final = _interpolate_alpha(alpha_matched, matched_indices)

        params.set_alpha(alpha_final)

def black_price_torch(S0, K, T, sigma, A0, is_call):
    """Black (1976) price in PyTorch (differentiable)."""
    sqrt_T = torch.sqrt(T)
    d1 = (torch.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    Phi = lambda x: 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

    call_price = A0 * (S0 * Phi(d1) - K * Phi(d2))
    put_price = A0 * (K * Phi(-d2) - S0 * Phi(-d1))

    return torch.where(is_call, call_price, put_price)

def mc_prices_to_black_iv(
    mc_prices: torch.Tensor,
    swn: "SwaptionData",
) -> torch.Tensor:
    """Invert MC prices to Black IVs via ``black_iv`` (non-differentiable)."""
    ivs = torch.full_like(mc_prices, float('nan'))
    S0 = swn.S0.item()
    A0 = swn.A0.item()
    T = swn.expiry_years

    for i in range(swn.n_strikes):
        target = mc_prices[i].item()
        K = swn.strikes[i].item()
        ic = bool(swn.is_call[i].item())

        iv = black_iv(target, S0, K, T, annuity=A0, is_call=ic)
        if not np.isnan(iv):
            ivs[i] = iv

    return ivs

def compute_loss_vegaweighted(
    mc_prices: torch.Tensor,
    swn: "SwaptionData",
) -> torch.Tensor:
    """Layer 6a: Vega-weighted price loss for a single swaption."""
    residuals = (mc_prices - swn.target_prices) / (swn.vegas + 1e-30)
    return (residuals ** 2).sum()

def compute_loss_ivspace(
    mc_prices: torch.Tensor,
    swn: "SwaptionData",
) -> torch.Tensor:
    """Layer 6b: IV-space loss for a single swaption."""
    model_ivs = mc_prices_to_black_iv(mc_prices.detach(), swn)
    valid = ~torch.isnan(model_ivs)
    if valid.sum() == 0:
        return torch.tensor(0.0, dtype=torch.float64)
    diff = model_ivs[valid] - swn.ivs_black[valid]
    return (diff ** 2).sum()

def compute_total_loss(
    params: "MappedRoughSABRParams",
    mkt: "MarketData",
    N_paths: int = 10000,
    M: int = 50,
    use_exact: bool = True,
    variance_curve_mode: str = "simplified",
    cholesky_cache: Optional[dict] = None,
    seed: Optional[int] = None,
    swaption_keys: Optional[list] = None,
    scheme: Optional[str] = None,
    hybrid_kappa: int = 2,
    compute_diagnostics: bool = True,
    atm_only: bool = False,
) -> dict:
    """Layer 6: Compute the total calibration loss over all swaptions."""
    keys = swaption_keys if swaption_keys is not None else list(mkt.swaptions.keys())

    total_loss = torch.tensor(0.0, dtype=torch.float64, device=params._device)
    total_loss_iv = 0.0
    n_valid_strikes = 0
    per_swaption = {}

    for key in keys:
        swn = mkt.swaptions[key]

        if seed is not None:
            torch.manual_seed(seed + _stable_key_hash(key))

        S_T = simulate_swaption(
            params, swn, mkt,
            N_paths=N_paths, M=M,
            use_exact=use_exact,
            variance_curve_mode=variance_curve_mode,
            cholesky_cache=cholesky_cache,
            scheme=scheme,
            hybrid_kappa=hybrid_kappa,
        )

        mc_prices = compute_swaption_prices(S_T, swn)

        if atm_only:
            atm_mask = (swn.strikes - swn.S0).abs() < ATM_TOL
            if atm_mask.sum() > 0:
                loss_vw = ((mc_prices[atm_mask] - swn.target_prices[atm_mask])
                           / (swn.vegas[atm_mask] + 1e-30)) ** 2
                loss_vw = loss_vw.sum()
            else:
                loss_vw = torch.tensor(0.0, dtype=torch.float64,
                                       device=params._device)
        else:
            loss_vw = compute_loss_vegaweighted(mc_prices, swn)
        total_loss = total_loss + loss_vw

        record = {
            "loss": loss_vw.item(),
            "mc_prices": mc_prices.detach(),
        }

        if compute_diagnostics:
            model_ivs = mc_prices_to_black_iv(mc_prices.detach(), swn)
            valid = ~torch.isnan(model_ivs)
            n_valid = int(valid.sum().item())
            if n_valid > 0:
                diff = model_ivs[valid] - swn.ivs_black[valid]
                loss_iv_val = (diff ** 2).sum().item()
            else:
                loss_iv_val = 0.0
            total_loss_iv += loss_iv_val
            n_valid_strikes += n_valid
            record["loss_iv"] = loss_iv_val
            record["model_ivs"] = model_ivs

        per_swaption[key] = record

    return {
        "loss": total_loss,
        "loss_iv": total_loss_iv,
        "n_valid_strikes": n_valid_strikes,
        "per_swaption": per_swaption,
    }

class SteepCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with power-steepened progress."""
    def __init__(self, optimizer, T_max, eta_min=1e-6, power=0.5,
                 last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        T = self.T_max
        p = self.power
        progress = min(max(t / T, 0.0), 1.0)
        steepened = progress ** p
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * steepened))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]

def calibrate(
    params: "MappedRoughSABRParams",
    mkt: "MarketData",
    n_iterations: int = 200,
    lr: float = 1e-3,
    N_paths: int = 10000,
    M: int = 50,
    use_exact: bool = False,
    variance_curve_mode: str = "simplified",
    use_crn: bool = True,
    crn_seed: int = 42,
    log_every: int = 20,
    swaption_keys: Optional[list] = None,
    scheduler_type: str = "plateau",
    scheduler_patience: int = 50,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    warmup_steps: int = 0,
    cosine_power: float = 1.0,
    scheme: Optional[str] = None,
    hybrid_kappa: int = 2,
    early_stop_patience: Optional[int] = None,
    early_stop_tol: float = 1e-4,
    H_lr_factor: float = 1.0,
    grad_clip_norm: float = 10.0,
    atm_only: bool = False,
) -> dict:
    """Layer 7: Run the calibration loop."""
    effective_scheme = scheme if scheme is not None else ("exact" if use_exact else "approx")

    if H_lr_factor < 1.0 and params.H_tilde.requires_grad:
        other_params = [p for n, p in params.named_parameters()
                        if n != "H_tilde" and p.requires_grad]
        param_groups = [
            {"params": [params.H_tilde], "lr": lr * H_lr_factor},
            {"params": other_params, "lr": lr},
        ]
        optimizer = torch.optim.Adam(param_groups, lr=lr)
    else:
        optimizer = torch.optim.Adam(params.parameters(), lr=lr)

    if scheduler_type == "cosine":
        cosine_steps = max(1, n_iterations - warmup_steps)
        cosine_scheduler = SteepCosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=min_lr,
            power=cosine_power,
        )
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=max(min_lr / lr, 1e-8),
                end_factor=1.0, total_iters=warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = cosine_scheduler
        scheduler_is_plateau = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=scheduler_patience,
            factor=scheduler_factor, min_lr=min_lr,
        )
        scheduler_is_plateau = True

    cholesky_cache = {} if effective_scheme == "exact" else None
    history = []
    best_loss = float('inf')
    best_state = None
    steps_without_improvement = 0

    for step in range(n_iterations):
        optimizer.zero_grad()

        seed = crn_seed + step if use_crn else None

        is_log_step = (step % log_every == 0) or (step == n_iterations - 1)

        result = compute_total_loss(
            params, mkt,
            N_paths=N_paths, M=M,
            use_exact=use_exact,
            variance_curve_mode=variance_curve_mode,
            cholesky_cache=cholesky_cache,
            seed=seed,
            swaption_keys=swaption_keys,
            scheme=scheme,
            hybrid_kappa=hybrid_kappa,
            compute_diagnostics=is_log_step,
            atm_only=atm_only,
        )

        loss = result["loss"]
        loss.backward()

        grad_norms = {}
        for name, param in params.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
            else:
                grad_norms[name] = 0.0

        torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        if scheduler_is_plateau:
            scheduler.step(loss.item())
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        if loss.item() < best_loss * (1.0 - early_stop_tol):
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in params.state_dict().items()}
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        record = {
            "step": step,
            "loss": loss.item(),
            "loss_iv": result["loss_iv"],
            "lr": current_lr,
            "grad_norms": grad_norms,
        }
        history.append(record)

        if step % log_every == 0 or step == n_iterations - 1:
            with torch.no_grad():
                p = params()
                H_val = p["H"].item()
                eta_val = p["eta"].item()
                kappa_adachi = eta_val * np.sqrt(2.0 * H_val)
            if result["loss_iv"] > 0 and result["n_valid_strikes"] > 0:
                rmse_iv = np.sqrt(result["loss_iv"] / result["n_valid_strikes"]) * 100
                rmse_str = f"RMSE_IV={rmse_iv:.3f}%"
            else:
                rmse_str = "RMSE_IV=n/a"
            print(f"  [{step:4d}/{n_iterations}]  "
                  f"loss={loss.item():.6f}  "
                  f"{rmse_str}  "
                  f"H={H_val:.4f}  η={eta_val:.4f}  κ={kappa_adachi:.4f}  "
                  f"lr={current_lr:.2e}")

        if early_stop_patience is not None and steps_without_improvement >= early_stop_patience:
            print(f"  Early stopping at step {step}: no improvement "
                  f"for {early_stop_patience} steps (best loss={best_loss:.6f})")
            break

        if effective_scheme == "exact" and step > 0 and step % 50 == 0:
            cholesky_cache.clear()

    return {
        "history": history,
        "best_state": best_state,
        "best_loss": best_loss,
    }

def calibrate_two_stage(
    params: "MappedRoughSABRParams",
    mkt: "MarketData",
    stage1_iterations: int = 300,
    stage1_lr: float = 5e-3,
    stage1_N_paths: int = 5000,
    stage1_M: int = 30,
    stage1_keys: Optional[list] = None,
    stage2_iterations: int = 200,
    stage2_lr: float = 1e-3,
    stage2_N_paths: int = 20000,
    stage2_M: int = 50,
    stage2_variance_mode: str = "full",
    stage2_keys: Optional[list] = None,
    stage2_scheduler: str = "cosine",
    stage2_warmup_steps: int = 20,
    stage2_cosine_power: float = 0.5,
    use_crn: bool = True,
    crn_seed: int = 42,
    log_every: int = 20,
    early_stop_patience: Optional[int] = None,
    early_stop_tol: float = 1e-4,
) -> dict:
    """Two-stage calibration following Adachi et al."""
    print("=" * 60)
    print("STAGE 1: Approximate scheme (H differentiable)")
    print("=" * 60)

    params.unfix_H()
    stage1_result = calibrate(
        params, mkt,
        n_iterations=stage1_iterations, lr=stage1_lr,
        N_paths=stage1_N_paths, M=stage1_M,
        use_exact=False, variance_curve_mode="simplified",
        use_crn=use_crn, crn_seed=crn_seed,
        log_every=log_every,
        swaption_keys=stage1_keys,
        scheduler_type="plateau",
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
    )

    if stage1_result["best_state"] is not None:
        params.load_state_dict(stage1_result["best_state"])

    with torch.no_grad():
        H_calibrated = params.get_H().item()
        eta_calibrated = params.get_eta().item()
        kappa_adachi = eta_calibrated * np.sqrt(2.0 * H_calibrated)
    print(f"\nStage 1 complete. H = {H_calibrated:.4f},  "
          f"η = {eta_calibrated:.4f},  κ = {kappa_adachi:.4f}")

    print("\n" + "=" * 60)
    print(f"STAGE 2: Exact Cholesky scheme (H fixed at {H_calibrated:.4f})")
    print("=" * 60)

    params.fix_H()
    stage2_result = calibrate(
        params, mkt,
        n_iterations=stage2_iterations, lr=stage2_lr,
        N_paths=stage2_N_paths, M=stage2_M,
        use_exact=True, variance_curve_mode=stage2_variance_mode,
        use_crn=use_crn, crn_seed=crn_seed + 10000,
        log_every=log_every,
        swaption_keys=stage2_keys,
        scheduler_type=stage2_scheduler,
        warmup_steps=stage2_warmup_steps,
        cosine_power=stage2_cosine_power,
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
    )

    if stage2_result["best_state"] is not None:
        params.load_state_dict(stage2_result["best_state"])

    print(f"\nStage 2 complete.")
    print(params.summary())

    return {
        "stage1": stage1_result,
        "stage2": stage2_result,
        "H_calibrated": H_calibrated,
    }

def print_market_summary(mkt: MarketData):
    """Print a summary of the loaded market data."""
    print("=" * 60)
    print("Market Data Summary")
    print("=" * 60)
    print(f"N = {mkt.N} (annual grid T_0=0,...,T_{mkt.N})")
    print(f"Device: {mkt.device}")
    print(f"Swaptions: {len(mkt.swaptions)}")
    print(f"(I,J) groups: {len(mkt.groups)}")
    print()

    print("Discount factors:")
    for j in range(min(mkt.N + 1, 12)):
        print(f"  P(T_{j}) = {mkt.P[j].item():.6f}", end="")
        if j > 0:
            print(f"    R_{j} = {mkt.R[j].item()*100:.3f}%", end="")
        print()

    print()
    print(f"{'Key':>12s}  {'I':>2s}  {'J':>2s}  {'S0(%)':>7s}  "
          f"{'ATM IV':>7s}  {'#K':>3s}  {'sum(pi)':>8s}")
    print("-" * 60)
    for key in sorted(mkt.swaptions.keys()):
        swn = mkt.swaptions[key]
        atm_mask = (swn.strikes - swn.S0).abs() < ATM_TOL
        atm_iv = swn.ivs_black[atm_mask]
        atm_str = f"{atm_iv[0].item()*100:.2f}%" if len(atm_iv) > 0 else "n/a"
        print(f"  {key[0]:.0f}Y x {key[1]:.0f}Y"
              f"  {swn.I:2d}  {swn.J:2d}"
              f"  {swn.S0.item()*100:7.3f}"
              f"  {atm_str:>7s}"
              f"  {swn.n_strikes:3d}"
              f"  {swn.pi.sum().item():8.4f}")

def print_smile(mkt: MarketData, key: tuple):
    """Print the smile for a specific swaption."""
    swn = mkt.swaptions[key]
    print(f"\n{key[0]:.0f}Y x {key[1]:.0f}Y  (I={swn.I}, J={swn.J})")
    print(f"S0 = {swn.S0.item()*100:.4f}%,  A0 = {swn.A0.item():.6f}")
    print(f"Pi = {swn.Pi.numpy()}")
    print(f"pi = {swn.pi.numpy()}")
    print(f"\n{'Strike(%)':>10s}  {'Offset(bp)':>10s}  {'IV(%)':>8s}  "
          f"{'Price':>10s}  {'Vega':>10s}  {'C/P':>4s}")
    print("-" * 60)
    for i in range(swn.n_strikes):
        K = swn.strikes[i].item()
        offset = (K - swn.S0.item()) * 10000
        iv = swn.ivs_black[i].item()
        price = swn.target_prices[i].item()
        vega = swn.vegas[i].item()
        cp = "C" if swn.is_call[i].item() else "P"
        print(f"  {K*100:8.4f}  {offset:+10.0f}  {iv*100:8.2f}  "
              f"{price:10.6f}  {vega:10.6f}  {cp:>4s}")

def compute_model_smile(
    params: "MappedRoughSABRParams",
    swn: "SwaptionData",
    mkt: "MarketData",
    variance_curve_mode: str = "simplified",
    N_paths: int = 50000,
    M: int = 100,
    seed: int = 42,
    cholesky_cache: Optional[dict] = None,
    scheme: str = "exact",
    hybrid_kappa: int = 2,
) -> dict:
    """Compute the full model smile for a single swaption via Monte Carlo."""
    p = params()
    eff = compute_effective_params(p["alpha"], p["rho0"], p["rho"], swn)

    torch.manual_seed(seed)
    S_T = simulate_swaption(
        params, swn, mkt,
        N_paths=N_paths, M=M,
        use_exact=(scheme == "exact"),
        variance_curve_mode=variance_curve_mode,
        cholesky_cache=cholesky_cache,
        scheme=scheme,
        hybrid_kappa=hybrid_kappa,
    )
    mc_prices = compute_swaption_prices(S_T, swn)
    model_ivs = mc_prices_to_black_iv(mc_prices.detach(), swn)

    market_ivs = swn.ivs_black
    iv_errors = model_ivs - market_ivs

    return {
        "model_ivs": model_ivs.detach(),
        "model_prices": mc_prices.detach(),
        "market_ivs": market_ivs,
        "strikes": swn.strikes,
        "iv_errors": iv_errors.detach(),
        "S0": swn.S0.item(),
        "key": (swn.expiry_years, swn.J - swn.I),
    }

def print_calibration_report(
    params: "MappedRoughSABRParams",
    mkt: "MarketData",
    variance_curve_mode: str = "simplified",
    N_paths: int = 50000,
    M: int = 100,
    seed: int = 42,
    swaption_keys: Optional[list] = None,
    scheme: str = "exact",
    hybrid_kappa: int = 2,
) -> dict:
    """Print a comprehensive calibration report with per-swaption RMSE."""
    keys = swaption_keys or sorted(mkt.swaptions.keys())
    cholesky_cache = {}
    results = {}
    all_sq_errors = []

    print("=" * 72)
    print("CALIBRATION REPORT")
    print("=" * 72)

    with torch.no_grad():
        p = params()
        H_val = p['H'].item()
        eta_val = p['eta'].item()
        kappa_adachi = eta_val * np.sqrt(2.0 * H_val)
        print(f"H = {H_val:.4f},  η = {eta_val:.4f},  κ = {kappa_adachi:.4f}  (κ = η√(2H))")
        print(f"MC: {N_paths} paths, {M} steps,  variance curve: {variance_curve_mode}")
        print(f"Scheme: {scheme}" + (f" (κ_BLP={hybrid_kappa})" if scheme == "hybrid" else ""))
        print()

    print(f"{'Swaption':>10s}  {'RMSE(bp)':>10s}  {'MaxErr(bp)':>10s}  "
          f"{'ATM Err(bp)':>12s}  {'#K':>4s}")
    print("-" * 55)

    for key in keys:
        if key not in mkt.swaptions:
            continue
        swn = mkt.swaptions[key]

        res = compute_model_smile(
            params, swn, mkt,
            variance_curve_mode=variance_curve_mode,
            N_paths=N_paths, M=M, seed=seed,
            cholesky_cache=cholesky_cache,
            scheme=scheme,
            hybrid_kappa=hybrid_kappa,
        )
        results[key] = res

        valid = ~torch.isnan(res["iv_errors"])
        if valid.sum() == 0:
            print(f"  {key[0]:.0f}Y×{key[1]:.0f}Y  {'N/A':>10s}")
            continue

        errors_bp = res["iv_errors"][valid] * 10000
        rmse_bp = torch.sqrt((errors_bp ** 2).mean()).item()
        max_err_bp = errors_bp.abs().max().item()
        all_sq_errors.extend((errors_bp ** 2).tolist())

        atm_mask = (swn.strikes - swn.S0).abs() < ATM_TOL
        if atm_mask.sum() > 0 and not torch.isnan(res["iv_errors"][atm_mask][0]):
            atm_err_bp = res["iv_errors"][atm_mask][0].item() * 10000
            atm_str = f"{atm_err_bp:+12.1f}"
        else:
            atm_str = f"{'N/A':>12s}"

        print(f"  {key[0]:.0f}Y×{key[1]:.0f}Y  {rmse_bp:10.1f}  "
              f"{max_err_bp:10.1f}  {atm_str}  {swn.n_strikes:4d}")

    total_rmse = np.sqrt(np.mean(all_sq_errors)) if all_sq_errors else float('nan')
    print("-" * 55)
    print(f"  {'Total':>8s}  {total_rmse:10.1f} bp RMSE")

    return {
        "per_swaption": results,
        "total_rmse_bp": total_rmse,
    }

def print_smile_comparison(
    params: "MappedRoughSABRParams",
    swn: "SwaptionData",
    mkt: "MarketData",
    variance_curve_mode: str = "simplified",
    N_paths: int = 50000,
    M: int = 100,
    seed: int = 42,
    cholesky_cache: Optional[dict] = None,
    scheme: str = "exact",
    hybrid_kappa: int = 2,
):
    """Print a detailed smile comparison for one swaption (MC-based)."""
    res = compute_model_smile(
        params, swn, mkt,
        variance_curve_mode=variance_curve_mode,
        N_paths=N_paths, M=M, seed=seed,
        cholesky_cache=cholesky_cache,
        scheme=scheme,
        hybrid_kappa=hybrid_kappa,
    )

    key = res["key"]
    print(f"\nSmile: {key[0]:.0f}Y × {key[1]:.0f}Y   S₀ = {res['S0']*100:.4f}%")
    print(f"{'Strike(%)':>10s}  {'Offset(bp)':>10s}  {'Mkt IV(%)':>10s}  "
          f"{'Mod IV(%)':>10s}  {'Error(bp)':>10s}")
    print("-" * 58)

    for i in range(swn.n_strikes):
        K = swn.strikes[i].item()
        offset = (K - res["S0"]) * 10000
        mkt_iv = res["market_ivs"][i].item() * 100
        mod_iv = res["model_ivs"][i].item()
        if np.isnan(mod_iv):
            print(f"  {K*100:8.4f}  {offset:+10.0f}  {mkt_iv:10.2f}  {'NaN':>10s}  {'NaN':>10s}")
        else:
            err_bp = (mod_iv - res["market_ivs"][i].item()) * 10000
            print(f"  {K*100:8.4f}  {offset:+10.0f}  {mkt_iv:10.2f}  "
                  f"{mod_iv*100:10.2f}  {err_bp:+10.1f}")

def generate_smile_plot_data(
    params: "MappedRoughSABRParams",
    mkt: "MarketData",
    variance_curve_mode: str = "simplified",
    N_paths: int = 50000,
    M: int = 100,
    seed: int = 42,
    swaption_keys: Optional[list] = None,
) -> dict:
    """Generate data for smile plots (model vs market, MC-based)."""
    keys = swaption_keys or sorted(mkt.swaptions.keys())
    cholesky_cache = {}
    plot_data = {}

    for key in keys:
        if key not in mkt.swaptions:
            continue
        swn = mkt.swaptions[key]
        res = compute_model_smile(
            params, swn, mkt,
            variance_curve_mode=variance_curve_mode,
            N_paths=N_paths, M=M, seed=seed,
            cholesky_cache=cholesky_cache,
        )
        plot_data[key] = {
            "strikes_pct": (swn.strikes.numpy() * 100),
            "offsets_bp": ((swn.strikes - swn.S0).numpy() * 10000),
            "market_ivs_pct": (res["market_ivs"].numpy() * 100),
            "model_ivs_pct": (res["model_ivs"].numpy() * 100),
            "errors_bp": (res["iv_errors"].numpy() * 10000),
            "S0_pct": res["S0"] * 100,
        }

    return plot_data

if __name__ == "__main__":

    print("Loading market data...")
    mkt = load_market_data(
        "usd_swaption_data.pkl",
        subset="joint_all_smiles",
        date="2024-12-09",
        device="cpu",
    )
    print_market_summary(mkt)

    print("\n\nInitializing parameters...")
    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    p = params()

    print("\n" + "=" * 60)
    print("α ATM matching test (σ_ATM = √v̄(T))")
    print("=" * 60)

    alpha_before = p["alpha"].detach().clone()
    alpha_matched = match_all_alphas(
        mkt, p["H"], p["eta"], p["rho0"], p["rho"],
        alpha_init=p["alpha"],
        variance_curve_mode="simplified",
        method="formula",
    )

    print(f"  {'Rate':>6s}  {'α_init':>10s}  {'α_matched':>10s}  {'Ratio':>8s}")
    print(f"  {'-'*38}")
    one_y_keys = sorted([k for k in mkt.swaptions.keys() if k[1] == 1])
    for key in one_y_keys:
        swn = mkt.swaptions[key]
        j = swn.I
        print(f"  R_{j+1:>3d}  {alpha_before[j].item():10.4f}  "
              f"{alpha_matched[j].item():10.4f}  "
              f"{alpha_matched[j].item()/alpha_before[j].item():8.3f}")

    print("\n  Verification (ATM variance check: σ_ATM = √v̄ vs market):")
    for key in one_y_keys[:3]:
        swn = mkt.swaptions[key]
        eff2 = compute_effective_params(alpha_matched, p["rho0"], p["rho"], swn)
        vbar2 = compute_vbar(
            swn.expiry_years, eff2["v0"], p["H"], p["eta"],
            mode="simplified",
        )
        atm_iv_model = torch.sqrt(vbar2).item() * 100
        atm_mask = (swn.strikes - swn.S0).abs() < ATM_TOL
        atm_iv_mkt = swn.ivs_black[atm_mask][0].item() * 100
        print(f"    {key[0]:.0f}Y×{key[1]:.0f}Y:  model={atm_iv_model:.2f}%  "
              f"market={atm_iv_mkt:.2f}%  diff={atm_iv_model-atm_iv_mkt:+.2f}bp")

    print("\n" + "=" * 60)
    print("Gradient flow: Layers 0→6 (approximate scheme)")
    print("=" * 60)

    params.zero_grad()
    result = compute_total_loss(
        params, mkt,
        N_paths=500, M=10,
        use_exact=False, variance_curve_mode="simplified",
        seed=123, swaption_keys=list(mkt.swaptions.keys())[:3],
    )
    result["loss"].backward()

    print(f"Loss = {result['loss'].item():.6f}")
    for name, param in params.named_parameters():
        if param.grad is not None:
            gn = param.grad.norm().item()
            print(f"  {name:20s}: |grad| = {gn:.6e}  {'✓' if gn > 0 else '✗'}")
        else:
            print(f"  {name:20s}: grad = None  ✗")

    print("\n" + "=" * 60)
    print("Gradient flow: Layers 0→6 (BLP hybrid scheme, κ=2)")
    print("=" * 60)

    params.zero_grad()
    result_hybrid = compute_total_loss(
        params, mkt,
        N_paths=500, M=10,
        scheme="hybrid", hybrid_kappa=2,
        variance_curve_mode="simplified",
        seed=123, swaption_keys=list(mkt.swaptions.keys())[:3],
    )
    result_hybrid["loss"].backward()

    print(f"Loss = {result_hybrid['loss'].item():.6f}")
    for name, param in params.named_parameters():
        if param.grad is not None:
            gn = param.grad.norm().item()
            print(f"  {name:20s}: |grad| = {gn:.6e}  {'✓' if gn > 0 else '✗'}")
        else:
            print(f"  {name:20s}: grad = None  ✗")

    print("\n" + "=" * 60)
    print("Layer 7: Mini calibration (5 steps, for testing)")
    print("=" * 60)

    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    test_keys = [k for k in mkt.swaptions.keys() if k[1] == 1][:3]
    if not test_keys:
        test_keys = list(mkt.swaptions.keys())[:3]
    print(f"Calibrating on: {test_keys}")

    cal_result = calibrate(
        params, mkt,
        n_iterations=5, lr=1e-2,
        N_paths=500, M=10,
        use_exact=False, variance_curve_mode="simplified",
        use_crn=True, crn_seed=42,
        log_every=1,
        swaption_keys=test_keys,
    )

    print("\n" + "=" * 60)
    print("Post-calibration diagnostics (MC)")
    print("=" * 60)

    print_calibration_report(
        params, mkt,
        variance_curve_mode="simplified",
        N_paths=5000, M=50,
        swaption_keys=test_keys,
    )

    if test_keys:
        swn = mkt.swaptions[test_keys[0]]
        print_smile_comparison(
            params, swn, mkt,
            variance_curve_mode="simplified",
            N_paths=5000, M=50,
        )

    print("\nDone.")
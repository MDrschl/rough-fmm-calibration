"""
main.py
==============================
Automatic Monte Carlo Calibration for the Mapped Rough SABR Forward Market Model.

Layers 0-1: Parameter space, constraints, and market data loading.

Copyright: Maximilian Droschl, 2026, Master Thesis
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.special import hyp2f1 as scipy_hyp2f1


# =============================================================================
# Utility: deterministic swaption key hashing (for CRN reproducibility)
# =============================================================================

def _stable_key_hash(key: tuple) -> int:
    """
    Deterministic hash for swaption keys (expiry, tenor).

    Python's built-in hash() on tuples is randomised per process since 3.3
    (unless PYTHONHASHSEED=0), breaking CRN reproducibility across runs.
    This function produces a stable integer from (expiry, tenor) pairs.
    """
    return int(key[0] * 1000) * 100 + int(key[1])


# Tolerance for identifying ATM strikes: |K - S₀| < ATM_TOL.
# Forward swap rates are ~0.03-0.05 in decimal, so 1e-6 is sub-basis-point
# and more robust to rounding than the previous 1e-10.
ATM_TOL = 1e-6


# =============================================================================
# Black and Bachelier pricing utilities
# =============================================================================

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


def bachelier_to_black_iv(S0, K, T, sigma_n, annuity, is_call=True):
    """
    Convert Bachelier (normal) IV to Black (lognormal) IV via root finding.
    sigma_n: normal vol in decimal (NOT basis points).
    Returns: Black IV in decimal.
    """
    if K <= 0 or S0 <= 0 or T <= 0:
        return np.nan

    target_price = bachelier_price_np(S0, K, T, sigma_n, annuity, is_call)

    # Quick check: intrinsic value
    intrinsic = annuity * max(0.0, (S0 - K) if is_call else (K - S0))
    if target_price <= intrinsic + 1e-16:
        return np.nan

    def objective(sigma_black):
        return black_price_np(S0, K, T, sigma_black, annuity, is_call) - target_price

    # Initial bracket: use approximation sigma_B ≈ sigma_N / S0 as starting point
    sigma_approx = sigma_n / S0
    try:
        result = brentq(objective, 1e-6, 10.0, xtol=1e-10, maxiter=200)
        return result
    except ValueError:
        return np.nan


# =============================================================================
# Layer 1: Market data container
# =============================================================================

@dataclass
class SwaptionData:
    """Market data for a single swaption (expiry, tenor) pair."""
    I: int                      # Start index in annual grid
    J: int                      # End index in annual grid
    expiry_years: float
    tenor_years: float
    S0: torch.Tensor            # Forward swap rate (scalar)
    A0: torch.Tensor            # Forward annuity (scalar)
    Pi: torch.Tensor            # Frozen weights Pi^0_j, shape (J-I,)
    pi: torch.Tensor            # Normalized weights pi_j, shape (J-I,)
    strikes: torch.Tensor       # Strike grid, shape (n_strikes,)
    ivs_black: torch.Tensor     # Black (lognormal) IVs, shape (n_strikes,)
    ivs_normal: torch.Tensor    # Bachelier (normal) IVs in decimal, shape (n_strikes,)
    is_call: torch.Tensor       # OTM flag, shape (n_strikes,)
    target_prices: torch.Tensor # Black model prices, shape (n_strikes,)
    vegas: torch.Tensor         # Black vegas for weighting, shape (n_strikes,)
    n_strikes: int


@dataclass
class MarketData:
    """
    Complete market data for calibration.

    All quantities are torch tensors on the specified device.
    Swaptions are organized for efficient batch computation:
    swaptions sharing the same (I, J) pair use the same simulated paths.
    """
    # Curve data
    P: torch.Tensor                           # Discount factors P(T_j), shape (N+1,)
    R: torch.Tensor                           # Forward term rates R_j, shape (N+1,)
    N: int                                    # Number of forward rates
    theta: float                              # Year fraction (=1 for annual)

    # Individual swaption data
    swaptions: dict                           # key: (expiry, tenor) -> SwaptionData

    # Pre-grouped by (I, J) for batch simulation
    groups: dict = field(default_factory=dict) # key: (I, J) -> list of SwaptionData

    # Calibration subset keys
    subset_keys: dict = field(default_factory=dict)

    device: str = "cpu"


def load_market_data(
    pkl_path: str,
    subset: str = "joint_all_smiles",
    date: str = "2024-12-09",
    device: str = "cpu",
) -> 'MarketData':
    """
    Load preprocessed market data and convert to torch tensors.

    The pickle file contains data for multiple dates, each with its own
    SOFR curve and swaption IVs.  All IVs are stored as Bachelier (normal)
    vol in decimal; the Black (lognormal) IV conversion is done during
    preprocessing and stored alongside.

    Args:
        pkl_path: Path to the multi-date pickle from preprocess_usd_swaptions.py
        subset:   Which calibration subset to load. One of:
                  'stage1_1y_tenor', 'stage1_alpha_expiries',
                  'stage2_multi_tenor', 'joint_all_smiles', 'joint_all_atm'
        date:     Date string, e.g. '2024-12-09'. Must be present in the pickle.
        device:   torch device ('cpu' or 'cuda')

    Returns:
        MarketData object with all tensors on the specified device.
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    # --- Multi-date or legacy format detection ---
    if "dates" in raw and date in raw:
        # New multi-date format
        date_data = raw[date]
        N = date_data["T_N"]
        theta = date_data["theta"]
        P_np = date_data["discount_factors"][:N + 1]
        R_np = date_data["forward_term_rates"][:N + 1]
        swaptions_raw = date_data["swaptions"]
        subset_keys_all = date_data["calibration_subsets"]
    elif "metadata" in raw and "swaptions" in raw:
        # Legacy single-date format (backward compatibility)
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

    # Select swaption subset
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

        # --- IV handling ---
        # New format: both ivs_normal and ivs_black are pre-computed
        if "ivs_black" in swn_raw and "ivs_normal" in swn_raw:
            ivs_black_np = swn_raw["ivs_black"]
            ivs_normal_np = swn_raw["ivs_normal"]
        elif "ivs_normal" in swn_raw:
            # Only normal: convert to Black
            ivs_normal_np = swn_raw["ivs_normal"]
            ivs_black_np = np.array([
                bachelier_to_black_iv(S0, K, T_exp, sig_n, A0, bool(ic))
                for K, sig_n, ic in zip(strikes_np, ivs_normal_np, is_call_np)
            ])
        else:
            # Legacy: ivs field with mixed convention
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

            # Approximate normal vol for legacy data
            ivs_normal_np = np.array([
                sig_b * S0 for sig_b in ivs_black_np
            ])

        # Compute Black prices and vegas
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

    # Group swaptions by (I, J) for batch simulation
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


# =============================================================================
# Layer 0: Parameter module with smooth constraints
# =============================================================================

class MappedRoughSABRParams(nn.Module):
    """
    Layer 0: Learnable parameters for the Mapped Rough SABR FMM.

    Stores unconstrained parameters and maps them to the constrained space
    via smooth bijections, ensuring stable gradient flow.

    Following Adachi et al. Section 6.2, we parametrize the FULL (N+1)×(N+1)
    correlation matrix Σ of (W⁰, W¹, ..., Wᴺ) via a single Rapisarda angular
    decomposition (Rapisarda, Brigo & Mercurio 2007).

    Layout:  row/col 0 = W⁰ (vol driver),  rows/cols 1..N = W¹..Wᴺ (rates).

    This jointly constrains spot-vol correlations ρ_{0,j} and forward-rate
    correlations ρ_{ij}, guaranteeing that the full system is a valid
    correlation matrix (symmetric PSD, unit diagonal) for any angle values.

    The first-column angles ω_{i,0} are mapped to (π/2, π) so that
    ρ_{i,0} = cos(ω_{i,0}) ∈ (-1, 0), enforcing Adachi's condition (18).

    Constrained parameter space:
        H       ∈ (0, 0.5)     Hurst exponent
        eta     ∈ (0, ∞)       Vol-of-vol (η in the rBergomi convention)
        alpha_j ∈ (0, ∞)       Variance level per forward rate, j=1,...,N
        Σ       (N+1)×(N+1)    Unified correlation matrix via Rapisarda
          └─ ρ_{0,j} ∈ (-1, 0)    spot-vol correlations  [from Σ[0, 1:]]
          └─ ρ_{ij}   PSD          forward-rate correlations [from Σ[1:, 1:]]

    Unconstrained → constrained mappings:
        H     = 0.5 · sigmoid(H̃)
        eta   = softplus(η̃)
        alpha = softplus(α̃)
        ω_{i,0} = π/2 + (π/2)·sigmoid(ω̃_{i,0})   ∈ (π/2, π)   [→ ρ₀<0]
        ω_{i,j} = π · sigmoid(ω̃_{i,j})             ∈ (0, π)     [j ≥ 1]
        Σ = B Bᵀ  where  B = lower_tri(ω)
    """

    def __init__(self, N: int, device: str = "cpu"):
        """
        Args:
            N: Number of forward rates (= T_N, e.g. 11).
        """
        super().__init__()
        self.N = N
        self._device = device

        # --- Unconstrained parameters ---

        # Hurst exponent: H ∈ (0, 0.5)
        # Initialize near H = 0.2 (Adachi's optimal)
        # sigmoid⁻¹(0.2/0.5) = sigmoid⁻¹(0.4) = log(0.4/0.6) ≈ -0.405
        self.H_tilde = nn.Parameter(
            torch.tensor(-0.405, dtype=torch.float64, device=device)
        )

        # Vol-of-vol: eta ∈ (0, ∞) — matches the rBergomi convention η
        # Initialize near eta = 2.3 (notebook default)
        # softplus⁻¹(2.3) = log(exp(2.3)-1) ≈ 2.194
        self.eta_tilde = nn.Parameter(
            torch.tensor(2.194, dtype=torch.float64, device=device)
        )

        # Variance levels: alpha_j ∈ (0, ∞), j = 1,...,N
        # Initialize near alpha = 0.3 (typical vol level for rates)
        # softplus⁻¹(0.3) = log(exp(0.3)-1) ≈ -0.853
        self.alpha_tilde = nn.Parameter(
            torch.full((N,), -0.853, dtype=torch.float64, device=device)
        )

        # Unified Rapisarda angles: (N+1)×(N+1) unconstrained
        # Row 0 has no free angles (b_{0,0} = 1 trivially).
        # Rows 1..N: omega_{i,0} controls ρ_{i,0} (spot-vol),
        #            omega_{i,j≥1} controls inter-rate correlations.
        #
        # Initialization for column 0 (spot-vol):
        #   Want ρ_{i,0} = cos(ω_{i,0}) ≈ -0.5
        #   → ω_{i,0} = 2π/3
        #   → ω_{i,0} = π/2 + (π/2)·sigmoid(x) = 2π/3
        #   → sigmoid(x) = 1/3  → x = log(1/2) ≈ -0.693
        #
        # Initialization for columns ≥ 1 (inter-rate):
        #   sigmoid(0) = 0.5 → ω = π/2 → cos = 0, sin = 1
        #   This concentrates each row on the diagonal,
        #   giving near-identity inter-rate correlation.
        init_omega = torch.zeros(N + 1, N + 1, dtype=torch.float64, device=device)
        init_omega[1:, 0] = np.log(0.5)  # → sigmoid ≈ 1/3 → ρ₀ ≈ -0.5
        # columns ≥ 1 stay at 0 (→ ω = π/2)
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
        """
        Full (N+1)×(N+1) correlation matrix Σ for (W⁰, W¹, ..., Wᴺ).

        Constructed via the Rapisarda angular parametrization
        (Rapisarda, Brigo & Mercurio 2007), as used in Adachi et al. §6.2.

        The lower-triangular factor B has entries:
            b_{0,0} = 1
            b_{i,j} = cos(ω_{i,j}) · ∏_{k=0}^{j-1} sin(ω_{i,k})   for 0 ≤ j < i
            b_{i,i} = ∏_{k=0}^{i-1} sin(ω_{i,k})

        Then Σ = B Bᵀ is guaranteed PSD with unit diagonal.

        First-column angles ω_{i,0} ∈ (π/2, π) ensure ρ_{i,0} ∈ (-1, 0)
        per Adachi's condition (18): ρ_{0,j} ≤ 0.

        Returns: shape (N+1, N+1) correlation matrix.
        """
        Np1 = self.N + 1

        # Map unconstrained → angles
        omega = torch.zeros(Np1, Np1, dtype=torch.float64, device=self._device)

        # Column 0 (spot-vol): ω_{i,0} ∈ (π/2, π) for i ≥ 1
        # → cos(ω_{i,0}) ∈ (-1, 0)  [condition (18)]
        omega[1:, 0] = (
            torch.pi / 2 + (torch.pi / 2) * torch.sigmoid(self.omega_tilde[1:, 0])
        )

        # Columns ≥ 1 (inter-rate): ω_{i,j} ∈ (0, π)
        omega[1:, 1:] = torch.pi * torch.sigmoid(self.omega_tilde[1:, 1:])

        # Build lower-triangular B
        B = torch.zeros(Np1, Np1, dtype=torch.float64, device=self._device)

        for i in range(Np1):
            sin_prod = torch.ones(1, dtype=torch.float64, device=self._device)
            for l in range(i):
                B[i, l] = torch.cos(omega[i, l]) * sin_prod
                sin_prod = sin_prod * torch.sin(omega[i, l])
            B[i, i] = sin_prod

        return B @ B.T

    def get_rho0(self) -> torch.Tensor:
        """
        Spot-vol correlations ρ_{0,j} for j = 1,...,N.

        Extracted from the first row/column of the full Σ matrix.
        Guaranteed ∈ (-1, 0) by the angular constraint on ω_{j,0}.

        Returns: shape (N,)
        """
        Sigma = self.get_full_correlation_matrix()
        return Sigma[0, 1:]

    def get_correlation_matrix(self) -> torch.Tensor:
        """
        Forward-rate correlation matrix ρ_{ij} for i,j = 1,...,N.

        Extracted as the N×N lower-right subblock of the full Σ.
        Guaranteed PSD as a principal submatrix of a PSD matrix.

        Returns: shape (N, N)
        """
        Sigma = self.get_full_correlation_matrix()
        return Sigma[1:, 1:]

    def forward(self) -> dict:
        """
        Compute all constrained parameters from unconstrained ones.

        Builds the full (N+1)×(N+1) Rapisarda matrix once and extracts
        both ρ₀ and ρ from it, ensuring joint consistency.

        Returns dict with:
            H:         scalar tensor
            eta:       scalar tensor — vol-of-vol
            alpha:     shape (N,)
            rho0:      shape (N,)       — Σ[0, 1:]
            rho:       shape (N, N)     — Σ[1:, 1:]
            Sigma:     shape (N+1, N+1) — full correlation matrix
        """
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
            # Invert: H = 0.5 · sigmoid(H̃) ⇒ H̃ = log(2H / (1 - 2H))
            x = 2.0 * H_value
            self.H_tilde.fill_(np.log(x / (1.0 - x)))

    def set_eta(self, eta_value: float):
        """Set eta to a specific value."""
        with torch.no_grad():
            # Invert softplus: η = log(1+exp(x)) ⇒ x = log(exp(η)-1)
            self.eta_tilde.fill_(np.log(np.exp(eta_value) - 1.0))


# =============================================================================
# Layer 2: Forward variance curves  ξ_j(t)
# =============================================================================
#
# From Adachi et al. §6.2, the forward variance curve for rate j is:
#
#   ξ_j(t) = α_j² exp{ η²t^{2H}/4
#            − Σ_{i=j+1}^{N} [θ_i η_i(R⁰_i)/(1+θ_i R⁰_i)] α_i ρ_{0i} η√(2H)
#              × ∫₀ᵗ (t−s)^{H−1/2} γ_i(s) ds }                   (p.22)
#
# where γ_i is the piecewise linear vol loading from eq. (14):
#   γ_i(s) = 1                          if s ≤ T_{i−1}
#   γ_i(s) = (T_i − s)/(T_i − T_{i−1}) if s ∈ (T_{i−1}, T_i]
#   γ_i(s) = 0                          if s > T_i
#
# With η_i(r) = r  (lognormal SABR), the coefficient simplifies to:
#   θ_i R⁰_i / (1 + θ_i R⁰_i)
#
# The Volterra–gamma integral has a closed form (p.22):
#   ∫₀ᵗ (t−s)^{H−1/2} γ_i(s) ds
#     = t^{H+1/2}/(H+1/2)
#       − [(t−T_{i−1})₊^{H+3/2} − (t−T_i)₊^{H+3/2}]
#         / [(H+1/2)(H+3/2)(T_i − T_{i−1})]
#
# Simplified first-step version (§6.1, p.21):
#   Assuming E*[√V_t] = √V_0, we set
#     v(t) = v(0) exp(η² t^{2H} / 4)
#   which coincides with log-normal SABR when H = 0.5.
#
# Vol-of-vol convention:
#   η is the rBergomi vol-of-vol parameter as in Bayer, Friz & Gatheral.
#   V_t = v(0)·ξ(t)·exp(η√(2H) W̃ᴴ_t − η²/2 · t^{2H})
#   Some references use κ = η√(2H); we follow the notebook and use η directly.

def volterra_gamma_integral(
    t: torch.Tensor,
    H: torch.Tensor,
    T_i_minus_1: float,
    T_i: float,
) -> torch.Tensor:
    """
    Closed-form ∫₀ᵗ (t−s)^{H−1/2} γ_i(s) ds.

    Args:
        t:            scalar or (M,) time points
        H:            scalar Hurst parameter
        T_i_minus_1:  float T_{i−1}
        T_i:          float T_i

    Returns:
        Same shape as t — integral values
    """
    Hp12 = H + 0.5
    Hp32 = H + 1.5
    delta_T = T_i - T_i_minus_1  # = θ_i for annual grid

    # First term: always present
    result = t ** Hp12 / Hp12

    # Second term: correction from [T_{i-1}, T_i] region
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
    """
    Layer 2 (full): Compute ξ_j(t) on a time grid using Adachi eq. (p.22).

    ξ_j(t) = α_j² exp{ η²t^{2H}/4 − Σ_{i=j+1}^{N} c_i · ∫₀ᵗ ... ds }

    where c_i = [θ_i R⁰_i/(1+θ_i R⁰_i)] α_i ρ_{0i} η√(2H).

    Args:
        t_grid: shape (M,) — time points
        j:      int — 1-indexed forward rate index
        H:      scalar Hurst parameter
        eta:    scalar vol-of-vol (rBergomi convention)
        alpha:  shape (N,) — 0-indexed vol levels  (alpha[i-1] = α_i)
        rho0:   shape (N,) — 0-indexed spot-vol     (rho0[i-1] = ρ_{0,i})
        R:      shape (N+1,) — forward rates R[i] = R_i  (1-indexed in storage)
        theta:  float year fraction (= 1 for annual)
        N:      int number of forward rates
        precomputed_integrals: optional dict mapping i → integral values on t_grid.
            If provided, avoids redundant Volterra-gamma integral evaluations
            when multiple rates share the same integrals.

    Returns:
        shape (M,) — ξ_j(t) evaluated on the grid
    """
    alpha_j = alpha[j - 1]  # 0-indexed
    two_H = 2.0 * H
    sqrt_2H = torch.sqrt(two_H)

    # Growth term: η² t^{2H} / 4
    exponent = eta ** 2 * t_grid ** two_H / 4.0

    # Drift correction: sum over i = j+1, ..., N
    for i in range(j + 1, N + 1):
        # Coefficient:  θ_i R⁰_i / (1 + θ_i R⁰_i) · α_i · ρ_{0,i} · η√(2H)
        R_i = R[i]                    # R_i  (1-indexed in R tensor)
        c_i = (theta * R_i / (1.0 + theta * R_i)
               * alpha[i - 1]        # α_i  (0-indexed)
               * rho0[i - 1]         # ρ_{0,i}  (0-indexed)
               * eta * sqrt_2H)

        # Use precomputed integral if available, else compute fresh
        if precomputed_integrals is not None and i in precomputed_integrals:
            integral = precomputed_integrals[i]
        else:
            T_i_minus_1 = float(i - 1) * theta
            T_i = float(i) * theta
            integral = volterra_gamma_integral(t_grid, H, T_i_minus_1, T_i)

        exponent = exponent - c_i * integral

    return alpha_j ** 2 * torch.exp(exponent)


# =============================================================================
# Layer 3: FMM aggregation  →  effective parameters  v(t), ρ_eff
# =============================================================================
#
# The mapped model S* (eq. 23) is a rough Bergomi model with:
#
#   v(t) = Σ_{i,j=I+1}^{J} ρ_{ij} π_i π_j √(ξ_i(t) ξ_j(t))     (eq. 22)
#
#   ρ_eff = (1/√v(0)) Σ_{j=I+1}^{J} ρ_{0j} π_j √ξ_j(0)          (eq. 24)
#
# where π_j = Π⁰_j η_j(R⁰_j) / S₀ are the normalized frozen weights.
# With η_j(r) = r (lognormal mapping): π_j = Π⁰_j R⁰_j / S₀.
#
# Three modes for the forward variance curve:
#   "full"       — full ξ_j(t) from p.22, then v(t) via eq. (22)
#   "simplified" — v(t) = v(0) exp(κ² t^{2H} / (8H))  (§6.1, p.21)
#   "constant"   — v(t) = v(0)  (flat, for testing)

def compute_effective_params(
    alpha: torch.Tensor,
    rho0: torch.Tensor,
    rho_matrix: torch.Tensor,
    swn: "SwaptionData",
) -> dict:
    """
    Layer 3 (core): Compute v(0) and ρ_eff from frozen weights.

    Args:
        alpha:      shape (N,) — volatility levels
        rho0:       shape (N,) — spot-vol correlations
        rho_matrix: shape (N, N) — forward rate correlation matrix
        swn:        SwaptionData with frozen weights pi (shape J-I)

    Returns:
        dict with:
            v0:       scalar — v(0), aggregated forward variance at t=0
            rho_eff:  scalar — effective correlation ρ in (-1, 1)
    """
    I, J = swn.I, swn.J

    # Normalized weights π_j, shape (J-I,)
    pi = swn.pi  # precomputed: Π⁰_j R⁰_j / S₀

    # --- v(0) via eq. (22) at t=0 ---
    # ξ_j(0) = α_j² (all correction terms vanish at t=0)
    # v(0) = Σ ρ_{ij} π_i π_j α_i α_j = wᵀ ρ_sub w
    alpha_sub = alpha[I:J]  # α_{I+1},...,α_J (0-indexed)
    w = pi * alpha_sub      # shape (J-I,)

    rho_sub = rho_matrix[I:J, I:J]  # (J-I, J-I) submatrix
    v0 = w @ rho_sub @ w            # scalar

    # --- ρ_eff via eq. (24) ---
    # ρ_eff = (1/√v(0)) Σ_j ρ_{0j} w_j
    rho0_sub = rho0[I:J]
    rho_eff = (rho0_sub * w).sum() / torch.sqrt(v0 + 1e-30)

    # Clamp for numerical safety (should rarely activate with
    # unified Rapisarda, since the full system is jointly PSD).
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
    """
    Layer 3 (variance curve): Compute v(t)/v(0) on the simulation grid.

    Three modes:
        "full"       — compute ξ_j(t) for each tenor j, then v(t) via eq. (22),
                        then normalize by v(0).
        "simplified" — v(t)/v(0) = exp(η² t^{2H} / 4)  (first-step, §6.1)
        "constant"   — v(t)/v(0) = 1  (flat baseline)

    Args:
        time_grid:  shape (M,) — simulation time points
        swn:        SwaptionData
        mkt:        MarketData (for R, theta, N)
        H, eta, alpha, rho0, rho_matrix: model parameters
        v0:         scalar — v(0) from compute_effective_params
        mode:       "full", "simplified", or "constant"

    Returns:
        v_curve: shape (M,) — v(t_i)/v(0) ratio for the simulation
    """
    M = time_grid.shape[0]

    if mode == "constant":
        return torch.ones(M, dtype=torch.float64, device=time_grid.device)

    if mode == "simplified":
        # v(t) = v(0) exp(η² t^{2H} / 4)   →   ratio = exp(...)
        two_H = 2.0 * H
        return torch.exp(eta ** 2 * time_grid ** two_H / 4.0)

    if mode == "full":
        I, J = swn.I, swn.J
        pi = swn.pi  # shape (J-I,)
        n_rates = J - I

        # Precompute Volterra-gamma integrals for all rate indices.
        # These are shared across different j values in compute_xi_full,
        # avoiding redundant recomputation.
        precomputed_integrals = {}
        for i in range(1, mkt.N + 1):
            T_i_minus_1 = float(i - 1) * mkt.theta
            T_i = float(i) * mkt.theta
            precomputed_integrals[i] = volterra_gamma_integral(
                time_grid, H, T_i_minus_1, T_i,
            )

        # Compute ξ_j(t) for j = I+1, ..., J on the time grid
        # xi_all: shape (n_rates, M)
        xi_all = torch.stack([
            compute_xi_full(
                time_grid, j=I + 1 + k, H=H, eta=eta,
                alpha=alpha, rho0=rho0, R=mkt.R, theta=mkt.theta, N=mkt.N,
                precomputed_integrals=precomputed_integrals,
            )
            for k in range(n_rates)
        ], dim=0)  # (n_rates, M)

        # v(t_m) = Σ_{k,l} ρ_{kl} π_k π_l √(ξ_k(t_m) ξ_l(t_m))  via eq. (22)
        # = Σ_{k,l} ρ_{kl} π_k π_l √ξ_k √ξ_l
        # = (π ⊙ √ξ)ᵀ ρ (π ⊙ √ξ)
        rho_sub = rho_matrix[I:J, I:J]  # (n_rates, n_rates)
        sqrt_xi = torch.sqrt(xi_all)     # (n_rates, M)

        # w_m[k] = π_k · √ξ_k(t_m)  →  shape (n_rates, M)
        w_t = pi.unsqueeze(1) * sqrt_xi   # (n_rates, M)

        # v(t_m) = w_t[:,m]ᵀ ρ w_t[:,m]  for each m
        # = diag(w_tᵀ ρ w_t)
        v_t = (w_t * (rho_sub @ w_t)).sum(dim=0)  # (M,)

        # Ratio v(t)/v(0)
        return v_t / (v0 + 1e-30)

    raise ValueError(f"Unknown variance curve mode: {mode}")


# =============================================================================
# Layer 4: Rough Bergomi simulation for the mapped swap rate S*
# =============================================================================
#
# The mapped model (eq. 23):
#     dS*/S* = √V_t dW*,     S*_0 = S₀
#     V_t = v(t) exp(η√(2H) W̃ᴴ_t − η²/2 · t^{2H})
#     ⟨W^{0*}, W*⟩_t = ρ_eff t
#
# Three simulation schemes:
#  (a) Exact: Cholesky of 2M×2M covariance matrix of (W̃^H, W).
#      Accurate; but H not differentiable (Cholesky outside graph).
#  (b) Approximate: naive Riemann-sum kernel discretization.
#      H differentiable; but poor near the singularity for rough H.
#  (c) Hybrid (BLP 2017): near field (last κ cells) via exact Wiener
#      integrals of the power kernel + far field via optimal Riemann sum.
#      H differentiable AND accurate — best of both worlds.
#      Uses only a tiny (κ+1)×(κ+1) Cholesky (κ=2 recommended).

def covariance_matrix_rBergomi(H: float, M: int, T: float) -> np.ndarray:
    """
    Compute the 2M×2M covariance matrix for exact rough Bergomi simulation.

    Joint distribution of (W̃^H_{t_1}, W_{t_1}, ..., W̃^H_{t_M}, W_{t_M})
    on uniform grid t_i = i·h with h = T/M.

    Interleaved layout: indices [2i] = W̃^H_{t_{i+1}}, [2i+1] = W_{t_{i+1}}.

    Uses the formulas from the AMCC reference notebook (Bayer et al.)
    with scipy's hyp2f1 for the fBM autocovariance.

    Args:
        H: Hurst parameter (float, not tensor)
        M: number of time steps
        T: maturity

    Returns:
        np.ndarray of shape (2M, 2M)
    """
    h = T / M
    mat = np.zeros((2 * M, 2 * M))
    gamma_aux = 0.5 - H
    frac_aux = 1.0 / (1.0 - gamma_aux)

    for i in range(M):
        for j in range(M):
            mm = min(i, j) + 1     # min index + 1
            mm_x = max(i, j) + 1   # max index + 1

            # Cov[W̃^H_{t_{i+1}}, W_{t_{j+1}}]
            mat[2*i, 2*j+1] = (
                ((i+1)*h)**(H+0.5) - ((i+1)*h - mm*h)**(H+0.5)
            ) / (H + 0.5)

            # Cov[W̃^H_{t_{i+1}}, W̃^H_{t_{j+1}}] via hypergeometric
            mat[2*i, 2*j] = (
                ((mm*h)**(2*H))
                * float(scipy_hyp2f1(1.0, gamma_aux, 2 - gamma_aux, mm / mm_x))
                * frac_aux
                * (mm / mm_x)**gamma_aux
            )

            # Cov[W_{t_{i+1}}, W_{t_{j+1}}] = min(t_{i+1}, t_{j+1})
            mat[2*i+1, 2*j+1] = mm * h

            # Cov[W_{t_{i+1}}, W̃^H_{t_{j+1}}]
            mat[2*i+1, 2*j] = (
                ((j+1)*h)**(H+0.5) - ((j+1)*h - mm*h)**(H+0.5)
            ) / (H + 0.5)

    return mat


def build_cholesky(H: float, M: int, T: float, device: str = "cpu") -> torch.Tensor:
    """
    Build and Cholesky-decompose the fBM-BM covariance matrix.

    Args:
        H: Hurst parameter
        M: number of time steps
        T: maturity (expiry)
        device: torch device

    Returns:
        Lower-triangular Cholesky factor, shape (2M, 2M), float64
    """
    cov = covariance_matrix_rBergomi(H, M, T)
    L = np.linalg.cholesky(cov)
    return torch.tensor(L, dtype=torch.float64, device=device)


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
    """
    Layer 4 (exact scheme): Simulate S*_T via Cholesky-based exact fBM sampling.

    This mirrors the reference notebook's `simulated_paths_rough_bergomi_exact`.
    H is NOT differentiable in this scheme (the Cholesky is precomputed).

    The dynamics:
        log(S*_{t_{i+1}}) = log(S*_{t_i}) - ½ V_{t_i} h + √V_{t_i} (ρ ΔW⁰_i + √(1-ρ²) ΔW⊥_i)
        V_{t_i} = v(t_i) exp(η√(2H) W̃^H_{t_i} - η²/2 · t_i^{2H})

    Args:
        S0:         scalar — initial forward swap rate
        v0:         scalar — v(0) = wᵀρw (aggregated forward variance)
        eta:        scalar — vol-of-vol (rBergomi convention)
        H:          scalar — Hurst parameter (used only for the correction term)
        rho_eff:    scalar — effective spot-vol correlation
        T:          float — swaption expiry
        M:          int — number of time steps
        N_paths:    int — number of Monte Carlo paths
        cholesky_L: shape (2M, 2M) — precomputed Cholesky factor
        v_curve:    Optional shape (M,) — v(t_i)/v(0) forward variance curve ratio.
                    If None, assumed constant (v_curve = 1).
        antithetic: bool — if True, use antithetic variates for variance reduction.

    Returns:
        S_T: shape (N_paths,) — terminal swap rate values
    """
    h = T / M
    device = S0.device
    sqrt_rho = torch.sqrt(1.0 - rho_eff**2)

    # Draw standard normals (with antithetic variates if requested)
    if antithetic:
        N_half = N_paths // 2
        Z_corr_half = torch.randn(2 * M, N_half, dtype=torch.float64, device=device)
        Z_indep_half = torch.randn(N_half, M, dtype=torch.float64, device=device) * np.sqrt(h)
        Z_corr = torch.cat([Z_corr_half, -Z_corr_half], dim=1)
        Z_indep = torch.cat([Z_indep_half, -Z_indep_half], dim=0)
    else:
        Z_corr = torch.randn(2 * M, N_paths, dtype=torch.float64, device=device)
        Z_indep = torch.randn(N_paths, M, dtype=torch.float64, device=device) * np.sqrt(h)

    # Correlate: prod_mat = L @ Z_corr → shape (2M, N_paths), transpose to (N_paths, 2M)
    prod_mat = (cholesky_L @ Z_corr).T  # shape (N_paths, 2M)

    # Initialize
    log_S = torch.log(S0) * torch.ones(N_paths, dtype=torch.float64, device=device)

    # Correction term: -η²/2 · t^{2H}
    # Matches notebook: -self.eta**2 / 2 * ((i+1)*h)**(2*H)
    two_H = 2.0 * H
    sqrt_2H = torch.sqrt(two_H)
    eta_sq = eta ** 2

    # Initialize variance at LEFT endpoint (t = 0): V(0) = v(0)
    # This matches the notebook: vol_states = torch.ones(...) * self.v0
    V_current = v0 * torch.ones(N_paths, dtype=torch.float64, device=device)

    prev_W = torch.zeros(N_paths, dtype=torch.float64, device=device)

    for i in range(M):
        t_next = (i + 1) * h

        # W̃^H_{t_{i+1}} from the Cholesky product (even indices)
        fBm_next = prod_mat[:, 2 * i]

        # W⁰_{t_{i+1}} from the Cholesky product (odd indices)
        W0_next = prod_mat[:, 2 * i + 1]

        # ΔW⁰ over [t_i, t_{i+1}]
        dW0_i = W0_next - prev_W
        prev_W = W0_next

        # --- Step 1: Price update using LEFT-endpoint variance V(t_i) ---
        # (Euler-Maruyama: evaluate drift and diffusion at the left endpoint)
        brownian_incr = rho_eff * dW0_i + sqrt_rho * Z_indep[:, i]
        log_S = log_S - 0.5 * V_current * h + torch.sqrt(V_current) * brownian_incr

        # --- Step 2: Update variance to RIGHT endpoint V(t_{i+1}) for next step ---
        # V = v0 · ξ(t) · exp(η√(2H) · W̃ᴴ_t − η²/2 · t^{2H})
        correction = eta_sq / 2.0 * t_next ** two_H
        V_next = v0 * torch.exp(eta * sqrt_2H * fBm_next - correction)

        # Forward variance curve adjustment
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
    """
    Layer 4 (approximate scheme): Simulate S*_T via kernel discretization.

    H IS differentiable in this scheme — the kernel (t-s)^{H-1/2}
    is evaluated inside the computational graph.

    This mirrors the reference notebook's `simulated_paths_rough_bergomi`.

    Args:
        (same as simulate_exact, without cholesky_L)
        antithetic: bool — if True, use antithetic variates for variance reduction.

    Returns:
        S_T: shape (N_paths,) — terminal swap rate values
    """
    h = T / M
    device = S0.device
    sqrt_h = np.sqrt(h)
    sqrt_rho = torch.sqrt(1.0 - rho_eff**2)

    # Draw BM increments (with antithetic variates if requested)
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

    # Initialize variance at LEFT endpoint (t = 0): V(0) = v(0)
    # This matches the notebook: vol_states = torch.ones(...) * self.v0
    V_current = v0 * torch.ones(N_paths, dtype=torch.float64, device=device)

    for i in range(M):
        t_next = (i + 1) * h

        # --- Step 1: Price update using LEFT-endpoint variance V(t_i) ---
        brownian_incr = rho_eff * dW0[:, i] + sqrt_rho * dW_perp[:, i]
        log_S = log_S - 0.5 * V_current * h + torch.sqrt(V_current) * brownian_incr

        # --- Step 2: Update variance to RIGHT endpoint V(t_{i+1}) for next step ---
        # Kernel discretization of fBm at t_{i+1}:
        # W̃^H_{t_{i+1}} ≈ Σ_{k=0}^{i} ((i+1-k)h)^{H-1/2} ΔW⁰_k
        # Following the notebook: vec_tmp = h*(i+1 - arange(0, i+1))
        lags = h * (i + 1 - torch.arange(0, i + 1, dtype=torch.float64, device=device))
        kernels = lags ** (H - 0.5)  # shape (i+1,)
        fBm_next = dW0[:, 0:i+1] @ kernels  # shape (N_paths,)

        # V = v0 · ξ(t) · exp(η√(2H) · W̃ᴴ_t − η²/2 · t^{2H})
        correction = eta_sq / 2.0 * t_next ** two_H
        V_next = v0 * torch.exp(eta * sqrt_2H * fBm_next - correction)

        if v_curve is not None:
            V_next = V_next * v_curve[i]

        V_next = torch.clamp(V_next, min=0.0)
        V_current = V_next

    return torch.exp(log_S)


# ---------------------------------------------------------------------------
# BLP Hybrid scheme (Bennedsen, Lunde & Pakkanen, Finance & Stochastics 2017)
# ---------------------------------------------------------------------------
#
# Splits the Volterra integral for the fBm into a NEAR field (last κ cells,
# handled via exact Wiener integrals of the power kernel) and a FAR field
# (remaining cells, Riemann-sum with optimal evaluation points b*_k).
#
# The near-field covariance matrix Σ is only (κ+1)×(κ+1)  (κ = 2 or 3),
# so torch.linalg.cholesky keeps H in the computational graph.
# The far-field weights (b*_k)^α · h^α are also computed in the graph.
# ⇒ H is fully differentiable — no two-stage compromise needed.
#
# Reference: BLP (2017) eq. (3.7) & §3.1 for the TBSS (truncated BSS)
#            specialised to the power kernel g(x) = x^α, α = H − ½.
# ---------------------------------------------------------------------------


def _gauss_legendre_nodes_weights(n_quad: int = 64):
    """
    Return Gauss-Legendre nodes and weights on [0, 1] (numpy, float64).
    Cached at module level after first call.
    """
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_quad)
    # transform from [-1, 1] to [0, 1]
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    return nodes, weights


# Module-level cache (computed once)
_GL64_NODES, _GL64_WEIGHTS = _gauss_legendre_nodes_weights(64)


def build_hybrid_covariance(
    alpha: torch.Tensor,
    h: float,
    kappa: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build the (κ+1)×(κ+1) covariance matrix Σ for the BLP hybrid scheme.

    The random vector at each time step is:
        W⃗ᵢ = (ΔWᵢ,  W^near_{i,1},  …,  W^near_{i,κ})

    where ΔWᵢ is the BM increment and W^near_{i,j} is the Wiener integral
        W^near_{i,j} = ∫_{ih}^{(i+1)h} ((i+j)h − s)^α  dW(s)

    Entries (Itô isometry, substituting u = s − ih):

        Σ[0,0] = h

        Σ[0,j] = h^{α+1}/(α+1) · (j^{α+1} − (j−1)^{α+1})     j=1…κ

        Σ[j,k] = h^{2α+1} · ∫₀¹ (j−u)^α (k−u)^α du           j,k=1…κ

    The last integral is computed via 64-point Gauss-Legendre quadrature
    in torch, keeping α (and hence H) in the graph.

    Args:
        alpha:  scalar tensor — roughness index α = H − ½  (in graph)
        h:      float — time step size T/M
        kappa:  int — number of near-field cells (typically 2 or 3)
        device: torch device

    Returns:
        Σ: tensor of shape (kappa+1, kappa+1), dtype float64
    """
    dim = kappa + 1
    Sigma = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    # Quadrature nodes/weights on [0,1]
    u = torch.tensor(_GL64_NODES, dtype=torch.float64, device=device)  # (64,)
    w = torch.tensor(_GL64_WEIGHTS, dtype=torch.float64, device=device)

    alpha1 = alpha + 1.0      # α + 1
    two_alpha1 = 2.0 * alpha + 1.0  # 2α + 1
    h_t = torch.tensor(h, dtype=torch.float64, device=device)

    # --- Σ[0,0] = h ---
    Sigma[0, 0] = h_t

    for j in range(1, dim):
        jf = float(j)

        # --- Σ[0,j] = h^{α+1}/(α+1) · (j^{α+1} − (j−1)^{α+1}) ---
        j_t = torch.tensor(jf, dtype=torch.float64, device=device)
        jm1_t = torch.tensor(jf - 1.0, dtype=torch.float64, device=device)

        cov_0j = (h_t ** alpha1 / alpha1) * (j_t ** alpha1 - jm1_t ** alpha1)
        Sigma[0, j] = cov_0j
        Sigma[j, 0] = cov_0j

        for k in range(j, dim):
            kf = float(k)
            # --- Σ[j,k] = h^{2α+1} · ∫₀¹ (j−u)^α (k−u)^α du ---
            # Integrand at quadrature nodes:
            k_t = torch.tensor(kf, dtype=torch.float64, device=device)
            integrand = (j_t - u) ** alpha * (k_t - u) ** alpha  # (64,)
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
    """
    Compute optimal far-field kernel weights for the BLP hybrid scheme.

    For k = κ+1, …, M the optimal evaluation point (Proposition 2.8) is:

        b*_k = ((k^{α+1} − (k−1)^{α+1}) / (α+1))^{1/α}

    and the far-field weight is:

        w_k = (b*_k)^α · h^α  =  ((k^{α+1} − (k−1)^{α+1}) / (α+1)) · h^α

    Note: (b*_k)^α simplifies without taking the 1/α root,
    avoiding numerical issues when α is near zero.

    Args:
        alpha:  scalar tensor — α = H − ½  (kept in graph)
        h:      float — step size
        kappa:  int — number of near-field cells
        M:      int — total number of time steps

    Returns:
        weights: tensor of shape (M,) where
                 weights[k] = far-field kernel weight for lag k+1
                 (indices 0…κ−1 are zero; κ…M−1 are the active weights)
    """
    h_t = torch.tensor(h, dtype=torch.float64, device=device)
    alpha1 = alpha + 1.0
    weights = torch.zeros(M, dtype=torch.float64, device=device)

    k_vals = torch.arange(kappa + 1, M + 1, dtype=torch.float64, device=device)
    km1_vals = k_vals - 1.0

    # (b*_k)^α = (k^{α+1} − (k−1)^{α+1}) / (α+1)
    bstar_alpha = (k_vals ** alpha1 - km1_vals ** alpha1) / alpha1

    # w_k = (b*_k)^α · h^α
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
    """
    Layer 4 (BLP hybrid scheme): Simulate S*_T with H fully differentiable.

    Combines exact Wiener integrals for the last κ cells (near field)
    with optimally weighted Riemann sums for earlier cells (far field).

    The near-field covariance is a tiny (κ+1)×(κ+1) matrix whose Cholesky
    decomposition is differentiable via torch.linalg.cholesky.  The far-field
    kernel weights are also computed in the graph.
    ⇒ Gradients flow through H, η, ρ simultaneously.

    Implements BLP (2017) eq. (3.7) specialised to the power kernel
    g(x) = x^α with α = H − ½ and L_g ≡ 1.

    Args:
        S0:         scalar — initial forward swap rate
        v0:         scalar — v(0) = wᵀΡw
        eta:        scalar — vol-of-vol (rBergomi convention)
        H:          scalar — Hurst parameter (IN the graph)
        rho_eff:    scalar — effective spot-vol correlation
        T:          float  — swaption expiry
        M:          int    — number of time steps
        N_paths:    int    — number of MC paths
        v_curve:    Optional shape (M,) — v(tᵢ)/v(0) forward variance curve
        kappa:      int    — number of near-field cells (default 2, paper
                             shows κ≥1 already matches exact scheme for α<0)
        antithetic: bool   — if True, use antithetic variates for variance
                             reduction (Gonon & Stockinger §4.1). Generates
                             N_paths//2 base draws and their negations,
                             yielding N_paths total paths.

    Returns:
        S_T: shape (N_paths,) — terminal swap rate values
    """
    h = T / M
    device = S0.device
    alpha = H - 0.5          # roughness index, in graph
    two_H = 2.0 * H
    sqrt_2H = torch.sqrt(two_H)
    eta_sq = eta ** 2
    sqrt_rho = torch.sqrt(1.0 - rho_eff ** 2)

    # ---- 1.  Near-field covariance and Cholesky (differentiable in H) ----
    Sigma = build_hybrid_covariance(alpha, h, kappa, device=device)
    # Adaptive jitter for numerical safety — for very rough H (α near -0.5)
    # the near-field covariance can be ill-conditioned.
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

    # ---- 2.  Far-field optimal kernel weights (differentiable in H) ------
    far_weights = compute_bstar_weights(alpha, h, kappa, M, device=device)  # (M,)

    # ---- 3.  Generate i.i.d. (κ+1)-dimensional Gaussian vectors ----------
    #   With antithetic variates: generate half, then negate.
    if antithetic:
        N_half = N_paths // 2
        Z_half = torch.randn(N_half, M, kappa + 1, dtype=torch.float64, device=device)
        Z = torch.cat([Z_half, -Z_half], dim=0)  # (N_paths, M, kappa+1)
        dW_perp_half = torch.randn(N_half, M, dtype=torch.float64, device=device) * np.sqrt(h)
        dW_perp = torch.cat([dW_perp_half, -dW_perp_half], dim=0)
    else:
        Z = torch.randn(N_paths, M, kappa + 1, dtype=torch.float64, device=device)
        dW_perp = torch.randn(N_paths, M, dtype=torch.float64, device=device) * np.sqrt(h)

    # Correlate: W_vec = Z @ L_near^T  →  (N_paths, M, kappa+1)
    W_vec = Z @ L_near.T

    dW0 = W_vec[:, :, 0]          # BM increments,      (N_paths, M)
    W_near = W_vec[:, :, 1:]       # near-field integrals (N_paths, M, kappa)

    # ---- 5.  Time-stepping -----------------------------------------------
    log_S = torch.log(S0) * torch.ones(N_paths, dtype=torch.float64, device=device)
    V_current = v0 * torch.ones(N_paths, dtype=torch.float64, device=device)

    for i in range(M):
        t_next = (i + 1) * h

        # --- Step 1: Price update using LEFT-endpoint variance V(tᵢ) ---
        brownian_incr = rho_eff * dW0[:, i] + sqrt_rho * dW_perp[:, i]
        log_S = log_S - 0.5 * V_current * h + torch.sqrt(V_current) * brownian_incr

        # --- Step 2: fBm at t_{i+1} via hybrid decomposition ---
        # Index in the paper: we want W̃^H at time (i+1)h.
        #
        # Near field:  Σ_{k=1}^{min(i+1,κ)}  W^near_{(i+1)−k, k}
        #   k=1 → W_near[i,   0]   (lag-1 integral generated at step i)
        #   k=2 → W_near[i-1, 1]   (lag-2 integral generated at step i-1)
        #   ...
        #
        # Far field:   Σ_{k=κ+1}^{i+1}  far_weights[k-1] · ΔW_{(i+1)−k}
        #   (indices into dW0 run from i−κ down to 0)

        fBm_val = torch.zeros(N_paths, dtype=torch.float64, device=device)

        # -- Near field --
        n_near = min(i + 1, kappa)
        for k in range(1, n_near + 1):
            # W^near generated at timestep (i+1-k), for lag index k
            fBm_val = fBm_val + W_near[:, i + 1 - k, k - 1]

        # -- Far field --
        n_far = i + 1 - kappa  # number of far-field terms
        if n_far > 0:
            # Far lags: k = κ+1 … i+1
            # ΔW indices: (i+1) − k  for k = κ+1…i+1  →  i−κ … 0
            # So we need dW0[:, 0 : i+1−κ] with weights far_weights[κ : i+1]
            # Kernel: far_weights[κ], far_weights[κ+1], …, far_weights[i]
            # dW0:    dW0[:, i−κ],   dW0[:, i−κ−1], …, dW0[:, 0]
            #
            # → inner product: dW0[:, 0:n_far].flip(1)  ⊙  far_weights[κ:κ+n_far]
            # Equivalently: dW0[:, 0:n_far] @ far_weights[κ:κ+n_far].flip(0)

            fw = far_weights[kappa: kappa + n_far]   # (n_far,)
            fBm_val = fBm_val + dW0[:, 0:n_far] @ fw.flip(0)

        # --- Step 3: Variance update to V(t_{i+1}) ---
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
    """
    End-to-end simulation: Layers 0-4 combined.

    Given model parameters and swaption data, compute terminal swap rate
    samples S*_T for pricing.

    Args:
        params:              MappedRoughSABRParams module
        swn:                 SwaptionData for one swaption
        mkt:                 MarketData (needed for R, theta, N in full mode)
        N_paths:             number of MC paths
        M:                   number of time steps
        use_exact:           True = exact Cholesky scheme, False = approximate
                             (ignored if `scheme` is provided)
        variance_curve_mode: "full" (eq. 22 with ξ_j from p.22),
                             "simplified" (v(t)=v(0)exp(η²t^{2H}/4), §6.1),
                             "constant" (v(t)=v(0), flat baseline)
        cholesky_cache:      dict mapping cache_key -> Cholesky factor
        scheme:              "exact", "approx", or "hybrid"
                             (overrides use_exact if provided)
        hybrid_kappa:        number of near-field cells for hybrid scheme
                             (κ=2 is recommended; BLP show κ≥1 already
                             matches exact for α<0, i.e. H<0.5)
        antithetic:          if True, use antithetic variates for variance
                             reduction (Gonon & Stockinger §4.1).

    Returns:
        S_T: shape (N_paths,) — terminal swap rate samples
    """
    # Backward-compatible scheme resolution
    if scheme is None:
        scheme = "exact" if use_exact else "approx"

    p = params()
    T = swn.expiry_years

    # Layer 3: aggregate to effective parameters
    eff = compute_effective_params(p["alpha"], p["rho0"], p["rho"], swn)

    # Layer 2-3: compute variance curve v(t)/v(0)
    h = T / M
    time_grid = torch.arange(1, M + 1, dtype=torch.float64,
                             device=params._device) * h  # (M,)
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
    else:  # "approx"
        S_T = simulate_approx(
            S0=swn.S0, v0=eff["v0"], eta=p["eta"],
            H=p["H"], rho_eff=eff["rho_eff"],
            T=T, M=M, N_paths=N_paths,
            v_curve=v_curve, antithetic=antithetic,
        )

    return S_T


# =============================================================================
# Layer 5: Swaption payoff and price computation
# =============================================================================

def compute_swaption_prices(
    S_T: torch.Tensor,
    swn: "SwaptionData",
) -> torch.Tensor:
    """
    Layer 5: Compute Monte Carlo swaption prices for all strikes.

    Under the frozen-annuity swap measure approximation:
        Price = A₀ × E[(S*_T - K)⁺]  (payer / call on swap rate)
        Price = A₀ × E[(K - S*_T)⁺]  (receiver / put on swap rate)

    Using OTM convention (put for K < S₀, call for K ≥ S₀).

    Args:
        S_T:  shape (N_paths,) — terminal swap rate samples
        swn:  SwaptionData with strikes, is_call, A0

    Returns:
        prices: shape (n_strikes,) — Monte Carlo option prices
    """
    # S_T: (N_paths,) → (N_paths, 1) for broadcasting
    S_T_col = S_T.unsqueeze(1)  # (N_paths, 1)
    K = swn.strikes.unsqueeze(0)  # (1, n_strikes)

    # Payoff: (S_T - K)⁺ or (K - S_T)⁺ depending on is_call
    diff = S_T_col - K  # (N_paths, n_strikes)

    # Apply call/put flag: is_call = True → (S-K)⁺, False → (K-S)⁺
    sign = torch.where(swn.is_call.unsqueeze(0), torch.ones_like(diff), -torch.ones_like(diff))
    payoff = torch.clamp(sign * diff, min=0.0)  # (N_paths, n_strikes)

    # MC average, discounted by annuity
    mc_price = swn.A0 * payoff.mean(dim=0)  # (n_strikes,)

    return mc_price


# =============================================================================
# =============================================================================
# ATM variance relationship for α matching
# =============================================================================
#
# From §6.2 of Adachi et al.: "We determine α_i such that the ATM swaption
# price computed under the mapped rough Bergomi forward swap model matches
# the market IV."
#
# At ATM (k=0), the Black implied vol is:
#   σ_ATM(T) ≈ √v̄(T)
#
# where v̄(T) = ∫₀¹ v(Ts) ds is the time-averaged forward variance and
# v(0) = wᵀ ρ w with w_j = π_j α_j.
#
# This is NOT an asymptotic expansion — it is simply the definition of
# time-averaged variance.  The actual smile (OTM IVs) is always computed
# via Monte Carlo simulation.
#
# NOTE: An earlier version of this code included a first-order short-time
# asymptotic skew formula σ(k,t) = √v̄(t)·(1 + ψ·k·t^{H-1/2}) from the
# Fukasawa-Gatheral line of work.  This was removed because:
#   (a) Adachi et al. do not use it in their calibration experiment,
#   (b) it is inaccurate at longer maturities and deeper OTM strikes,
#   (c) all smile computation should use MC, consistent with the paper.

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
    """
    Compute the time-averaged forward variance v̄(T) = ∫₀¹ v(Ts) ds.

    Args:
        T:     expiry (years)
        v0:    v(0) — aggregated forward variance at t=0
        H:     Hurst parameter
        eta:   vol-of-vol (rBergomi convention)
        mode:  "simplified" or "full"
        swn, mkt, alpha, rho0, rho_matrix: needed for "full" mode
        n_quad: quadrature points

    Returns:
        scalar tensor — v̄(T)
    """
    # Gauss-Legendre quadrature on [0, 1]
    # v̄(T) = ∫₀¹ v(Ts) ds
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)
    # Map from [-1, 1] to [0, 1]
    s_np = 0.5 * (nodes_np + 1.0)
    w_np = 0.5 * weights_np

    device = v0.device if hasattr(v0, 'device') else 'cpu'
    s = torch.tensor(s_np, dtype=torch.float64, device=device)
    w = torch.tensor(w_np, dtype=torch.float64, device=device)
    t_points = T * s  # (n_quad,)

    if mode == "simplified":
        # v(t) = v(0) · exp(η² t^{2H} / 4)
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


# =============================================================================
# α_i ATM matching (root-finding)
# =============================================================================
#
# From §6.2: "We determine α_i such that the ATM swaption price computed
# under the mapped rough Bergomi forward swap model matches the market IV."
#
# Using the rough SABR formula at ATM (k=0):
#   σ_ATM = √v̄(T)
# and v̄(T) is a known function of α (via v(0) and the variance curve).
#
# For the simplified case:
#   v̄(T) = v(0) · G(H, η, T)   where G = ∫₀¹ exp(η²T^{2H}s^{2H}/4) ds
#   v(0) = wᵀ ρ w    with  w_j = π_j · α_j
#
# For a single-rate swaption (1Y tenor, J = I+1):
#   v(0) = (π_j α_j)²,   v̄ = (π_j α_j)² · G
#   σ_ATM = π_j α_j √G   →   α_j = σ_ATM / (π_j √G)
#
# For multi-rate swaptions, α affects v(0) quadratically; Brent solves it.

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
    N_paths: int = 10000,
    M: int = 50,
    seed: int = 42,
) -> torch.Tensor:
    """
    Find α_j for rate j such that model ATM IV = market ATM IV.

    For 1Y-tenor swaptions (J = I+1), this is a single rate, and we can
    solve analytically via the ATM variance relationship. For multi-rate
    swaptions, we use Brent root-finding.

    Args:
        swn:          SwaptionData
        mkt:          MarketData
        H, eta, rho0, rho_matrix: current model parameters
        alpha_other:  shape (N,) — current α values (used for other rates)
        variance_curve_mode: "simplified" or "full"
        method:       "formula" (ATM variance: σ_ATM = √v̄) or "mc" (Monte Carlo)
        N_paths, M:   MC settings (if method="mc")
        seed:         RNG seed (if method="mc")

    Returns:
        scalar tensor — optimal α_j value
    """
    I, J = swn.I, swn.J
    n_rates = J - I

    # Find ATM IV from market
    atm_mask = (swn.strikes - swn.S0).abs() < ATM_TOL
    if atm_mask.sum() == 0:
        # Fallback: closest to ATM
        atm_idx = (swn.strikes - swn.S0).abs().argmin()
        atm_mask = torch.zeros(swn.n_strikes, dtype=torch.bool)
        atm_mask[atm_idx] = True
    sigma_atm_mkt = swn.ivs_black[atm_mask][0]

    pi = swn.pi

    if method == "formula":
        # --- ATM variance approach ---

        if n_rates == 1:
            # Single rate: analytic solution
            # v(0) = (π_j α_j)², v̄ = v(0) · G → σ_ATM = π_j α_j √G
            # → α_j = σ_ATM / (π_j √G)
            j_idx = I  # 0-indexed in alpha
            pi_j = pi[0]

            # Compute G = ∫₀¹ exp(η²T^{2H}s^{2H}/4) ds via quadrature
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
                # For full mode, G depends on alpha itself — need iterative solve
                # _match_alpha_brent returns a scaling factor; convert to alpha
                c = _match_alpha_brent(
                    swn, mkt, H, eta, rho0, rho_matrix,
                    alpha_other, sigma_atm_mkt, variance_curve_mode,
                    method="formula",
                )
                return torch.clamp(alpha_other[j_idx] * c, min=1e-6)

            alpha_j = sigma_atm_mkt / (pi_j * torch.sqrt(G) + 1e-30)
            return torch.clamp(alpha_j, min=1e-6)

        else:
            # Multi-rate: Brent root-finding → returns a scaling factor
            c = _match_alpha_brent(
                swn, mkt, H, eta, rho0, rho_matrix,
                alpha_other, sigma_atm_mkt, variance_curve_mode,
                method="formula",
            )
            return c  # scaling factor, NOT an alpha value

    elif method == "mc":
        # MC-based ATM matching via Brent → returns a scaling factor
        c = _match_alpha_brent(
            swn, mkt, H, eta, rho0, rho_matrix,
            alpha_other, sigma_atm_mkt, variance_curve_mode,
            method="mc", N_paths=N_paths, M=M, seed=seed,
        )
        # For single-rate, convert scaling factor to alpha value
        if n_rates == 1:
            return torch.clamp(alpha_other[I] * c, min=1e-6)
        return c  # multi-rate: return scaling factor

    else:
        raise ValueError(f"Unknown ATM matching method: {method}")


def _match_alpha_brent(
    swn, mkt, H, eta, rho0, rho_matrix,
    alpha_other, sigma_atm_mkt, variance_curve_mode,
    method="formula", N_paths=10000, M=50, seed=42,
):
    """
    Internal: Brent root-finding for α matching.

    Scales all α_{I+1},...,α_J by a common factor c, and finds c
    such that model ATM IV = market ATM IV.
    """
    I, J = swn.I, swn.J
    T = swn.expiry_years
    pi = swn.pi

    # We scale alpha[I:J] by factor c: alpha_scaled[I:J] = c * alpha_other[I:J]
    # Start from alpha_other (current values)
    alpha_base = alpha_other[I:J].detach().clone()

    # Pre-build Cholesky for MC mode (depends only on H, T, M — not on α)
    cholesky_L = None
    if method == "mc":
        H_val = H.detach().item()
        cholesky_L = build_cholesky(H_val, M, T)

    def objective(log_c):
        c = np.exp(log_c)
        alpha_trial = alpha_other.detach().clone()
        alpha_trial[I:J] = alpha_base * c

        if method == "formula":
            # Compute v(0), v̄, then ATM IV via variance relationship
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
            # Full MC simulation using exact Cholesky scheme
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
            # Invert ATM price to IV
            atm_payoff = torch.clamp(S_T - swn.S0, min=0.0).mean()
            atm_price = (swn.A0 * atm_payoff).item()
            try:
                sigma_model = brentq(
                    lambda s: black_price_np(
                        swn.S0.item(), swn.S0.item(), T, s,
                        swn.A0.item(), True,
                    ) - atm_price,
                    1e-6, 5.0,
                )
            except ValueError:
                sigma_model = 0.3  # fallback

        return sigma_model - sigma_atm_mkt.item()

    # Brent search over log(c) in [-3, 3] → c ∈ [0.05, 20]
    try:
        log_c_opt = brentq(objective, -3.0, 3.0, xtol=1e-8, maxiter=100)
        c_opt = np.exp(log_c_opt)
    except ValueError:
        c_opt = 1.0  # no solution found, keep current

    # Return the scaling factor so the caller can apply it to the full block
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
) -> torch.Tensor:
    """
    Match α_i for all smile tenors via ATM root-finding.

    For each 1Y-tenor swaption (first-step calibration), find α_i
    such that model ATM IV = market ATM IV.

    Args:
        mkt:          MarketData
        H, eta, rho0, rho_matrix: current parameters
        alpha_init:   shape (N,) — starting α values
        smile_keys:   list of (expiry, tenor) keys to match; default = all 1Y tenors
        variance_curve_mode: "simplified" or "full"
        method:       "formula" or "mc"

    Returns:
        alpha: shape (N,) — updated α values
    """
    alpha = alpha_init.detach().clone()

    if smile_keys is None:
        # Default: 1Y-tenor smiles (first-step)
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
        )

        # Update: for 1Y-tenor, J = I+1, so only one rate per smile
        if J - I == 1:
            alpha[I] = alpha_j
        else:
            # Multi-rate: alpha_j is a scaling factor c from Brent
            # Apply it directly to preserve relative proportions
            alpha[I:J] = alpha[I:J] * alpha_j

    return alpha

# =============================================================================
# Layer 6: Loss function
# =============================================================================
#
# Adachi et al. §6.2: "the sum of squared differences between the market IVs
# in smile tenors and those generated by the mapped SMM model (23)."
#
# Two approaches:
#
# (a) Vega-weighted price loss (default — fully differentiable):
#       L = Σ_k ((Price_MC(K_k) - Price_Mkt(K_k)) / Vega(K_k))²
#     This is ≈ Σ_k (σ_model - σ_market)² by the first-order expansion
#     Price ≈ Price_mkt + Vega · (σ_model - σ_market), so
#     (Price_MC - Price_mkt)/Vega ≈ σ_model - σ_market.
#     Advantages: stays inside the PyTorch graph, smooth gradients.
#
# (b) IV-space loss (exact — requires root-finding):
#       L = Σ_k (σ_model(K_k) - σ_market(K_k))²
#     where σ_model is obtained by inverting the MC price via Brent.
#     More faithful to the paper's objective, but the inversion is
#     non-differentiable. Use for reporting; for optimization, use (a).

def black_price_torch(S0, K, T, sigma, A0, is_call):
    """
    Black (1976) price in PyTorch (differentiable).

    Args:
        S0, K, T, sigma, A0: scalar or broadcastable tensors
        is_call: bool tensor

    Returns:
        price tensor
    """
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
    """
    Invert MC prices to Black IVs via Brent root-finding (non-differentiable).

    Used for reporting and IV-space loss evaluation, NOT for backprop.

    Args:
        mc_prices: shape (n_strikes,) — MC option prices
        swn:       SwaptionData with S0, A0, strikes, is_call, expiry_years

    Returns:
        ivs: shape (n_strikes,) — Black implied vols (NaN where inversion fails)
    """
    ivs = torch.full_like(mc_prices, float('nan'))
    S0 = swn.S0.item()
    A0 = swn.A0.item()
    T = swn.expiry_years

    for i in range(swn.n_strikes):
        target = mc_prices[i].item()
        K = swn.strikes[i].item()
        ic = swn.is_call[i].item()

        # Intrinsic value check
        intrinsic = A0 * max(0.0, (S0 - K) if ic else (K - S0))
        if target <= intrinsic + 1e-16:
            continue

        def obj(sigma):
            return float(black_price_np(S0, K, T, sigma, A0, ic)) - target

        try:
            iv = brentq(obj, 1e-6, 10.0, xtol=1e-10, maxiter=200)
            ivs[i] = iv
        except (ValueError, RuntimeError):
            pass

    return ivs


def compute_loss_vegaweighted(
    mc_prices: torch.Tensor,
    swn: "SwaptionData",
) -> torch.Tensor:
    """
    Layer 6a: Vega-weighted price loss for a single swaption.

    L = Σ_k ((Price_MC(K_k) - Price_Mkt(K_k)) / Vega(K_k))²

    This approximates the IV-space loss while remaining fully differentiable.

    Args:
        mc_prices: shape (n_strikes,) — MC option prices (in the graph)
        swn:       SwaptionData with target_prices, vegas

    Returns:
        scalar loss tensor
    """
    # Price residuals normalized by vega
    residuals = (mc_prices - swn.target_prices) / (swn.vegas + 1e-30)
    return (residuals ** 2).sum()


def compute_loss_ivspace(
    mc_prices: torch.Tensor,
    swn: "SwaptionData",
) -> torch.Tensor:
    """
    Layer 6b: IV-space loss for a single swaption.

    L = Σ_k (σ_model(K_k) - σ_market(K_k))²

    Non-differentiable (uses root-finding). For monitoring only.

    Args:
        mc_prices: shape (n_strikes,) — MC option prices
        swn:       SwaptionData with ivs_black

    Returns:
        scalar loss tensor (detached from graph)
    """
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
) -> dict:
    """
    Layer 6: Compute the total calibration loss over all swaptions.

    For each swaption:
      1. Simulate S*_T (Layers 2-4)
      2. Compute MC prices (Layer 5)
      3. Compute vega-weighted loss

    Aggregates across swaptions with equal weighting.

    Args:
        params:              MappedRoughSABRParams module
        mkt:                 MarketData
        N_paths:             MC paths per swaption
        M:                   time steps
        use_exact:           Cholesky vs approximate scheme
                             (ignored if `scheme` is provided)
        variance_curve_mode: "full", "simplified", "constant"
        cholesky_cache:      reusable Cholesky factors
        seed:                if set, fix the RNG for common random numbers (CRN)
        swaption_keys:       if set, only compute loss for these (expiry, tenor) keys;
                             otherwise use all swaptions in mkt.swaptions
        scheme:              "exact", "approx", or "hybrid"
        hybrid_kappa:        near-field cells for hybrid scheme
        compute_diagnostics: if True, compute IV-space loss and model IVs via
                             Brent root-finding (expensive, for logging only).
                             Set to False during non-logging optimisation steps.

    Returns:
        dict with:
            loss:          scalar tensor — total vega-weighted loss (for backprop)
            loss_iv:       scalar — total IV-space loss (for monitoring, detached)
                           (0.0 if compute_diagnostics=False)
            per_swaption:  dict (key → {loss, loss_iv, mc_prices, model_ivs})
                           (model_ivs and loss_iv omitted if compute_diagnostics=False)
    """
    keys = swaption_keys if swaption_keys is not None else list(mkt.swaptions.keys())

    total_loss = torch.tensor(0.0, dtype=torch.float64, device=params._device)
    total_loss_iv = 0.0
    per_swaption = {}

    for key in keys:
        swn = mkt.swaptions[key]

        # Common random numbers: same seed offset per swaption for reproducibility
        if seed is not None:
            torch.manual_seed(seed + _stable_key_hash(key))

        # Layers 2-4: simulate
        S_T = simulate_swaption(
            params, swn, mkt,
            N_paths=N_paths, M=M,
            use_exact=use_exact,
            variance_curve_mode=variance_curve_mode,
            cholesky_cache=cholesky_cache,
            scheme=scheme,
            hybrid_kappa=hybrid_kappa,
        )

        # Layer 5: prices
        mc_prices = compute_swaption_prices(S_T, swn)

        # Layer 6: vega-weighted loss (always computed — needed for backprop)
        loss_vw = compute_loss_vegaweighted(mc_prices, swn)
        total_loss = total_loss + loss_vw

        record = {
            "loss": loss_vw.item(),
            "mc_prices": mc_prices.detach(),
        }

        # IV-space diagnostics (expensive Brent root-finding — skip during hot path)
        if compute_diagnostics:
            model_ivs = mc_prices_to_black_iv(mc_prices.detach(), swn)
            valid = ~torch.isnan(model_ivs)
            if valid.sum() > 0:
                diff = model_ivs[valid] - swn.ivs_black[valid]
                loss_iv_val = (diff ** 2).sum().item()
            else:
                loss_iv_val = 0.0
            total_loss_iv += loss_iv_val
            record["loss_iv"] = loss_iv_val
            record["model_ivs"] = model_ivs

        per_swaption[key] = record

    return {
        "loss": total_loss,
        "loss_iv": total_loss_iv,
        "per_swaption": per_swaption,
    }


# =============================================================================
# Layer 7: Calibration loop (Adam + common random numbers)
# =============================================================================
#
# Two-stage calibration strategy following Adachi et al. and the AMCC framework:
#
#   Stage 1 (approximate scheme):
#     - H is differentiable → optimize all parameters including H
#     - Higher MC variance, but can search over H
#     - Use "simplified" variance curve for speed
#     - Fewer paths, more iterations
#
#   Stage 2 (exact Cholesky scheme):
#     - Fix H at the value from Stage 1, rebuild Cholesky
#     - Lower MC variance for precise α, ρ₀, ρ calibration
#     - Use "full" or "simplified" variance curve
#     - More paths, fewer iterations
#
# Common random numbers (CRN): fix the seed each iteration so that
# the same noise realization is used, reducing MC gradient variance.

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
    scheme: Optional[str] = None,
    hybrid_kappa: int = 2,
    early_stop_patience: Optional[int] = None,
    early_stop_tol: float = 1e-4,
) -> dict:
    """
    Layer 7: Run the calibration loop.

    Uses Adam optimizer with configurable learning rate scheduling.
    Supports common random numbers (CRN) for variance reduction.

    Args:
        params:              MappedRoughSABRParams (modified in place)
        mkt:                 MarketData
        n_iterations:        maximum optimization steps (upper bound)
        lr:                  initial (peak) learning rate
        N_paths:             MC paths per swaption per iteration
        M:                   simulation time steps
        use_exact:           exact (Cholesky) vs approximate scheme
                             (ignored if `scheme` is provided)
        variance_curve_mode: "full", "simplified", "constant"
        use_crn:             enable common random numbers
        crn_seed:            base seed for CRN
        log_every:           print progress every N steps
        swaption_keys:       subset of swaptions to calibrate on
        scheduler_type:      "plateau" — ReduceLROnPlateau (good for Stage 1,
                                 hybrid; halves lr when loss stalls)
                             "cosine"  — CosineAnnealingLR with optional linear
                                 warmup (good for Stage 2; guarantees smooth
                                 decay from lr → min_lr over n_iterations)
        scheduler_patience:  (plateau only) reduce LR after this many steps
                             without improvement
        scheduler_factor:    (plateau only) LR reduction factor
        min_lr:              minimum learning rate (used by both schedulers)
        warmup_steps:        (cosine only) linear warmup from min_lr to lr
                             over this many steps before cosine decay begins
        scheme:              "exact", "approx", or "hybrid"
        hybrid_kappa:        near-field cells for hybrid scheme
        early_stop_patience: stop if best loss hasn't improved for this many
                             steps. None disables early stopping.
        early_stop_tol:      relative improvement threshold — a new best must
                             beat best_loss * (1 - tol) to count as improvement.

    Returns:
        dict with:
            history:   list of dicts {step, loss, loss_iv, lr, params_snapshot}
            best:      dict of best parameters (lowest loss)
    """
    # Resolve scheme for cache decision
    effective_scheme = scheme if scheme is not None else ("exact" if use_exact else "approx")

    optimizer = torch.optim.Adam(params.parameters(), lr=lr)

    # Build scheduler
    if scheduler_type == "cosine":
        # Cosine annealing: lr decays smoothly from `lr` to `min_lr`
        # over (n_iterations - warmup_steps) steps, with optional linear warmup.
        cosine_steps = max(1, n_iterations - warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=min_lr,
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

        # Only compute expensive IV diagnostics at logging steps
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
        )

        loss = result["loss"]
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=10.0)

        optimizer.step()

        # Step the scheduler (plateau needs the loss value; cosine does not)
        if scheduler_is_plateau:
            scheduler.step(loss.item())
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Track best with relative tolerance
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
        }
        history.append(record)

        if step % log_every == 0 or step == n_iterations - 1:
            with torch.no_grad():
                p = params()
                H_val = p["H"].item()
                eta_val = p["eta"].item()
                kappa_adachi = eta_val * np.sqrt(2.0 * H_val)
            n_strikes_total = sum(
                mkt.swaptions[k].n_strikes
                for k in (swaption_keys if swaption_keys else mkt.swaptions)
            )
            if result["loss_iv"] > 0:
                rmse_iv = np.sqrt(result["loss_iv"] / max(1, n_strikes_total)) * 100
                rmse_str = f"RMSE_IV={rmse_iv:.3f}%"
            else:
                rmse_str = "RMSE_IV=n/a"
            print(f"  [{step:4d}/{n_iterations}]  "
                  f"loss={loss.item():.6f}  "
                  f"{rmse_str}  "
                  f"H={H_val:.4f}  η={eta_val:.4f}  κ={kappa_adachi:.4f}  "
                  f"lr={current_lr:.2e}")

        # Early stopping check
        if early_stop_patience is not None and steps_without_improvement >= early_stop_patience:
            print(f"  Early stopping at step {step}: no improvement "
                  f"for {early_stop_patience} steps (best loss={best_loss:.6f})")
            break

        # Rebuild Cholesky cache if H changed and using exact scheme
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
    # Stage 1 settings
    stage1_iterations: int = 300,
    stage1_lr: float = 5e-3,
    stage1_N_paths: int = 5000,
    stage1_M: int = 30,
    stage1_keys: Optional[list] = None,
    # Stage 2 settings
    stage2_iterations: int = 200,
    stage2_lr: float = 1e-3,
    stage2_N_paths: int = 20000,
    stage2_M: int = 50,
    stage2_variance_mode: str = "full",
    stage2_keys: Optional[list] = None,
    stage2_scheduler: str = "cosine",
    stage2_warmup_steps: int = 20,
    # Common
    use_crn: bool = True,
    crn_seed: int = 42,
    log_every: int = 20,
    early_stop_patience: Optional[int] = None,
    early_stop_tol: float = 1e-4,
) -> dict:
    """
    Two-stage calibration following Adachi et al.

    Stage 1: Approximate scheme (H differentiable).
        - Fewer paths, more iterations, simplified variance curve
        - Optimizes ALL parameters including H
        - Use 1Y-tenor smiles as in the paper's first step
        - Uses ReduceLROnPlateau scheduler (exploration)

    Stage 2: Exact Cholesky scheme (H fixed).
        - More paths, fewer iterations, full variance curve
        - H frozen at Stage 1 value
        - Calibrate α, ρ₀, ρ for precise fit
        - Uses cosine annealing scheduler with warmup (refinement)

    Args:
        params:  MappedRoughSABRParams (modified in place)
        mkt:     MarketData
        stage1/stage2 settings: see calibrate() for details
        stage2_scheduler: "cosine" (recommended) or "plateau"
        stage2_warmup_steps: linear warmup steps for cosine scheduler
        early_stop_patience: stop each stage if no improvement for this many steps
        early_stop_tol: relative improvement threshold

    Returns:
        dict with stage1_result, stage2_result, and final params
    """
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
        scheduler_type="plateau",       # exploration: reduce lr on stall
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
    )

    # Load best Stage 1 parameters
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
        scheduler_type=stage2_scheduler,    # refinement: smooth cosine decay
        warmup_steps=stage2_warmup_steps,
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
    )

    # Load best Stage 2 parameters
    if stage2_result["best_state"] is not None:
        params.load_state_dict(stage2_result["best_state"])

    print(f"\nStage 2 complete.")
    print(params.summary())

    return {
        "stage1": stage1_result,
        "stage2": stage2_result,
        "H_calibrated": H_calibrated,
    }


# =============================================================================
# Helpers: swaption data inspection
# =============================================================================

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


# =============================================================================
# Post-calibration diagnostics
# =============================================================================

def compute_model_smile(
    params: "MappedRoughSABRParams",
    swn: "SwaptionData",
    mkt: "MarketData",
    variance_curve_mode: str = "simplified",
    N_paths: int = 50000,
    M: int = 100,
    seed: int = 42,
    cholesky_cache: Optional[dict] = None,
) -> dict:
    """
    Compute the full model smile for a single swaption via Monte Carlo.

    Args:
        params:  calibrated MappedRoughSABRParams
        swn:     SwaptionData
        mkt:     MarketData
        variance_curve_mode, N_paths, M, seed: simulation settings

    Returns:
        dict with model_ivs, model_prices, market_ivs, strikes, errors
    """
    p = params()
    eff = compute_effective_params(p["alpha"], p["rho0"], p["rho"], swn)

    torch.manual_seed(seed)
    S_T = simulate_swaption(
        params, swn, mkt,
        N_paths=N_paths, M=M,
        use_exact=True,
        variance_curve_mode=variance_curve_mode,
        cholesky_cache=cholesky_cache,
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
) -> dict:
    """
    Print a comprehensive calibration report with per-swaption RMSE.

    All smile computation is done via Monte Carlo.

    Args:
        params:  calibrated model
        mkt:     market data
        ...      simulation settings

    Returns:
        dict with per_swaption results and summary statistics
    """
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

        # ATM error
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
):
    """
    Print a detailed smile comparison for one swaption (MC-based).
    """
    res = compute_model_smile(
        params, swn, mkt,
        variance_curve_mode=variance_curve_mode,
        N_paths=N_paths, M=M, seed=seed,
        cholesky_cache=cholesky_cache,
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
    """
    Generate data for smile plots (model vs market, MC-based).

    Returns a dict of {key: {strikes, market_ivs, model_ivs, errors}}
    suitable for matplotlib plotting.
    """
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


# =============================================================================
# Main: demonstrate Layers 0-7
# =============================================================================

if __name__ == "__main__":

    # --- Layer 1: Load market data ---
    print("Loading market data...")
    mkt = load_market_data(
        "usd_swaption_data.pkl",
        subset="joint_all_smiles",
        date="2024-12-09",
        device="cpu",
    )
    print_market_summary(mkt)

    # --- Layer 0: Initialize parameters ---
    print("\n\nInitializing parameters...")
    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    p = params()

    # --- α ATM matching test ---
    print("\n" + "=" * 60)
    print("α ATM matching test (σ_ATM = √v̄(T))")
    print("=" * 60)

    alpha_before = p["alpha"].detach().clone()
    alpha_matched = match_all_alphas(
        mkt, p["H"], p["eta"], p["rho0"], p["rho"],
        alpha_init=p["alpha"],
        variance_curve_mode="simplified",
        method="formula",    # "formula" here = ATM variance relationship, not skew formula
    )

    print(f"  {'Rate':>6s}  {'α_init':>10s}  {'α_matched':>10s}  {'Ratio':>8s}")
    print(f"  {'-'*38}")
    one_y_keys = sorted([k for k in mkt.swaptions.keys() if k[1] == 1])
    for key in one_y_keys:
        swn = mkt.swaptions[key]
        j = swn.I  # 0-indexed
        print(f"  R_{j+1:>3d}  {alpha_before[j].item():10.4f}  "
              f"{alpha_matched[j].item():10.4f}  "
              f"{alpha_matched[j].item()/alpha_before[j].item():8.3f}")

    # Verify: after matching, ATM should fit (using variance relationship)
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

    # --- Gradient flow test ---
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

    # --- Hybrid scheme gradient flow test ---
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

    # --- Mini calibration demo ---
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

    # --- Post-calibration diagnostics (MC-based) ---
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
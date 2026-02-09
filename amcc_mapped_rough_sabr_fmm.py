"""
amcc_mapped_rough_sabr_fmm.py
==============================
Automatic Monte Carlo Calibration for the Mapped Rough SABR Forward Market Model.

Layers 0-1: Parameter space, constraints, and market data loading.


Reference: Droschl, 2026
           "Fast Swaption Calibration via Automatic Differentiation in a Mapped SABR FMM"
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm
from scipy.optimize import brentq


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
    convert_otm_from_bachelier: bool = True,
    device: str = "cpu",
) -> MarketData:
    """
    Load preprocessed market data and convert to torch tensors.

    Args:
        pkl_path: Path to the pickle from preprocess_usd_swaptions.py
        subset: Which calibration subset to load. One of:
                'stage1_1y_tenor', 'stage1_alpha_expiries',
                'stage2_multi_tenor', 'joint_all_smiles', 'joint_all_atm'
        convert_otm_from_bachelier: If True, treat OTM IVs as Bachelier (bps)
                                     and convert to Black IVs.
        device: torch device ('cpu' or 'cuda')

    Returns:
        MarketData object with all tensors on the specified device.
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    N = raw["metadata"]["T_N"]
    theta = raw["metadata"]["theta"]

    # Discount factors and forward rates
    P = torch.tensor(raw["discount_factors"][:N + 1], dtype=torch.float64, device=device)
    R = torch.tensor(raw["forward_term_rates"][:N + 1], dtype=torch.float64, device=device)

    # Select swaption subset
    subset_keys_all = raw["calibration_subsets"]
    active_keys = subset_keys_all.get(subset, list(raw["swaptions"].keys()))

    swaptions = {}
    for key in active_keys:
        swn_raw = raw["swaptions"][key]
        I, J = swn_raw["I"], swn_raw["J"]
        S0 = swn_raw["S0"]
        A0 = swn_raw["A0"]

        strikes_np = swn_raw["strikes"]
        ivs_raw = swn_raw["ivs"]
        is_call_np = swn_raw["is_call"]

        # ---------------------------------------------------------------
        # IV Convention handling
        # ATM point: always Black (lognormal) IV in decimal
        # OTM points: Bachelier (normal) IV in bps (raw value / 100 = decimal,
        #             but that decimal is the Bachelier vol, not Black!)
        # ---------------------------------------------------------------
        n_strikes = len(strikes_np)
        atm_mask = np.abs(strikes_np - S0) < 1e-10
        otm_mask = ~atm_mask

        if convert_otm_from_bachelier and np.any(otm_mask):
            ivs_black_np = np.copy(ivs_raw)

            for i in np.where(otm_mask)[0]:
                # Raw OTM value was divided by 100 in preprocessing,
                # giving e.g. 0.965 for 96.5 bps.
                # True Bachelier vol in decimal: raw_bps / 10000.
                # Since preprocessing did raw/100, we need another /100.
                sigma_n = ivs_raw[i] / 100.0  # bps -> decimal

                iv_black = bachelier_to_black_iv(
                    S0, strikes_np[i], swn_raw["expiry_years"],
                    sigma_n, A0, bool(is_call_np[i])
                )
                ivs_black_np[i] = iv_black if not np.isnan(iv_black) else ivs_raw[i]
        else:
            ivs_black_np = ivs_raw

        # Recompute Black prices with correct IVs
        T_exp = swn_raw["expiry_years"]
        prices_np = np.array([
            float(black_price_np(S0, K, T_exp, sig, A0, ic))
            for K, sig, ic in zip(strikes_np, ivs_black_np, is_call_np)
        ])

        # Compute vegas
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
    via smooth bijections. All bijections have well-defined, nonzero Jacobians
    everywhere, ensuring stable gradient flow.

    Constrained parameter space:
        H       ∈ (0, 0.5)     Hurst exponent
        kappa   ∈ (0, ∞)       Vol-of-vol kernel scaling
        alpha_j ∈ (0, ∞)       Variance level per forward rate, j=1,...,N
        rho0_j  ∈ (-1, 0)      Spot-vol correlation per forward rate
        rho_ij  via Rapisarda   Full correlation matrix among forward rates

    Unconstrained → constrained mappings:
        H     = 0.5 * sigmoid(H_tilde)
        kappa = softplus(kappa_tilde)
        alpha = softplus(alpha_tilde)
        rho0  = -sigmoid(rho0_tilde)              [enforces condition (18)]
        omega = pi * sigmoid(omega_tilde)          [Rapisarda angles]
        rho   = B @ B^T  where  B = lower_tri(omega)
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
        # sigmoid^{-1}(0.2/0.5) = sigmoid^{-1}(0.4) = log(0.4/0.6) ≈ -0.405
        self.H_tilde = nn.Parameter(
            torch.tensor(-0.405, dtype=torch.float64, device=device)
        )

        # Vol-of-vol: kappa ∈ (0, ∞)
        # Initialize near kappa = 1.0
        # softplus^{-1}(1.0) = log(exp(1)-1) ≈ 0.541
        self.kappa_tilde = nn.Parameter(
            torch.tensor(0.541, dtype=torch.float64, device=device)
        )

        # Variance levels: alpha_j ∈ (0, ∞), j = 1,...,N
        # Initialize near alpha = 0.3 (typical vol level for rates)
        # softplus^{-1}(0.3) = log(exp(0.3)-1) ≈ -0.853
        self.alpha_tilde = nn.Parameter(
            torch.full((N,), -0.853, dtype=torch.float64, device=device)
        )

        # Spot-vol correlations: rho0_j ∈ (-1, 0), j = 1,...,N
        # Initialize near rho0 = -0.5
        # -sigmoid(x) = -0.5 => sigmoid(x) = 0.5 => x = 0
        self.rho0_tilde = nn.Parameter(
            torch.zeros(N, dtype=torch.float64, device=device)
        )

        # Rapisarda correlation angles: omega_{i,l} for i=1,...,N, l=0,...,N-1
        # Lower triangular structure: only l < i+1 are used
        # Initialize near identity correlation (omega small)
        # To get rho_ij ≈ delta_ij, set omega_{i,0} ≈ 0 for all i
        # and omega_{i,l} ≈ pi/2 for l >= 1 (so sin(omega) ≈ 1, cos(omega) ≈ 0)
        # This gives b_{i,0} = cos(omega_{i,0}) ≈ 1, b_{i,l} ≈ 0 for l >= 1
        # Actually for moderate correlation, start with uniform angles
        # sigmoid^{-1}(0.5) = 0 → omega = pi/2
        self.omega_tilde = nn.Parameter(
            torch.zeros(N, N, dtype=torch.float64, device=device)
        )

    def get_H(self) -> torch.Tensor:
        """Hurst exponent H ∈ (0, 0.5)."""
        return 0.5 * torch.sigmoid(self.H_tilde)

    def get_kappa(self) -> torch.Tensor:
        """Vol-of-vol kernel scaling kappa ∈ (0, ∞)."""
        return nn.functional.softplus(self.kappa_tilde)

    def get_alpha(self) -> torch.Tensor:
        """Variance levels alpha_j ∈ (0, ∞), shape (N,)."""
        return nn.functional.softplus(self.alpha_tilde)

    def get_rho0(self) -> torch.Tensor:
        """Spot-vol correlations rho0_j ∈ (-1, 0), shape (N,)."""
        return -torch.sigmoid(self.rho0_tilde)

    def get_correlation_matrix(self) -> torch.Tensor:
        """
        Correlation matrix [rho_{ij}] for i,j = 1,...,N via Rapisarda (2007).

        Constructs a lower-triangular matrix B from angular parameters omega,
        then computes rho = B @ B^T. This guarantees a valid (symmetric,
        positive semi-definite, unit diagonal) correlation matrix for any
        values of omega.

        The parametrization from Rapisarda, Mercurio & Legnani (2007):
            b_{i,0} = cos(omega_{i,0})
            b_{i,l} = cos(omega_{i,l}) * prod_{k=0}^{l-1} sin(omega_{i,k})
            for l = 1, ..., i-1
            b_{i,i} = prod_{k=0}^{i-1} sin(omega_{i,k})

        Returns: shape (N, N) correlation matrix.
        """
        N = self.N
        omega = torch.pi * torch.sigmoid(self.omega_tilde)  # (N, N), all in (0, pi)

        B = torch.zeros(N, N, dtype=torch.float64, device=self._device)

        for i in range(N):
            # Accumulate product of sines
            sin_prod = torch.ones(1, dtype=torch.float64, device=self._device)
            for l in range(i):
                B[i, l] = torch.cos(omega[i, l]) * sin_prod
                sin_prod = sin_prod * torch.sin(omega[i, l])
            # Last entry: pure product of sines
            B[i, i] = sin_prod

        # Correlation matrix
        rho = B @ B.T
        return rho

    def forward(self) -> dict:
        """
        Compute all constrained parameters from unconstrained ones.

        Returns dict with:
            H:      scalar tensor
            kappa:  scalar tensor
            alpha:  shape (N,)
            rho0:   shape (N,)
            rho:    shape (N, N) correlation matrix
        """
        return {
            "H": self.get_H(),
            "kappa": self.get_kappa(),
            "alpha": self.get_alpha(),
            "rho0": self.get_rho0(),
            "rho": self.get_correlation_matrix(),
        }

    def summary(self) -> str:
        """Print current parameter values."""
        with torch.no_grad():
            p = self.forward()
            lines = [
                "=== Mapped Rough SABR FMM Parameters ===",
                f"H     = {p['H'].item():.4f}",
                f"kappa = {p['kappa'].item():.4f}",
                f"alpha = {p['alpha'].numpy()}",
                f"rho0  = {p['rho0'].numpy()}",
                f"rho   = (N×N matrix, diag check: {torch.diag(p['rho']).numpy()})",
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
            # Invert: H = 0.5 * sigmoid(H_tilde) => sigmoid(H_tilde) = 2H
            # => H_tilde = log(2H / (1 - 2H))
            x = 2.0 * H_value
            self.H_tilde.fill_(np.log(x / (1.0 - x)))

    def set_kappa(self, kappa_value: float):
        """Set kappa to a specific value."""
        with torch.no_grad():
            # Invert softplus: kappa = log(1+exp(x)) => x = log(exp(kappa)-1)
            self.kappa_tilde.fill_(np.log(np.exp(kappa_value) - 1.0))


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
        atm_mask = (swn.strikes - swn.S0).abs() < 1e-10
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
# Main: demonstrate Layer 0 + Layer 1
# =============================================================================

if __name__ == "__main__":

    # --- Layer 1: Load market data ---
    print("Loading market data...")
    mkt = load_market_data(
        "usd_swaption_data.pkl",
        subset="joint_all_smiles",
        convert_otm_from_bachelier=True,
        device="cpu",
    )
    print_market_summary(mkt)

    # Show a few smiles
    for key in [(1.0, 1), (5.0, 5), (10.0, 1)]:
        if key in mkt.swaptions:
            print_smile(mkt, key)

    # --- Layer 0: Initialize parameters ---
    print("\n\nInitializing parameters...")
    params = MappedRoughSABRParams(N=mkt.N, device=mkt.device)
    print(params.summary())

    # Verify constraints
    print("\n--- Constraint verification ---")
    p = params()
    print(f"H in (0, 0.5): {p['H'].item():.4f}")
    print(f"kappa > 0: {p['kappa'].item():.4f}")
    print(f"alpha > 0: {(p['alpha'] > 0).all().item()}")
    print(f"rho0 in (-1, 0): min={p['rho0'].min().item():.4f}, max={p['rho0'].max().item():.4f}")
    print(f"rho diagonal = 1: {torch.allclose(torch.diag(p['rho']), torch.ones(mkt.N, dtype=torch.float64))}")
    print(f"rho symmetric: {torch.allclose(p['rho'], p['rho'].T)}")
    # Check positive semi-definite
    eigvals = torch.linalg.eigvalsh(p['rho'])
    print(f"rho PSD (min eigenvalue): {eigvals.min().item():.6f}")

    # Verify gradients flow
    print("\n--- Gradient flow verification ---")
    loss = p["H"]**2 + p["kappa"]**2 + p["alpha"].sum() + p["rho0"].sum()
    loss.backward()
    for name, param in params.named_parameters():
        grad_ok = param.grad is not None and param.grad.abs().sum().item() > 0
        print(f"  {name:20s}: grad={'OK' if grad_ok else 'MISSING'}")

    # Test set/fix H
    print("\n--- H manipulation ---")
    params.set_H(0.2)
    print(f"After set_H(0.2): H = {params.get_H().item():.4f}")
    params.fix_H()
    print(f"H frozen: requires_grad = {params.H_tilde.requires_grad}")
    params.unfix_H()
    print(f"H unfrozen: requires_grad = {params.H_tilde.requires_grad}")

    # Show (I,J) groups for batch simulation planning
    print("\n--- Simulation groups ---")
    print("Swaptions sharing (I,J) can reuse the same MC paths:")
    for ij, group in sorted(mkt.groups.items()):
        keys = [(s.expiry_years, s.tenor_years) for s in group]
        print(f"  (I={ij[0]}, J={ij[1]}): {len(group)} swaptions — {keys}")

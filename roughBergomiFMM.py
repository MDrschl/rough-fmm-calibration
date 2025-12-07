# roughBergomiFMM.py

import numpy as np
import mpmath as mp


class HybridBSS:
    """
    Hybrid scheme (Bennedsen–Lunde–Pakkanen, 2017) for a Brownian
    semistationary process X_t with kernel

        g(x) = x^a * L(x),

    observed on the grid t_i = i * T/n, i = 1,...,n.

    In our rough Bergomi application we use
        a = H - 1/2 in (-1/2, 0)
        L(x) = kappa   (constant),
    so that g(x) = kappa * x^{H-1/2} = ζ(x).
    """

    def __init__(self, a, Lfct, n, T, gamma=0.5):
        """
        Parameters
        ----------
        a : float
            Roughness parameter in (-0.5, 0.5).
        Lfct : callable
            Slowly varying function L(x); must accept numpy arrays.
        n : int
            Number of observation points.
        T : float
            Terminal time.
        gamma : float
            Burn-in exponent for the hybrid scheme (typically 0.5).
        """
        self.a = float(a)
        self.Lfct = Lfct
        self.n = int(n)
        self.T = float(T)
        self.gamma = float(gamma)

        dt = T / n
        self.dt = dt

        # Burn-in length N = floor(n^(1+gamma))
        N = int(np.floor(n ** (1.0 + gamma)))
        if N < 3:
            raise ValueError("n^(1+gamma) must be at least 3 for the hybrid scheme (N >= 3).")
        self.N = N

        # Precompute kernel g on {dt, 2dt, ..., N dt}
        k = np.arange(1, N + 1, dtype=float)  # 1..N
        a = self.a

        b = ((k**(a + 1.0) - (k - 1.0)**(a + 1.0)) / (a + 1.0))**(1.0 / a) - k
        t1 = k * dt
        t2 = t1 + b * dt
        self.t2 = t2

        g = (t2 ** a) * self.Lfct(t2)
        self.g = g

        # Kernel with first 3 terms treated by hybrid part
        g_ext = np.concatenate([np.zeros(3), g[3:]])
        self.g_ext = g_ext

        # L(t2) at first three points (scalars)
        self.L_t2_0 = float(np.asarray(self.Lfct(t2[0])).item())
        self.L_t2_1 = float(np.asarray(self.Lfct(t2[1])).item())
        self.L_t2_2 = float(np.asarray(self.Lfct(t2[2])).item())

        # 4x4 covariance matrix used in the hybrid scheme
        covM = np.zeros((4, 4), dtype=float)
        dtPowA1 = dt ** (a + 1.0)
        dtPow2a1 = dt ** (2.0 * a + 1.0)

        # Diagonal
        covM[0, 0] = dt
        covM[1, 1] = dtPow2a1 / (2.0 * a + 1.0)
        covM[2, 2] = dtPow2a1 * (2.0**(2.0 * a + 1.0) - 1.0) / (2.0 * a + 1.0)
        covM[3, 3] = dtPow2a1 * (3.0**(2.0 * a + 1.0) - 2.0**(2.0 * a + 1.0)) / (2.0 * a + 1.0)

        # Row 1
        covM[0, 1] = dtPowA1 / (a + 1.0)
        covM[0, 2] = dtPowA1 * (2.0**(a + 1.0) - 1.0) / (a + 1.0)
        covM[0, 3] = dtPowA1 * (3.0**(a + 1.0) - 2.0**(a + 1.0)) / (a + 1.0)
        covM[1, 0] = covM[0, 1]
        covM[2, 0] = covM[0, 2]
        covM[3, 0] = covM[0, 3]

        # Hypergeometric constants 2F1(1, 2(a+1); a+2; z)
        hyp1 = float(mp.hyp2f1(1.0, 2.0 * (a + 1.0), a + 2.0, -1.0))
        hyp2 = float(mp.hyp2f1(1.0, 2.0 * (a + 1.0), a + 2.0, -0.5))
        hyp3 = float(mp.hyp2f1(1.0, 2.0 * (a + 1.0), a + 2.0, -2.0))

        # Row 2
        covM[2, 1] = dtPow2a1 * (2.0**(a + 1.0) * hyp1) / (a + 1.0)
        covM[1, 2] = covM[2, 1]
        covM[3, 1] = dtPow2a1 * (3.0**(a + 1.0) * hyp2) / ((3.0 - 1.0) * (a + 1.0))
        covM[1, 3] = covM[3, 1]

        # Row 3
        covM[3, 2] = dtPow2a1 * (
            ((3.0 * 2.0)**(a + 1.0) * hyp3 - 2.0**(a + 1.0) * hyp1)
            / ((3.0 - 2.0) * (a + 1.0))
        )
        covM[2, 3] = covM[3, 2]

        self.covM = covM
        # Cholesky factor (lower triangular)
        self.chol = np.linalg.cholesky(covM)

    def sample(self, n_paths, rng=None):
        """
        Simulate n_paths paths of X_t and the driving Brownian increments.

        Returns
        -------
        X   : (n_paths, n)  values of X at times dt,2dt,...,T
        dW0 : (n_paths, n)  Brownian increments over (0,dt],...,(T-dt,T]
              (this is W^{0*} in the rough Bergomi model).
        """
        if rng is None:
            rng = np.random.default_rng()

        n = self.n
        N = self.N

        Z = rng.normal(size=(n_paths, 4, n + N))
        # We[p] = chol @ Z[p]
        We = np.einsum("ij,pjn->pin", self.chol, Z)  # (n_paths,4,n+N)
        dW1_all = We[:, 0, :]  # Brownian increments including burn-in

        X = np.empty((n_paths, n), dtype=float)
        dW0 = np.empty((n_paths, n), dtype=float)

        for p in range(n_paths):
            dW1 = dW1_all[p]  # length n+N
            conv_full = np.convolve(self.g_ext, dW1, mode="full")
            start = N
            end = N + n
            base = conv_full[start:end]

            X[p] = (
                base
                + We[p, 1, start:end] * self.L_t2_0
                + We[p, 2, start - 1 : end - 1] * self.L_t2_1
                + We[p, 3, start - 2 : end - 2] * self.L_t2_2
            )
            # Brownian increments on (0,T] (we drop burn-in part)
            dW0[p] = dW1[start:end]

        return X, dW0


class RoughBergomiForwardSwapPricerHybrid:
    """
    Rough Bergomi forward swap model under the annuity measure Q*:

        dS*_t / S*_t = sqrt(V_t) dW*_t,

    with

        V_t = v(t) * exp( X_t - 0.5 Var[X_t] ),    X_t = ∫ ζ(t-s) dW^{0*}_s,
        ζ(u) = κ u^{H-1/2},   H in (0, 0.5].

    For H = 0.5, ζ(u) = κ and the model coincides with the lognormal SABR model
    with vol-of-vol ν = κ/2. In that case we simulate SABR directly.
    """

    def __init__(self, H, kappa, rho, v0,
                 S0=1.0,
                 T=1.0,
                 n_steps=200,
                 n_paths=10000,
                 gamma=0.5,
                 seed=None):
        """
        v_cap (optional): if not None, cap instantaneous variance at this value
        when updating S* to avoid numerical explosions.
        """
        self.H = float(H)
        if not (0.0 < self.H <= 0.5):
            raise ValueError("H must be in (0, 0.5].")

        # flag for the SABR limit case H = 1/2
        self.is_sabr = (abs(self.H - 0.5) < 1e-12)

        self.kappa = float(kappa)
        self.rho = float(rho)
        self.v0 = float(v0)
        self.S0 = float(S0)
        self.T = float(T)
        self.n_steps = int(n_steps)
        self.n_paths = int(n_paths)
        self.gamma = float(gamma)
        self.dt = self.T / self.n_steps
        self.rng = np.random.default_rng(seed)

        # For H < 1/2 we need the Volterra driver; for H = 1/2 we don't.
        if not self.is_sabr:
            a = self.H - 0.5
            Lfct = lambda x, kappa=self.kappa: kappa * np.ones_like(x)
            self.bss = HybridBSS(a, Lfct, self.n_steps, self.T, gamma=self.gamma)
        else:
            self.bss = None  # not used in SABR limit

    def _forward_variance_curve(self, t):
        """
        v(t) = v0 * exp( κ^2 t^{2H} / (8H) )
        (Sect. 6.1; ensures E[√V_t] is flat). For H = 0.5 this is
        v(t) = v0 * exp( κ^2 t / 4 ), consistent with the SABR limit.
        """
        t = np.asarray(t, dtype=float)
        out = np.empty_like(t)
        out[...] = self.v0
        mask = t > 0
        out[mask] = self.v0 * np.exp(
            (self.kappa ** 2) * (t[mask] ** (2.0 * self.H)) / (8.0 * self.H)
        )
        return out

    def _simulate_paths(self):
        """
        Simulate S*_t and V_t on {0, dt, ..., T}.

        - If H < 0.5: use HybridBSS for X_t and construct V_t as in rough Bergomi.
        - If H = 0.5: simulate lognormal SABR directly with ν = κ / 2.
        """
        n = self.n_steps
        dt = self.dt
        times = np.linspace(0.0, self.T, n + 1)

        # ----- SABR limit: H = 1/2 -----
        if self.is_sabr:
            # 1) Brownian for W^{0*}
            dW0 = self.rng.normal(size=(self.n_paths, n)) * np.sqrt(dt)

            # 2) Volatility process alpha_t = sqrt(V_t)
            nu = 0.5 * self.kappa  # SABR vol-of-vol
            alpha = np.empty((self.n_paths, n + 1), dtype=float)
            alpha[:, 0] = np.sqrt(self.v0)

            for i in range(n):
                # exact log-Euler for dα_t = ν α_t dW^{0*}_t
                alpha[:, i + 1] = alpha[:, i] * np.exp(
                    -0.5 * nu**2 * dt + nu * dW0[:, i]
                )

            V = alpha**2

            # 3) Correlated Brownian motion for S*_t:
            #    dW*_t = ρ dW^{0*}_t + sqrt(1-ρ^2) dZ_t
            dWperp = self.rng.normal(size=(self.n_paths, n)) * np.sqrt(dt)
            dWstar = self.rho * dW0 + np.sqrt(1.0 - self.rho ** 2) * dWperp

            # 4) Forward swap rate S*_t (Euler in S)
            S = np.empty((self.n_paths, n + 1), dtype=float)
            S[:, 0] = self.S0
            for i in range(n):
                vol_i = alpha[:, i]
                S[:, i + 1] = S[:, i] + S[:, i] * vol_i * dWstar[:, i]

            return times, S, V

        # ----- Rough case: 0 < H < 1/2 -----
        # 1) Volterra driver and W^{0*}
        X, dW0 = self.bss.sample(self.n_paths, rng=self.rng)  # X at t=dt,...,T

        # 2) Variance process V_t
        V = np.empty((self.n_paths, n + 1), dtype=float)
        V[:, 0] = self.v0
        for i in range(1, n + 1):
            t = times[i]
            varX = (self.kappa ** 2) * (t ** (2.0 * self.H)) / (2.0 * self.H)
            V[:, i] = self._forward_variance_curve(t) * np.exp(
                X[:, i - 1] - 0.5 * varX
            )

        # 3) Correlated Brownian motion for S*_t:
        dWperp = self.rng.normal(size=(self.n_paths, n)) * np.sqrt(dt)
        dWstar = self.rho * dW0 + np.sqrt(1.0 - self.rho ** 2) * dWperp

        # 4) Forward swap rate S*_t (Euler–Maruyama in S)
        S = np.empty((self.n_paths, n + 1), dtype=float)
        S[:, 0] = self.S0
        for i in range(n):
            vi = np.maximum(V[:, i], 0.0)
            S[:, i + 1] = S[:, i] + S[:, i] * np.sqrt(vi) * dWstar[:, i]

        return times, S, V

    def price_swaption(self, K, option_type="payer"):
        _, S, _ = self._simulate_paths()
        ST = S[:, -1]
        if option_type.lower() == "payer":
            payoff = np.maximum(ST - K, 0.0)
        elif option_type.lower() == "receiver":
            payoff = np.maximum(K - ST, 0.0)
        else:
            raise ValueError("option_type must be 'payer' or 'receiver'.")
        return float(payoff.mean())

    def price_put_logstrike(self, k):
        _, S, _ = self._simulate_paths()
        ST_over_S0 = S[:, -1] / self.S0
        payoff = np.maximum(np.exp(k) - ST_over_S0, 0.0)
        return float(payoff.mean())



pricer = RoughBergomiForwardSwapPricerHybrid(
    H=0.3,
    kappa=1.5,
    rho=-0.7,
    v0=0.04,
    S0=0.02,
    T=5.0,
    n_steps=500,
    n_paths=1000,
    gamma=0.5,
    seed=123,
)

K = 0.02
k = np.log(K / pricer.S0)
p_atm = pricer.price_put_logstrike(k)
print(p_atm)
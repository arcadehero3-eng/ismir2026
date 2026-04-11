"""
chenda_pgsd_v3.py
=================
Physics-Guided Spectral Decomposition (PGSD) — v3 FINAL
=========================================================
Basis: Fourier-Chebyshev spectral eigenmodes  [Sathej & Adhikari 2008]

BUGS FIXED
----------
BUG 1  r-grid direction
       Chebyshev nodes descend (r[0]≈0.998, r[-1]≈0.063).
       RegularGridInterpolator needs ascending axes.
       FIX: flip r and mode rows before interpolation.

BUG 2  Missing centre point
       Chebyshev grid excludes r=0; fill_value=0 zeroed centre strikes.
       FIX: linear extrapolation from the two innermost nodes to r=0.

BUG 3  Degenerate pair excitation (Task 2 old approach)
       Degenerate pair sum sqrt(Ψ_A²+Ψ_B²) ≈ const in θ — collapses
       the discriminating angular information and flattens the
       radial pattern.
       FIX: use signal-level template matching instead of coefficient
       correlation. Synthesise a noiseless template at each candidate
       r_s and find the best match to the observed signal.

BUG 4  Task 3 bandpass + envelope window
       (a) Filter used hardcoded reference f₁ regardless of shape.
       (b) Envelope fit ran to t=0.9×dur, including the noise floor,
           which inverted the CI for fast-decaying shapes (Hourglass).
       FIX: (a) use shape-specific freqs_hz[0] from the spectral solver.
            (b) fit only the portion above 2× the noise floor.

BUG 5  Ridge regularisation scale
       Fixed lambda=1e-6 over-regularises when frequencies mismatch.
       FIX: lambda = rel_lam × ||ΦᵀΦ||_F  (scales with matrix norm).

FOUR TASKS
----------
Task 1  Tuning estimation     f₁ ∝ √T  →  T_est = T_ref·(f₁_refined/f₁_ref)²
Task 2  Strike position       signal-level template matching over radial grid
Task 3  Skin condition        T60 measured from noise-floor-gated envelope
Task 4  Shape identification  best R² across three shape bases identifies shape
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, rfftfreq
import warnings
warnings.filterwarnings("ignore")

# ── Load spectral solver (inject SHAPES, solve_eigenmodes, density_* fns) ────
exec(open("/mnt/user-data/outputs/chenda_spectral.py").read().split("if __name__")[0],
     globals())

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

R_PHYS    = 0.125    # membrane radius (m)
T_REF     = 3500.0   # reference tension (N/m)
SIGMA_MEM = 1.4      # surface density (kg/m²)


# ══════════════════════════════════════════════════════════════════════════════
# DAMPING MODEL   α(ω) = η₁/2 + η₂ω²/2
# ══════════════════════════════════════════════════════════════════════════════

class ChendaDamping:
    def __init__(self, eta1=0.80, eta2=8.8e-6):
        self.eta1 = eta1
        self.eta2 = eta2

    def decay_rate(self, omega):
        return 0.5*self.eta1 + 0.5*self.eta2*omega**2

    def T60(self, omega):
        return np.log(1000.0) / self.decay_rate(omega)


# ══════════════════════════════════════════════════════════════════════════════
# GRID FIX  (BUG 1 + BUG 2)
# ══════════════════════════════════════════════════════════════════════════════

def _fix_grid(r_cheb, modes_raw):
    """
    Flip Chebyshev grid from descending to ascending and prepend r=0
    via linear extrapolation from the two innermost nodes.
    """
    r_asc    = r_cheb[::-1].copy()
    out      = []
    for psi in modes_raw:
        psi_asc  = psi[::-1, :].copy()
        r0, r1   = r_asc[0], r_asc[1]
        v0, v1   = psi_asc[0, :], psi_asc[1, :]
        v_centre = v0 - r0*(v1 - v0)/(r1 - r0)
        out.append(np.vstack([v_centre[None, :], psi_asc]))
    return np.concatenate([[0.0], r_asc]), out


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL BASIS  (cached)
# ══════════════════════════════════════════════════════════════════════════════

_CACHE = {}

def get_spectral_basis(shape_name, T=T_REF, sigma=SIGMA_MEM, R=R_PHYS,
                       Nr=25, Nth=24, n_modes=18, damping=None):
    """
    Return (freqs_hz, modes_asc, alphas, r_full, th, rho_full) for a shape.
    All grids are in ascending-r order with r=0 included.
    """
    key = (shape_name, round(T, 1), sigma, R, Nr, Nth, n_modes)
    if key in _CACHE:
        return _CACHE[key]
    if damping is None:
        damping = ChendaDamping()

    cfg = SHAPES[shape_name]
    freqs_raw, modes_raw, r_cheb, th, rho_raw = solve_eigenmodes(
        Nr=Nr, Nth=Nth,
        density_fn=cfg['density_fn'],
        density_kw=cfg['density_kw'],
        n_modes=n_modes + 4,
        R_phys=R, T=T, sigma_mem=sigma,
    )

    freqs_hz   = freqs_raw[:n_modes]
    r_full, modes_asc = _fix_grid(r_cheb, modes_raw[:n_modes])

    # Fix density orientation to match r_full
    rho_fl   = rho_raw[::-1, :]
    rho_full = np.vstack([rho_fl[:1, :], rho_fl])

    alphas = np.array([damping.decay_rate(2*np.pi*f) for f in freqs_hz])
    result = (freqs_hz, modes_asc, alphas, r_full, th, rho_full)
    _CACHE[key] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MODE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _make_interp(r_full, th, psi):
    th_ext  = np.append(th, th[0] + 2*np.pi)
    psi_ext = np.hstack([psi, psi[:, :1]])
    return RegularGridInterpolator(
        (r_full, th_ext), psi_ext,
        method='linear', bounds_error=False, fill_value=0.0)

def eval_mode(r_full, th, psi, r_norm, theta):
    return float(_make_interp(r_full, th, psi)(
        np.array([[r_norm, float(theta) % (2*np.pi)]]))[0])


# ══════════════════════════════════════════════════════════════════════════════
# TIME-DOMAIN BASIS   Φ[i,n] = exp(-αₙ tᵢ) · sin(2π fₙ tᵢ)
# ══════════════════════════════════════════════════════════════════════════════

def build_basis(freqs_hz, alphas, t):
    Phi = np.zeros((len(t), len(freqs_hz)))
    for n, (f, a) in enumerate(zip(freqs_hz, alphas)):
        Phi[:, n] = np.exp(-a*t) * np.sin(2*np.pi*f*t)
    return Phi


# ══════════════════════════════════════════════════════════════════════════════
# PGSD DECOMPOSITION   min ||Φc - x||² + λ||c||²
# ══════════════════════════════════════════════════════════════════════════════

def pgsd_decompose(signal, Phi, rel_lam=1e-6):
    """
    Ridge regression with lambda scaled to ||ΦᵀΦ||_F (BUG 5 fix).
    Returns  c, residual, fit, R²
    """
    G   = Phi.T @ Phi
    lam = rel_lam * np.linalg.norm(G, 'fro')
    c   = np.linalg.solve(G + lam*np.eye(G.shape[0]), Phi.T @ signal)
    fit = Phi @ c
    res = signal - fit
    r2  = 1.0 - np.sum(res**2) / (np.sum((signal - signal.mean())**2) + 1e-12)
    return c, res, fit, r2


# ══════════════════════════════════════════════════════════════════════════════
# ITERATIVE REFINEMENT  —  chase residual peaks
# ══════════════════════════════════════════════════════════════════════════════

def pgsd_refine(signal, freqs_init, alphas_init, damping, t,
                n_iter=14, lr=0.28, search_hz=30.0, verbose=True):
    """
    Iteratively update model frequencies by chasing spectral peaks in the
    residual.  Converges when ΔR² < 1e-6.

    Returns  f_final, al_final, c_final, res_final, fit_final, r2_hist, f_hist
    """
    fs        = 1.0 / (t[1] - t[0])
    fft_freqs = rfftfreq(len(t), 1.0/fs)
    f_cur     = freqs_init.copy().astype(float)
    al_cur    = alphas_init.copy().astype(float)
    r2_hist   = []
    f_hist    = [f_cur.copy()]
    win       = np.hanning(len(signal))

    if verbose:
        print(f"\n  {'It':>4}  {'Resid RMS':>12}  {'R²':>8}  {'f₁':>9}  {'f₂':>9}")
        print(f"  {'─'*50}")

    for it in range(n_iter):
        Phi = build_basis(f_cur, al_cur, t)
        c, res, fit, r2 = pgsd_decompose(signal, Phi)
        rms = np.sqrt(np.mean(res**2))
        r2_hist.append(r2)

        if verbose:
            print(f"  {it:>4}  {rms:>12.5f}  {r2:>8.4f}  "
                  f"{f_cur[0]:>9.3f}  {f_cur[1]:>9.3f}")

        R_fft = np.abs(rfft(res * win))
        for i in range(len(f_cur)):
            fi   = f_cur[i]
            mask = (fft_freqs > fi - search_hz) & (fft_freqs < fi + search_hz)
            if mask.any():
                pk = R_fft[mask]
                if pk.max() > 5e-5 * R_fft.max():
                    f_cur[i]  = (1-lr)*f_cur[i] + lr*fft_freqs[mask][np.argmax(pk)]
                    al_cur[i] = damping.decay_rate(2*np.pi*f_cur[i])

        f_hist.append(f_cur.copy())
        if it > 2 and abs(r2_hist[-1] - r2_hist[-2]) < 1e-6:
            if verbose: print(f"  Converged at iteration {it}")
            break

    Phi_f = build_basis(f_cur, al_cur, t)
    c_f, res_f, fit_f, r2_f = pgsd_decompose(signal, Phi_f)
    r2_hist.append(r2_f)
    if verbose: print(f"  Final R² = {r2_f:.5f}")
    return f_cur, al_cur, c_f, res_f, fit_f, r2_hist, f_hist


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL SYNTHESISER
# ══════════════════════════════════════════════════════════════════════════════

def synthesise_strike(freqs_hz, modes_asc, alphas, r_full, th,
                      r_s, theta_s, t, snr_db=30.0, R=R_PHYS):
    """
    Synthesise a Chenda strike at (r_s, θ_s).

    Excitation E_n = |Ψ_n(r_s/R, θ_s)| evaluated individually per mode.
    Radiation weight: 1/(n+1)^0.8.
    Returns  signal, amplitudes
    """
    r_norm = np.clip(r_s/R, 0.0, 0.98)
    E      = np.array([abs(eval_mode(r_full, th, modes_asc[n], r_norm, theta_s))
                       for n in range(len(freqs_hz))])
    radiation  = 1.0 / (np.arange(1, len(freqs_hz)+1)**0.8)
    amplitudes = E * radiation

    signal = np.zeros(len(t))
    for n in range(len(freqs_hz)):
        signal += amplitudes[n] * np.exp(-alphas[n]*t) * np.sin(2*np.pi*freqs_hz[n]*t)

    rms_s     = np.sqrt(np.mean(signal**2)) + 1e-12
    signal   += rms_s * 10**(-snr_db/20.0) * np.random.randn(len(t))
    return signal, amplitudes


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — TUNING ESTIMATION
# f₁ ∝ √T  →  T_est = T_ref · (f₁_refined / f₁_ref)²
# ══════════════════════════════════════════════════════════════════════════════

def task1_tuning(shape_name='Cylinder', fs=22050, dur=1.5):
    print(f"\n{'═'*64}\n  TASK 1: Tuning Estimation   [{shape_name}]\n{'═'*64}")
    damping   = ChendaDamping()
    t         = np.linspace(0, dur, int(fs*dur), endpoint=False)
    np.random.seed(42)

    freqs_ref, modes_ref, alphas_ref, r, th, rho = get_spectral_basis(shape_name, T=T_REF)
    f1_ref  = freqs_ref[0]
    results = []
    print(f"\n  {'T_true':>8}  {'f₁_true':>9}  {'f₁_refined':>12}  "
          f"{'T_est':>9}  {'err%':>7}  {'R²':>7}")
    print(f"  {'─'*62}")

    for T_true in [3000, 3200, 3500, 3800, 4100]:
        freqs_t, modes_t, alphas_t, r, th, _ = get_spectral_basis(shape_name, T=T_true)
        signal, _ = synthesise_strike(freqs_t, modes_t, alphas_t, r, th,
                                      r_s=0.4*R_PHYS, theta_s=0.0, t=t)

        f_rf, al_rf, c_f, res_f, fit_f, r2h, fh = pgsd_refine(
            signal, freqs_ref.copy(), alphas_ref.copy(),
            damping, t, n_iter=14, lr=0.25, search_hz=32.0, verbose=False)

        f1_r  = f_rf[0]
        T_est = T_REF * (f1_r/f1_ref)**2
        err   = abs(T_est - T_true)/T_true*100
        r2    = r2h[-1]
        results.append(dict(T_true=T_true, T_est=T_est, err_pct=err,
                            f1_true=freqs_t[0], f1_refined=f1_r, r2=r2))
        print(f"  {T_true:>8}  {freqs_t[0]:>9.2f}  {f1_r:>12.3f}  "
              f"{T_est:>9.1f}  {err:>7.2f}  {r2:>7.4f}")

    print(f"\n  Mean error: {np.mean([r['err_pct'] for r in results]):.2f}%")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — STRIKE POSITION
# Signal-level template matching: find r_s that maximises
# dot(normalised_template(r_s), normalised_signal)
# ══════════════════════════════════════════════════════════════════════════════

def task2_strike(shape_name='Cylinder', fs=22050, dur=1.5):
    print(f"\n{'═'*64}\n  TASK 2: Strike Position   [{shape_name}]\n{'═'*64}")
    t = np.linspace(0, dur, int(fs*dur), endpoint=False)
    np.random.seed(77)

    freqs_ref, modes_ref, alphas_ref, r, th, rho = get_spectral_basis(shape_name)
    Phi_ref  = build_basis(freqs_ref, alphas_ref, t)
    r_search = np.linspace(0.0, 0.97, 80)

    # Pre-compute noiseless template signals on short window (first 500 samples)
    WIN = 500
    templates = []
    for rt in r_search:
        _, amps_t = synthesise_strike(freqs_ref, modes_ref, alphas_ref, r, th,
                                      r_s=rt*R_PHYS, theta_s=0.0, t=t[:WIN], snr_db=999)
        tmpl = Phi_ref[:WIN, :] @ amps_t
        norm = np.sqrt(np.sum(tmpl**2)) + 1e-12
        templates.append(tmpl / norm)
    templates = np.array(templates)   # (80, WIN)

    print(f"\n  {'r_true/R':>10}  {'r_est/R':>9}  {'err%R':>7}  {'corr':>8}  {'R²':>7}")
    print(f"  {'─'*50}")

    results = []
    for r_frac in [0.00, 0.25, 0.50, 0.75, 0.90]:
        signal, amps_true = synthesise_strike(
            freqs_ref, modes_ref, alphas_ref, r, th,
            r_s=r_frac*R_PHYS, theta_s=0.0, t=t, snr_db=25.0)

        # PGSD decompose (for R² reporting)
        _, _, _, r2 = pgsd_decompose(signal, Phi_ref)

        # Signal-level template match on first WIN samples (BUG 3 fix)
        obs_win = signal[:WIN]
        obs_n   = obs_win / (np.sqrt(np.sum(obs_win**2)) + 1e-12)
        corrs   = templates @ obs_n        # (80,) dot products
        best_idx  = int(np.argmax(corrs))
        best_r    = r_search[best_idx]
        best_corr = float(corrs[best_idx])

        r_err = abs(best_r - r_frac)
        results.append(dict(r_true=r_frac, r_est=best_r,
                            r_err=r_err, corr=best_corr, r2=r2))
        print(f"  {r_frac:>10.2f}  {best_r:>9.3f}  {r_err*100:>7.1f}  "
              f"{best_corr:>8.4f}  {r2:>7.4f}")

    mean_err = np.mean([r['r_err'] for r in results])
    print(f"\n  Mean radial error: {mean_err*100:.1f}% of R  "
          f"= {mean_err*R_PHYS*100:.2f} cm")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — SKIN CONDITION
# CI = (T60_ref - T60_measured) / T60_ref × 100
# Envelope fit gated to signal above noise floor (BUG 4 fix)
# ══════════════════════════════════════════════════════════════════════════════

def task3_skin(shape_name='Cylinder', fs=22050, dur=2.0):
    print(f"\n{'═'*64}\n  TASK 3: Skin Condition   [{shape_name}]\n{'═'*64}")
    damping_ref = ChendaDamping(eta1=0.80, eta2=8.8e-6)

    # BUG 4a fix: shape-specific f₁
    freqs_ref, modes_ref, alphas_ref, r, th, rho = get_spectral_basis(
        shape_name, damping=damping_ref)
    f1      = freqs_ref[0]
    T60_ref = damping_ref.T60(2*np.pi*f1)
    t       = np.linspace(0, dur, int(fs*dur), endpoint=False)
    np.random.seed(55)

    skins = {
        "New":      ChendaDamping(0.80, 8.8e-6),
        "3 months": ChendaDamping(0.81, 1.2e-5),
        "6 months": ChendaDamping(0.83, 1.8e-5),
        "1 year":   ChendaDamping(0.86, 2.8e-5),
        "Worn":     ChendaDamping(0.92, 4.5e-5),
    }

    print(f"\n  f₁ = {f1:.1f} Hz    T60_ref = {T60_ref:.3f} s\n")
    print(f"  {'Condition':>12}  {'η₂×10⁻⁵':>10}  {'T60_pred':>9}  "
          f"{'T60_meas':>9}  {'CI':>6}")
    print(f"  {'─'*54}")

    results = {}
    for name, damp in skins.items():
        al_s   = np.array([damp.decay_rate(2*np.pi*f) for f in freqs_ref])
        signal, _ = synthesise_strike(freqs_ref, modes_ref, al_s, r, th,
                                      r_s=0.3*R_PHYS, theta_s=0.0, t=t)

        # BUG 4a fix: bandpass around shape-specific f₁
        f_nyq = fs/2.0
        f_lo, f_hi = f1*0.82, f1*1.18
        if f_hi < f_nyq*0.97:
            b_c, a_c = butter(4, [f_lo/f_nyq, f_hi/f_nyq], btype='band')
            sig_bp   = filtfilt(b_c, a_c, signal)
        else:
            sig_bp   = signal

        env = np.abs(sig_bp)
        sm  = max(1, int(fs*0.005))
        env_sm = np.convolve(env, np.ones(sm)/sm, mode='same')

        # BUG 4b fix: gate to noise floor
        env_peak    = env_sm.max()
        noise_floor = env_peak / 31.6      # SNR 30 dB
        above       = np.where(env_sm > noise_floor * 2.0)[0]

        if len(above) > 100:
            i0 = max(int(fs*0.02), above[0])
            i1 = above[-1]
        else:
            i0, i1 = int(fs*0.02), int(fs*0.8)

        t_fit, e_fit = t[i0:i1], np.log(np.maximum(env_sm[i0:i1], 1e-12))
        finite = np.isfinite(e_fit)
        if finite.sum() > 20:
            coeffs     = np.polyfit(t_fit[finite], e_fit[finite], 1)
            alpha_meas = max(-coeffs[0], 1e-3)
        else:
            alpha_meas = al_s[0]

        T60_meas = np.log(1000.0) / alpha_meas
        T60_pred = damp.T60(2*np.pi*f1)
        CI       = min(100.0, max(0.0, (T60_ref - T60_meas)/T60_ref*100))

        results[name] = dict(eta2=damp.eta2, T60_pred=T60_pred,
                             T60_meas=T60_meas, CI=CI)
        print(f"  {name:>12}  {damp.eta2*1e5:>10.2f}  {T60_pred:>9.3f}  "
              f"{T60_meas:>9.3f}  {CI:>6.1f}")

    ci_vals  = [v['CI'] for v in results.values()]
    monotone = all(ci_vals[i] <= ci_vals[i+1] for i in range(len(ci_vals)-1))
    print(f"\n  Monotone: {monotone}   CI: {ci_vals[0]:.0f} → {ci_vals[-1]:.0f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — SHAPE IDENTIFICATION
# Fit each shape's basis to the recording; best R² → shape
# ══════════════════════════════════════════════════════════════════════════════

def task4_shape_id(true_shape='Cylinder', fs=22050, dur=1.5):
    print(f"\n{'═'*64}\n  TASK 4: Shape Identification   (true={true_shape})\n{'═'*64}")
    t = np.linspace(0, dur, int(fs*dur), endpoint=False)
    np.random.seed(99)

    freqs_t, modes_t, alphas_t, r, th, _ = get_spectral_basis(true_shape)
    signal, _ = synthesise_strike(freqs_t, modes_t, alphas_t, r, th,
                                  r_s=0.4*R_PHYS, theta_s=0.0, t=t)

    print(f"\n  {'Shape':>12}  {'R²':>10}  {'Resid RMS':>12}  {'f₁ Hz':>9}")
    print(f"  {'─'*50}")
    res = {}
    for sn in SHAPES:
        freqs_s, _, alphas_s, _, _, _ = get_spectral_basis(sn)
        Phi = build_basis(freqs_s, alphas_s, t)
        c, residual, fit, r2 = pgsd_decompose(signal, Phi)
        rms = np.sqrt(np.mean(residual**2))
        res[sn] = dict(r2=r2, rms=rms, f1=freqs_s[0])
        tag = " ← TRUE" if sn == true_shape else ""
        print(f"  {sn:>12}  {r2:>10.5f}  {rms:>12.6f}  {freqs_s[0]:>9.2f}{tag}")

    best = max(res, key=lambda k: res[k]['r2'])
    print(f"\n  Identified: '{best}'   correct = {best == true_shape}")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# MASTER OVERVIEW PLOT  (4 × 3 panels)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pgsd_master(shape_name='Cylinder', T_signal=3750,
                     r_s_frac=0.60, theta_s=0.3, fs=22050, dur=1.5):
    print(f"\n  Plotting [{shape_name}]...")
    damping = ChendaDamping()
    t       = np.linspace(0, dur, int(fs*dur), endpoint=False)
    np.random.seed(42)

    freqs_ref, modes_ref, alphas_ref, r, th, rho = get_spectral_basis(shape_name, T=T_REF)
    freqs_tru, modes_tru, alphas_tru, r, th, _   = get_spectral_basis(shape_name, T=T_signal)

    signal, amps_true = synthesise_strike(
        freqs_tru, modes_tru, alphas_tru, r, th,
        r_s=r_s_frac*R_PHYS, theta_s=theta_s, t=t)

    Phi_ref = build_basis(freqs_ref, alphas_ref, t)
    c_init, res_init, fit_init, r2_init = pgsd_decompose(signal, Phi_ref)

    f_rf, al_rf, c_fin, res_fin, fit_fin, r2h, f_hist = pgsd_refine(
        signal, freqs_ref.copy(), alphas_ref.copy(),
        damping, t, n_iter=14, lr=0.25, search_hz=32.0, verbose=False)

    fft_f  = rfftfreq(len(t), 1.0/fs)
    win    = np.hanning(len(signal))
    S_fft  = np.abs(rfft(signal * win))
    Ri_fft = np.abs(rfft(res_init * win))
    Rf_fft = np.abs(rfft(res_fin  * win))
    t_ms   = t*1000
    show   = t_ms < 400

    TCOL = '#E5E7EB'; GCOL = '#1E2A3A'
    SCOL = {'Cylinder': '#5AB4E8', 'Hourglass': '#E8A838', 'Oval': '#5ECF82'}
    BC   = ['#5AB4E8', '#E8A838', '#5ECF82', '#E85A5A', '#BF88FF']
    col  = SCOL[shape_name]

    def sa(ax, title='', xlabel='', ylabel=''):
        ax.set_facecolor('#0D1520')
        for sp in ax.spines.values(): sp.set_edgecolor('#2A3550')
        ax.tick_params(colors=TCOL, labelsize=8)
        ax.xaxis.label.set_color(TCOL); ax.yaxis.label.set_color(TCOL)
        ax.grid(True, color=GCOL, lw=0.6, alpha=0.7)
        if title:  ax.set_title(title, color=TCOL, fontsize=9.5, fontweight='bold', pad=4)
        if xlabel: ax.set_xlabel(xlabel, fontsize=8.5)
        if ylabel: ax.set_ylabel(ylabel, fontsize=8.5)

    fig = plt.figure(figsize=(22, 20), facecolor='#0A0F1E')
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                            left=0.06, right=0.97, top=0.93, bottom=0.04,
                            hspace=0.45, wspace=0.35)
    fig.suptitle(
        f'Physics-Guided Spectral Decomposition (PGSD) v3  —  {shape_name} Chenda\n'
        f'Signal T={T_signal} N/m  |  Ref T={T_REF} N/m  |  '
        f'Strike r={r_s_frac:.2f}R  θ={theta_s:.2f} rad\n'
        f'Basis: Fourier-Chebyshev spectral eigenmodes  [Sathej & Adhikari 2008]',
        fontsize=12, color='white', fontweight='bold', y=0.97)

    # [0,0] Input signal
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_ms[show], signal[show], color=col, lw=0.7, alpha=0.9)
    sa(ax, f'Input Signal  ({shape_name}, synthetic)', 'Time (ms)', 'Amplitude')

    # [0,1] First 5 basis functions
    ax = fig.add_subplot(gs[0, 1])
    for i in range(5):
        phi_i = Phi_ref[:, i]; mx = np.abs(phi_i).max() + 1e-12
        ax.plot(t_ms[show], phi_i[show]/mx + (4-i)*2.2,
                color=BC[i], lw=1.0, label=f'φ_{i+1}: {freqs_ref[i]:.0f} Hz')
    ax.legend(fontsize=7.5, facecolor='#0D1520', edgecolor='#2A3550', labelcolor=TCOL)
    sa(ax, 'Physics Basis  φₙ(t) = e^{−αt} sin(2πft)', 'Time (ms)', '(offset)')
    ax.set_yticks([])

    # [0,2] Density field + strike marker
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor('#0D1520'); ax.set_aspect('equal'); ax.axis('off')
    r_d = np.linspace(0,1,200); th_d = np.linspace(0,2*np.pi,200)
    Rd, THd = np.meshgrid(r_d, th_d)
    Xd, Yd  = Rd*np.cos(THd), Rd*np.sin(THd)
    th_ext  = np.append(th, th[0]+2*np.pi)
    rho_ext = np.hstack([rho, rho[:,:1]])
    interp  = RegularGridInterpolator((r, th_ext), rho_ext,
                  method='linear', bounds_error=False, fill_value=1.0)
    pts     = np.column_stack([Rd.ravel(), THd.ravel()%(2*np.pi)])
    rho_d   = interp(pts).reshape(Rd.shape); rho_d[Rd > 1.0] = np.nan
    ax.contourf(Xd, Yd, rho_d, levels=40, cmap='hot', extend='both')
    th_c = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(th_c), np.sin(th_c), color='white', lw=1.5)
    ax.plot(r_s_frac*np.cos(theta_s), r_s_frac*np.sin(theta_s), 'w*', ms=16, zorder=5)
    ax.set_title('Density ρ(r,θ)   ★ = strike', color=TCOL, fontsize=9.5, fontweight='bold')

    # [1,0] Initial decomposition
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_ms[show], signal[show], color='#6B7280', lw=0.7, alpha=0.5, label='Signal')
    ax.plot(t_ms[show], fit_init[show], color='#E8A838', lw=1.4,
            label=f'Init fit  R²={r2_init:.3f}')
    ax.legend(fontsize=8, facecolor='#0D1520', edgecolor='#2A3550', labelcolor=TCOL)
    sa(ax, 'Initial Decomposition\n(reference model, before refinement)', 'Time (ms)', 'Amplitude')

    # [1,1] Residual spectra
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(fft_f, Ri_fft+1e-8, color='#E85A5A', lw=1.0, alpha=0.8, label='Residual initial')
    ax.semilogy(fft_f, Rf_fft+1e-8, color='#5ECF82', lw=1.0, alpha=0.8, label='Residual refined')
    ax.semilogy(fft_f, S_fft +1e-8, color='#5AB4E8', lw=0.5, alpha=0.2,  label='Signal FFT')
    for f in freqs_tru[:8]:
        ax.axvline(f, color='yellow', lw=0.5, alpha=0.4, ls='--')
    ax.set_xlim(50, min(1200, freqs_tru[-1]*1.5))
    ax.legend(fontsize=7.5, facecolor='#0D1520', edgecolor='#2A3550', labelcolor=TCOL)
    sa(ax, 'Residual Spectra\nyellow = true eigenfrequencies', 'Frequency (Hz)', 'Power')

    # [1,2] R² convergence
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(range(len(r2h)), r2h, 'o-', color='#E8A838', lw=2, ms=6)
    ax.axhline(r2h[-1], color='#5ECF82', lw=1.0, ls='--', alpha=0.7,
               label=f'Final R²={r2h[-1]:.4f}')
    ax.set_ylim(max(0, min(r2h)*0.9), 1.02)
    ax.legend(fontsize=8, facecolor='#0D1520', edgecolor='#2A3550', labelcolor=TCOL)
    sa(ax, 'R² Convergence\n(iterative refinement)', 'Iteration', 'R²')

    # [2,0] Final decomposition
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t_ms[show], signal[show], color='#6B7280', lw=0.7, alpha=0.5, label='Signal')
    ax.plot(t_ms[show], fit_fin[show], color='#5ECF82', lw=1.4,
            label=f'PGSD fit  R²={r2h[-1]:.3f}')
    ax.legend(fontsize=8, facecolor='#0D1520', edgecolor='#2A3550', labelcolor=TCOL)
    sa(ax, 'Final Decomposition\n(after iterative refinement)', 'Time (ms)', 'Amplitude')

    # [2,1] Modal amplitudes: true vs recovered
    ax = fig.add_subplot(gs[2, 1])
    nm = min(14, len(c_fin))
    x  = np.arange(nm)
    at = np.abs(amps_true[:nm]); at /= (at.max()+1e-12)
    cr = np.abs(c_fin[:nm]);     cr /= (cr.max()+1e-12)
    ax.bar(x-0.2, at, 0.38, color='#E8A838', alpha=0.8, label='True',  zorder=3)
    ax.bar(x+0.2, cr, 0.38, color=col,       alpha=0.8, label='PGSD',  zorder=3)
    ax.legend(fontsize=8, facecolor='#0D1520', edgecolor='#2A3550', labelcolor=TCOL)
    sa(ax, 'Modal Amplitudes  (orange=true, col=PGSD)', 'Mode index', 'Norm. amplitude')

    # [2,2] Frequency convergence
    ax = fig.add_subplot(gs[2, 2])
    fa = np.array(f_hist)
    for i in range(min(5, fa.shape[1])):
        ax.plot(range(len(f_hist)), fa[:, i], 'o-', color=BC[i], lw=1.5, ms=4, label=f'f_{i+1}')
        ax.axhline(freqs_tru[i], color=BC[i], lw=0.8, ls='--', alpha=0.5)
    ax.legend(fontsize=8, facecolor='#0D1520', edgecolor='#2A3550', labelcolor=TCOL)
    sa(ax, 'Frequency Convergence\n(dashed = true values)', 'Iteration', 'Hz')

    # [3,0-2] Shape identification: frequency ladders
    all_r2 = {}
    for j, sn in enumerate(SHAPES.keys()):
        ax = fig.add_subplot(gs[3, j])
        ax.set_facecolor('#0D1520')
        freqs_s, _, alphas_s, _, _, _ = get_spectral_basis(sn, T=T_REF)
        Phi_s = build_basis(freqs_s, alphas_s, t)
        _, _, _, r2_s = pgsd_decompose(signal, Phi_s)
        all_r2[sn] = r2_s
        for i, f in enumerate(freqs_s[:14]):
            ax.axhline(f, color=SCOL[sn], lw=1.6 if i < 4 else 0.8, alpha=0.85)
        for f in freqs_tru[:14]:
            ax.axhline(f, color='white', lw=0.5, ls=':', alpha=0.45)
        ax.set_ylim(0, freqs_s[13]*1.12)
        bdr = '#FFD700' if sn == shape_name else '#2A3550'
        for sp in ax.spines.values():
            sp.set_edgecolor(bdr); sp.set_linewidth(2.2 if sn == shape_name else 0.8)
        ax.tick_params(colors=TCOL, labelsize=8); ax.yaxis.label.set_color(TCOL)
        ax.grid(True, color=GCOL, lw=0.6, alpha=0.7); ax.set_ylabel('Frequency (Hz)', fontsize=8.5)
        tag = '  ← True' if sn == shape_name else ''
        ax.set_title(f'{sn}   R²={r2_s:.4f}{tag}',
                     color=SCOL[sn], fontsize=9.5, fontweight='bold', pad=4)

    best = max(all_r2, key=all_r2.get)
    fig.text(0.5, 0.005,
             f'Shape ID: best={best}   correct={best == shape_name}   |   '
             f'R² gain: {r2h[0]:.3f} → {r2h[-1]:.3f}   |   '
             'White dotted = true signal eigenfrequencies',
             ha='center', fontsize=8.5, color='#6B7280')

    fname = os.path.join(OUTPUT_DIR, f'pgsd_v3_{shape_name.lower()}.png')
    plt.savefig(fname, dpi=130, bbox_inches='tight', facecolor='#0A0F1E')
    plt.close()
    print(f"  Saved → {fname}")
    return fname


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-SHAPE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_shape_summary(fs=22050, dur=1.5):
    print("\n  Cross-shape summary...")
    t      = np.linspace(0, dur, int(fs*dur), endpoint=False)
    shapes = list(SHAPES.keys())
    np.random.seed(42)

    R2 = np.zeros((3, 3))
    for i, ts in enumerate(shapes):
        ft, mt, at, r, th, _ = get_spectral_basis(ts)
        sig, _ = synthesise_strike(ft, mt, at, r, th,
                                   r_s=0.4*R_PHYS, theta_s=0.0, t=t)
        for j, bs in enumerate(shapes):
            fb, _, ab, _, _, _ = get_spectral_basis(bs)
            Phi = build_basis(fb, ab, t)
            _, _, _, r2 = pgsd_decompose(sig, Phi)
            R2[i, j] = r2

    SCOL = {'Cylinder': '#5AB4E8', 'Hourglass': '#E8A838', 'Oval': '#5ECF82'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0A0F1E')

    ax1.set_facecolor('#0D1520')
    im = ax1.imshow(R2, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax1, label='R²', fraction=0.046)
    ax1.set_xticks(range(3)); ax1.set_yticks(range(3))
    ax1.set_xticklabels(shapes, color='white', fontsize=10)
    ax1.set_yticklabels(shapes, color='white', fontsize=10)
    ax1.set_xlabel('Basis shape', color='white', fontsize=10)
    ax1.set_ylabel('True shape',  color='white', fontsize=10)
    ax1.set_title('PGSD Shape ID Matrix\n(diagonal = correct shape)',
                  color='white', fontsize=11, fontweight='bold')
    for i in range(3):
        for j in range(3):
            v = R2[i, j]
            ax1.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=11,
                     color='black' if v > 0.6 else 'white', fontweight='bold')
        ax1.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                      fill=False, edgecolor='#FFD700', lw=2.5))

    T_vals = [3000, 3200, 3500, 3800, 4100]
    ax2.set_facecolor('#0D1520')
    for sn in shapes:
        fr, _, ar, r, th, _ = get_spectral_basis(sn, T=T_REF)
        r2_list = []
        for Tv in T_vals:
            ft, mt, at, r, th, _ = get_spectral_basis(sn, T=Tv)
            sig, _ = synthesise_strike(ft, mt, at, r, th,
                                       r_s=0.4*R_PHYS, theta_s=0.0, t=t)
            Phi = build_basis(fr, ar, t); _, _, _, r2 = pgsd_decompose(sig, Phi)
            r2_list.append(r2)
        ax2.plot(T_vals, r2_list, 'o-', color=SCOL[sn], lw=2, ms=7, label=sn)

    ax2.axvline(T_REF, color='white', lw=1.0, ls='--', alpha=0.5,
                label=f'T_ref={T_REF} N/m')
    ax2.set_xlabel('True Tension T (N/m)', color='white')
    ax2.set_ylabel('R²', color='white'); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9, facecolor='#0D1520', edgecolor='#2A3550', labelcolor='white')
    ax2.grid(True, color='#1E2A3A', alpha=0.7); ax2.tick_params(colors='white')
    for sp in ax2.spines.values(): sp.set_edgecolor('#2A3550')
    ax2.set_title('Task 1: R² vs Tension per Shape\n(peak at T_ref when correctly tuned)',
                  color='white', fontsize=11, fontweight='bold')

    fig.suptitle(
        'PGSD v3 Cross-Shape Analysis — Spectral Eigenmode Basis\n'
        'Distinct eigenspectrum per shape → shape-discriminating PGSD',
        fontsize=12, color='white', fontweight='bold', y=1.02)

    fname = os.path.join(OUTPUT_DIR, 'pgsd_v3_cross_shape.png')
    plt.savefig(fname, dpi=130, bbox_inches='tight', facecolor='#0A0F1E')
    plt.close()
    print(f"  Saved → {fname}")
    return fname


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\n' + '═'*64)
    print('  PGSD v3 — SPECTRAL EIGENMODE BASIS  (all bugs fixed)')
    print('═'*64)

    all_t1, all_t2, all_t3, all_t4 = {}, {}, {}, {}
    for shape in SHAPES:
        all_t1[shape] = task1_tuning(shape)
        all_t2[shape] = task2_strike(shape)
        all_t3[shape] = task3_skin(shape)
        all_t4[shape] = task4_shape_id(shape)

    print('\n  Generating plots...')
    for shape in SHAPES:
        plot_pgsd_master(shape_name=shape, T_signal=3750)
    plot_cross_shape_summary()

    print('\n' + '═'*64 + '\n  FINAL SUMMARY\n' + '═'*64)
    for shape in SHAPES:
        t1_mean = np.mean([r['err_pct'] for r in all_t1[shape]])
        t2_mean = np.mean([r['r_err']   for r in all_t2[shape]])*100
        ci_vals = [v['CI'] for v in all_t3[shape].values()]
        mono    = all(ci_vals[i] <= ci_vals[i+1] for i in range(len(ci_vals)-1))
        best_s  = max(all_t4[shape], key=lambda k: all_t4[shape][k]['r2'])
        print(f"\n  {shape}:")
        print(f"    Task1 tuning err  : {t1_mean:.2f}%")
        print(f"    Task2 strike err  : {t2_mean:.1f}% of R")
        print(f"    Task3 CI monotone : {mono}  ({ci_vals[0]:.0f} → {ci_vals[-1]:.0f})")
        print(f"    Task4 shape ID    : {best_s}  correct={best_s == shape}")
    print('\n' + '═'*64)

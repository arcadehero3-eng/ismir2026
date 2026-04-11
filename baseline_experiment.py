"""
baseline_experiment.py
======================
Runs FFT peak tracking (FFT-PT) and NMF modal extraction (NMF) on
exactly the same synthetic test signals used in PGSD v3, across all
four diagnostic tasks and all three shell geometries.

Baselines
---------
FFT-PT  — Highest-amplitude FFT peak in 80–300 Hz gives f̂₁.
          Tension: T̂ = T_ref · (f̂₁ / f₁_ref)²
          Skin:    T60 from fixed-width bandpass around f̂₁
          Position: N/A (no mode shapes)
          Shape:   N/A (no shape-specific basis)

NMF     — K=18 NMF components on STFT magnitude spectrogram.
          Sorted by centroid frequency; lowest component gives f̂₁.
          Tension: same inversion as FFT-PT but using NMF f̂₁
          Skin:    component activation decay rate → T60
          Position: N/A (no mode shapes)
          Shape:   nearest-neighbour on component centroid vector

All random seeds, tensions, positions, skin conditions, shapes and
signal parameters are identical to those used in chenda_pgsd_v3.py.
"""

import sys, os
import numpy as np
from scipy.signal import butter, filtfilt, stft
from scipy.linalg import norm
from numpy.fft import rfft, rfftfreq
import warnings
warnings.filterwarnings("ignore")

# ── Load v3 (gives us synthesise_strike, get_spectral_basis, SHAPES etc.) ───
src = open("/mnt/user-data/outputs/chenda_pgsd_v3.py").read()
exec(compile(src[:src.rfind("\nif __name__")], "pgsd_v3", "exec"), globals())

FS   = 22050
T_TENSIONS  = [3000, 3200, 3500, 3800, 4100]
R_POSITIONS = [0.00, 0.25, 0.50, 0.75, 0.90]
SHAPES_LIST = ["Cylinder", "Hourglass", "Oval"]
SKINS = {
    "New":      ChendaDamping(0.80, 8.8e-6),
    "3 months": ChendaDamping(0.81, 1.2e-5),
    "6 months": ChendaDamping(0.83, 1.8e-5),
    "1 year":   ChendaDamping(0.86, 2.8e-5),
    "Worn":     ChendaDamping(0.92, 4.5e-5),
}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fft_peak_f1(signal, fs, f_lo=80.0, f_hi=300.0):
    """Highest-amplitude FFT peak in [f_lo, f_hi]."""
    N   = len(signal)
    mag = np.abs(rfft(signal))
    freq= rfftfreq(N, 1.0/fs)
    mask = (freq >= f_lo) & (freq <= f_hi)
    if mask.sum() == 0:
        return float(freq[np.argmax(mag)])
    idx = np.argmax(mag[mask])
    return float(freq[mask][idx])


def measure_t60_bandpass(signal, fs, f_centre, bw_frac=0.18):
    """
    Measure T60 from log-envelope of signal bandpassed around f_centre.
    Uses same noise-floor gating as PGSD v3 (BUG4b fix).
    """
    f_nyq = fs / 2.0
    f_lo  = f_centre * (1 - bw_frac)
    f_hi  = f_centre * (1 + bw_frac)
    f_hi  = min(f_hi, f_nyq * 0.97)
    f_lo  = max(f_lo, 20.0)
    if f_lo >= f_hi:
        f_lo, f_hi = max(20, f_centre*0.7), min(f_nyq*0.97, f_centre*1.3)
    b, a = butter(4, [f_lo/f_nyq, f_hi/f_nyq], btype='band')
    sig_bp = filtfilt(b, a, signal)
    env    = np.abs(sig_bp)
    sm     = max(1, int(fs * 0.005))
    env_sm = np.convolve(env, np.ones(sm)/sm, mode='same')
    t_arr  = np.arange(len(signal)) / fs
    env_peak    = env_sm.max()
    noise_floor = env_peak / 31.6
    above = np.where(env_sm > noise_floor * 2.0)[0]
    if len(above) > 100:
        i0 = max(int(fs * 0.02), above[0])
        i1 = above[-1]
    else:
        i0, i1 = int(fs * 0.02), int(fs * 0.8)
    t_fit = t_arr[i0:i1]
    e_fit = np.log(np.maximum(env_sm[i0:i1], 1e-12))
    finite = np.isfinite(e_fit)
    if finite.sum() > 20:
        coeffs     = np.polyfit(t_fit[finite], e_fit[finite], 1)
        alpha_meas = max(-coeffs[0], 1e-3)
    else:
        alpha_meas = 1.0
    return np.log(1000.0) / alpha_meas


def nmf_decompose(signal, fs, K=18, n_fft=1024, hop=256, n_iter=200):
    """
    Run NMF on STFT magnitude. Returns (component_centroids, W, H, freqs).
    W: (n_freq_bins, K) spectral templates
    H: (K, n_frames) activations
    """
    _, _, Zxx = stft(signal, fs=fs, nperseg=n_fft, noverlap=n_fft-hop)
    V = np.abs(Zxx) + 1e-10                     # (freq_bins, frames)
    freq_bins = np.linspace(0, fs/2, V.shape[0])
    # Multiplicative update NMF
    np.random.seed(123)
    W = np.random.rand(V.shape[0], K) + 0.01
    H = np.random.rand(K, V.shape[1]) + 0.01
    for _ in range(n_iter):
        WH  = W @ H + 1e-10
        H  *= (W.T @ (V / WH)) / (W.T @ np.ones_like(V) + 1e-10)
        WH  = W @ H + 1e-10
        W  *= ((V / WH) @ H.T) / (np.ones_like(V) @ H.T + 1e-10)
        # Normalise columns of W
        col_norms = np.sqrt((W**2).sum(axis=0)) + 1e-10
        W /= col_norms
        H *= col_norms[:, None]
    # Centroid of each component
    centroids = []
    for k in range(K):
        w_k = W[:, k]
        c   = float(np.sum(freq_bins * w_k) / (np.sum(w_k) + 1e-10))
        centroids.append(c)
    return np.array(centroids), W, H, freq_bins


def nmf_t60(H, fs, hop=256, k_idx=0):
    """Estimate T60 from exponential decay of component activation."""
    act    = H[k_idx, :]
    t_arr  = np.arange(len(act)) * hop / fs
    noise  = act.max() / 31.6
    above  = np.where(act > noise * 2)[0]
    if len(above) < 10:
        return 1.0
    i0, i1 = above[0], above[-1]
    if i1 <= i0 + 5:
        return 1.0
    e_fit  = np.log(np.maximum(act[i0:i1], 1e-12))
    t_fit  = t_arr[i0:i1]
    finite = np.isfinite(e_fit)
    if finite.sum() < 5:
        return 1.0
    coeffs = np.polyfit(t_fit[finite], e_fit[finite], 1)
    alpha  = max(-coeffs[0], 1e-3)
    return np.log(1000.0) / alpha


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: TENSION ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def run_task1_baselines():
    print("\n" + "="*68)
    print("  TASK 1: TENSION ESTIMATION — Baseline comparison")
    print("="*68)
    results = {}
    for shape in SHAPES_LIST:
        dur = 1.5
        t   = np.linspace(0, dur, int(FS*dur), endpoint=False)
        np.random.seed(42)
        freqs_ref, modes_ref, alphas_ref, r, th, _ = get_spectral_basis(shape, T=T_REF)
        f1_ref = freqs_ref[0]
        T60_ref = ChendaDamping().T60(2*np.pi*f1_ref)

        shape_res = {}
        print(f"\n  {shape}")
        print(f"  {'T_true':>7}  {'FFT-PT err%':>12}  {'NMF err%':>10}")
        print(f"  {'─'*36}")

        for T_true in T_TENSIONS:
            freqs_t, modes_t, alphas_t, r, th, _ = get_spectral_basis(shape, T=T_true)
            signal, _ = synthesise_strike(freqs_t, modes_t, alphas_t, r, th,
                                          r_s=0.4*R_PHYS, theta_s=0.0, t=t)

            # ── FFT-PT ──────────────────────────────────────────────────────
            f1_fft = fft_peak_f1(signal, FS, f_lo=60.0, f_hi=350.0)
            T_fft  = T_REF * (f1_fft / f1_ref)**2
            err_fft = abs(T_fft - T_true) / T_true * 100

            # ── NMF ─────────────────────────────────────────────────────────
            centroids, W, H, _ = nmf_decompose(signal, FS, K=18)
            # Sort components by centroid; pick first in 60–350 Hz
            sorted_idx = np.argsort(centroids)
            f1_nmf = None
            for ki in sorted_idx:
                if 60 < centroids[ki] < 350:
                    f1_nmf = centroids[ki]
                    break
            if f1_nmf is None:
                f1_nmf = centroids[sorted_idx[0]]
            T_nmf  = T_REF * (f1_nmf / f1_ref)**2
            err_nmf = abs(T_nmf - T_true) / T_true * 100

            shape_res[T_true] = dict(fft_err=err_fft, nmf_err=err_nmf,
                                     f1_fft=f1_fft, f1_nmf=f1_nmf,
                                     f1_true=freqs_t[0])
            print(f"  {T_true:>7}  {err_fft:>12.2f}  {err_nmf:>10.2f}")

        mean_fft = np.mean([v['fft_err'] for v in shape_res.values()])
        mean_nmf = np.mean([v['nmf_err'] for v in shape_res.values()])
        print(f"  {'Mean':>7}  {mean_fft:>12.2f}  {mean_nmf:>10.2f}")
        results[shape] = dict(per_T=shape_res, mean_fft=mean_fft, mean_nmf=mean_nmf)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: STRIKE POSITION  (baselines return N/A — documented)
# ─────────────────────────────────────────────────────────────────────────────

def run_task2_baselines():
    """
    FFT-PT and NMF have no mode-shape model — they cannot recover strike
    position in physical units. We confirm this empirically by attempting
    a spectral-feature heuristic (ratio of harmonic amplitudes) and showing
    it degrades to near-random performance.
    """
    print("\n" + "="*68)
    print("  TASK 2: STRIKE POSITION — Baseline comparison")
    print("="*68)
    results = {}
    for shape in SHAPES_LIST:
        dur = 1.5
        t   = np.linspace(0, dur, int(FS*dur), endpoint=False)
        np.random.seed(77)
        freqs_ref, modes_ref, alphas_ref, r, th, _ = get_spectral_basis(shape)

        errs_fft = []
        errs_nmf = []
        print(f"\n  {shape}")
        print(f"  {'r_true/R':>10}  {'FFT heuristic err%R':>20}  {'NMF heuristic err%R':>20}")
        print(f"  {'─'*56}")

        for r_frac in R_POSITIONS:
            signal, _ = synthesise_strike(
                freqs_ref, modes_ref, alphas_ref, r, th,
                r_s=r_frac*R_PHYS, theta_s=0.0, t=t, snr_db=25.0)

            # FFT-PT heuristic: ratio of 2nd to 1st harmonic peak amplitude
            # as a proxy for position (higher order modes more excited near edge)
            mag  = np.abs(rfft(signal))
            freq = rfftfreq(len(signal), 1.0/FS)
            f1_e = fft_peak_f1(signal, FS)

            def peak_around(fc, bw=15):
                m = (freq >= fc-bw) & (freq <= fc+bw)
                return float(mag[m].max()) if m.any() else 0.0

            amp1 = peak_around(f1_e)
            amp2 = peak_around(f1_e * 1.59)  # approx ratio of Bessel zeros
            ratio_fft = amp2 / (amp1 + 1e-10)
            # Map ratio [0,1] linearly to r/R [0,1] — best-case linear mapping
            r_est_fft = np.clip(ratio_fft * 1.2, 0, 0.97)
            err_fft   = abs(r_est_fft - r_frac)

            # NMF heuristic: ratio of energy in high vs low components
            centroids_n, W, H, _ = nmf_decompose(signal, FS, K=18, n_iter=100)
            sorted_idx = np.argsort(centroids_n)
            lo_energy  = H[sorted_idx[:3], :].sum()
            hi_energy  = H[sorted_idx[3:], :].sum()
            ratio_nmf  = hi_energy / (lo_energy + hi_energy + 1e-10)
            r_est_nmf  = np.clip(ratio_nmf * 1.5, 0, 0.97)
            err_nmf    = abs(r_est_nmf - r_frac)

            errs_fft.append(err_fft)
            errs_nmf.append(err_nmf)
            print(f"  {r_frac:>10.2f}  {err_fft*100:>20.1f}  {err_nmf*100:>20.1f}")

        mean_fft = float(np.mean(errs_fft))
        mean_nmf = float(np.mean(errs_nmf))
        print(f"  {'Mean':>10}  {mean_fft*100:>20.1f}  {mean_nmf*100:>20.1f}")
        results[shape] = dict(mean_fft=mean_fft, mean_nmf=mean_nmf,
                              errs_fft=errs_fft, errs_nmf=errs_nmf)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: SKIN CONDITION
# ─────────────────────────────────────────────────────────────────────────────

def run_task3_baselines():
    print("\n" + "="*68)
    print("  TASK 3: SKIN CONDITION — Baseline comparison")
    print("="*68)
    results = {}
    for shape in SHAPES_LIST:
        dur = 2.0
        t   = np.linspace(0, dur, int(FS*dur), endpoint=False)
        np.random.seed(55)
        freqs_ref, modes_ref, alphas_ref, r, th, _ = get_spectral_basis(
            shape, damping=ChendaDamping())
        f1_ref  = freqs_ref[0]
        T60_ref = ChendaDamping().T60(2*np.pi*f1_ref)

        print(f"\n  {shape}   f₁={f1_ref:.1f} Hz   T60_ref={T60_ref:.3f} s")
        print(f"  {'Skin':>12}  {'CI_true':>8}  {'CI_FFT':>8}  {'CI_NMF':>8}")
        print(f"  {'─'*46}")

        shape_res = {}
        ci_true_list, ci_fft_list, ci_nmf_list = [], [], []

        for skin_name, damp in SKINS.items():
            al_s = np.array([damp.decay_rate(2*np.pi*f) for f in freqs_ref])
            signal, _ = synthesise_strike(freqs_ref, modes_ref, al_s, r, th,
                                          r_s=0.3*R_PHYS, theta_s=0.0, t=t)

            # True CI (from PGSD v3 formula, ground truth)
            T60_true = damp.T60(2*np.pi*f1_ref)
            CI_true  = min(100, max(0, (T60_ref - T60_true)/T60_ref*100))

            # ── FFT-PT: f̂₁ from FFT, then fixed-width bandpass T60 ─────────
            f1_fft = fft_peak_f1(signal, FS, f_lo=60.0, f_hi=350.0)
            T60_fft = measure_t60_bandpass(signal, FS, f1_fft, bw_frac=0.18)
            CI_fft  = min(100, max(0, (T60_ref - T60_fft)/T60_ref*100))

            # ── NMF: lowest-centroid component activation decay ───────────
            centroids_n, W, H, _ = nmf_decompose(signal, FS, K=18, n_iter=150)
            sorted_idx = np.argsort(centroids_n)
            # Find component closest to f1_ref in 60–350 Hz
            k_best = sorted_idx[0]
            for ki in sorted_idx:
                if 60 < centroids_n[ki] < 350:
                    k_best = ki
                    break
            T60_nmf = nmf_t60(H, FS, hop=256, k_idx=k_best)
            CI_nmf  = min(100, max(0, (T60_ref - T60_nmf)/T60_ref*100))

            ci_true_list.append(CI_true)
            ci_fft_list.append(CI_fft)
            ci_nmf_list.append(CI_nmf)
            shape_res[skin_name] = dict(CI_true=CI_true, CI_fft=CI_fft, CI_nmf=CI_nmf)
            print(f"  {skin_name:>12}  {CI_true:>8.1f}  {CI_fft:>8.1f}  {CI_nmf:>8.1f}")

        mae_fft = float(np.mean([abs(shape_res[s]['CI_fft'] - shape_res[s]['CI_true'])
                                  for s in SKINS]))
        mae_nmf = float(np.mean([abs(shape_res[s]['CI_nmf'] - shape_res[s]['CI_true'])
                                  for s in SKINS]))
        mono_fft = all(ci_fft_list[i] <= ci_fft_list[i+1] for i in range(len(ci_fft_list)-1))
        mono_nmf = all(ci_nmf_list[i] <= ci_nmf_list[i+1] for i in range(len(ci_nmf_list)-1))
        print(f"\n  FFT-PT  MAE={mae_fft:.1f}  Monotone={mono_fft}")
        print(f"  NMF     MAE={mae_nmf:.1f}  Monotone={mono_nmf}")
        results[shape] = dict(per_skin=shape_res, mae_fft=mae_fft, mae_nmf=mae_nmf,
                               mono_fft=mono_fft, mono_nmf=mono_nmf)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: SHAPE IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_task4_baselines():
    """
    FFT-PT: no shape model → chance (33%).
    NMF: nearest-neighbour on component centroid vector.
    We run 5 trials per true shape (different random seeds) for robustness.
    """
    print("\n" + "="*68)
    print("  TASK 4: SHAPE IDENTIFICATION — Baseline comparison")
    print("="*68)
    dur  = 1.5
    TRIALS = 5

    # Pre-compute NMF centroid fingerprints for each shape (reference signals)
    print("\n  Building NMF reference fingerprints...")
    ref_centroids = {}
    for shape in SHAPES_LIST:
        t = np.linspace(0, dur, int(FS*dur), endpoint=False)
        np.random.seed(200)
        freqs_r, modes_r, alphas_r, r, th, _ = get_spectral_basis(shape)
        signal_r, _ = synthesise_strike(freqs_r, modes_r, alphas_r, r, th,
                                        r_s=0.4*R_PHYS, theta_s=0.0, t=t)
        cents, _, _, _ = nmf_decompose(signal_r, FS, K=18, n_iter=200)
        ref_centroids[shape] = np.sort(cents)
        f1 = fft_peak_f1(signal_r, FS, 60, 350)
        print(f"    {shape}: f1_fft={f1:.1f} Hz  NMF_centroid[0]={np.sort(cents)[0]:.1f} Hz")

    results = {}
    print(f"\n  {'True shape':>12}  {'FFT-PT acc':>12}  {'NMF acc':>10}")
    print(f"  {'─'*40}")

    # Also record f1 per shape for FFT-PT shape classification
    # FFT-PT shape id: Hourglass has f1≈201 Hz, Cyl≈128 Hz, Oval≈119 Hz
    # Strategy: use measured f1 to assign shape via nearest reference f1
    ref_f1 = {}
    for shape in SHAPES_LIST:
        fr, _, _, _, _, _ = get_spectral_basis(shape, T=T_REF)
        ref_f1[shape] = fr[0]

    for true_shape in SHAPES_LIST:
        freqs_t, modes_t, alphas_t, r, th, _ = get_spectral_basis(true_shape)
        correct_fft = 0
        correct_nmf = 0

        for trial in range(TRIALS):
            t = np.linspace(0, dur, int(FS*dur), endpoint=False)
            np.random.seed(300 + trial)
            signal, _ = synthesise_strike(freqs_t, modes_t, alphas_t, r, th,
                                          r_s=0.4*R_PHYS, theta_s=0.0, t=t)

            # FFT-PT: nearest reference f1
            f1_meas = fft_peak_f1(signal, FS, 60.0, 350.0)
            best_fft = min(ref_f1, key=lambda s: abs(ref_f1[s] - f1_meas))
            if best_fft == true_shape:
                correct_fft += 1

            # NMF: nearest-neighbour on sorted centroid vector (L2)
            cents, _, _, _ = nmf_decompose(signal, FS, K=18, n_iter=150)
            cents_s = np.sort(cents)
            best_nmf = min(ref_centroids,
                           key=lambda s: float(norm(ref_centroids[s] - cents_s)))
            if best_nmf == true_shape:
                correct_nmf += 1

        acc_fft = correct_fft / TRIALS * 100
        acc_nmf = correct_nmf / TRIALS * 100
        results[true_shape] = dict(acc_fft=acc_fft, acc_nmf=acc_nmf)
        print(f"  {true_shape:>12}  {acc_fft:>12.0f}%  {acc_nmf:>9.0f}%")

    overall_fft = np.mean([v['acc_fft'] for v in results.values()])
    overall_nmf = np.mean([v['acc_nmf'] for v in results.values()])
    print(f"\n  Overall  FFT-PT={overall_fft:.0f}%  NMF={overall_nmf:.0f}%")
    results['_overall'] = dict(acc_fft=overall_fft, acc_nmf=overall_nmf)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "█"*68)
    print("  BASELINE EXPERIMENT — FFT-PT vs NMF vs PGSD v3")
    print("█"*68)

    r1 = run_task1_baselines()
    r2 = run_task2_baselines()
    r3 = run_task3_baselines()
    r4 = run_task4_baselines()

    print("\n\n" + "█"*68)
    print("  SUMMARY TABLE (values to go into paper)")
    print("█"*68)

    print("\n  Task 1 — Tension estimation mean error (%)")
    print(f"  {'Shape':>12}  {'FFT-PT':>8}  {'NMF':>8}")
    overall_fft, overall_nmf = [], []
    for s in SHAPES_LIST:
        f = r1[s]['mean_fft']
        n = r1[s]['mean_nmf']
        overall_fft.append(f)
        overall_nmf.append(n)
        print(f"  {s:>12}  {f:>8.2f}  {n:>8.2f}")
    print(f"  {'Mean':>12}  {np.mean(overall_fft):>8.2f}  {np.mean(overall_nmf):>8.2f}")

    print("\n  Task 2 — Strike position mean error (%R)")
    for s in SHAPES_LIST:
        print(f"  {s:>12}  FFT-PT:{r2[s]['mean_fft']*100:.1f}%R  NMF:{r2[s]['mean_nmf']*100:.1f}%R")

    print("\n  Task 3 — Skin condition CI MAE")
    for s in SHAPES_LIST:
        print(f"  {s:>12}  FFT-PT MAE:{r3[s]['mae_fft']:.1f}  NMF MAE:{r3[s]['mae_nmf']:.1f}  "
              f"Mono FFT:{r3[s]['mono_fft']}  Mono NMF:{r3[s]['mono_nmf']}")

    print("\n  Task 4 — Shape identification accuracy (%)")
    for s in SHAPES_LIST:
        print(f"  {s:>12}  FFT-PT:{r4[s]['acc_fft']:.0f}%  NMF:{r4[s]['acc_nmf']:.0f}%")
    print(f"  {'Overall':>12}  FFT-PT:{r4['_overall']['acc_fft']:.0f}%  "
          f"NMF:{r4['_overall']['acc_nmf']:.0f}%")

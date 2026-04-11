"""
extended_baselines_kaggle.py
=============================
KAGGLE SETUP
------------
1. Upload these three files as a Kaggle dataset:
      chenda_pgsd_v3.py
      chenda_spectral.py
      extended_baselines_kaggle.py  (this file)

2. In your Kaggle notebook, the uploaded files will be at:
      /kaggle/input/<your-dataset-name>/

3. Edit the two paths below to match your dataset name, then run.

REQUIREMENTS (all pre-installed on Kaggle)
------------------------------------------
   numpy, scipy, scikit-learn
"""

import os

# ── EDIT THESE TO MATCH YOUR KAGGLE DATASET PATH ─────────────────────────────
# Example: if your dataset is called "chenda-pgsd"
# the files will be at /kaggle/input/chenda-pgsd/chenda_pgsd_v3.py etc.
PGSD_V3_PATH  = "/kaggle/input/chenda-pgsd/chenda_pgsd_v3.py"
SPECTRAL_PATH = "/kaggle/input/chenda-pgsd/chenda_spectral.py"
# ─────────────────────────────────────────────────────────────────────────────

import sys, json
import numpy as np
from scipy.signal import butter, filtfilt, stft
from scipy.linalg import svd, norm
from scipy.fft import rfft, rfftfreq
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Patch the hardcoded spectral path inside pgsd_v3 before loading ──────────
pgsd_src = open(PGSD_V3_PATH).read()
pgsd_src = pgsd_src.replace(
    'exec(open("/mnt/user-data/outputs/chenda_spectral.py")',
    f'exec(open("{SPECTRAL_PATH}")'
)
exec(compile(pgsd_src[:pgsd_src.rfind("\nif __name__")], "pgsd_v3", "exec"), globals())
print("PGSD v3 loaded OK")

# ─────────────────────────────────────────────────────────────────────────────
FS          = 22050
T_REF       = 3500.0
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

# =============================================================================
# SHARED UTILITIES
# =============================================================================

def measure_t60(signal, fs, f_centre, bw_frac=0.18):
    f_nyq = fs / 2.0
    f_lo  = max(20.0, f_centre * (1 - bw_frac))
    f_hi  = min(f_nyq * 0.97, f_centre * (1 + bw_frac))
    b, a  = butter(4, [f_lo/f_nyq, f_hi/f_nyq], btype='band')
    env   = np.abs(filtfilt(b, a, signal))
    sm    = max(1, int(fs * 0.005))
    env_s = np.convolve(env, np.ones(sm)/sm, mode='same')
    t_arr = np.arange(len(signal)) / fs
    noise = env_s.max() / 31.6
    above = np.where(env_s > noise * 2)[0]
    i0 = max(int(fs*0.02), above[0])  if len(above) > 100 else int(fs*0.02)
    i1 = above[-1]                     if len(above) > 100 else int(fs*0.8)
    t_f, e_f = t_arr[i0:i1], np.log(np.maximum(env_s[i0:i1], 1e-12))
    fin = np.isfinite(e_f)
    if fin.sum() > 20:
        alpha = max(-np.polyfit(t_f[fin], e_f[fin], 1)[0], 1e-3)
    else:
        alpha = 1.0
    return np.log(1000.0) / alpha

# =============================================================================
# B1 — ESPRIT (modal estimation method)
# =============================================================================

def esprit_freqs(signal, fs, n_modes=36, win=4096):
    x  = signal[:win].astype(complex)
    L  = n_modes * 2 + 4
    M  = len(x) - L + 1
    X  = np.array([x[i:i+M] for i in range(L)])
    U, _, _ = svd(X, full_matrices=False)
    Es  = U[:, :n_modes]
    Phi = np.linalg.lstsq(Es[:-1], Es[1:], rcond=None)[0]
    eigs = np.linalg.eigvals(Phi)
    freqs_hz = np.angle(eigs) * fs / (2 * np.pi)
    return np.sort(freqs_hz[freqs_hz > 0])

def esprit_f1(signal, fs, f_lo=60.0, f_hi=350.0):
    fq   = esprit_freqs(signal, fs)
    band = fq[(fq >= f_lo) & (fq <= f_hi)]
    if len(band) > 0: return float(band[0])
    pos = fq[fq > 20]
    return float(pos[0]) if len(pos) > 0 else 100.0

# =============================================================================
# B2 — MFCC + MLP (data-driven pipeline)
# =============================================================================

def hz_to_mel(hz):  return 2595.0 * np.log10(1.0 + hz / 700.0)
def mel_to_hz(mel): return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def compute_mfcc(signal, fs, n_mfcc=13, n_fft=1024, hop=512,
                 n_mels=40, f_min=60.0, f_max=8000.0, win_dur=0.1):
    sig  = signal[:int(fs * win_dur)]
    if len(sig) < n_fft:
        sig = np.pad(sig, (0, n_fft - len(sig)))
    frames = [np.abs(rfft(sig[s:s+n_fft] * np.hanning(n_fft)))**2
              for s in range(0, len(sig)-n_fft+1, hop)]
    if not frames: return np.zeros(n_mfcc)
    spec = np.array(frames)
    freq = rfftfreq(n_fft, 1.0/fs)
    pts  = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(min(f_max, fs/2)), n_mels+2))
    fbank = np.zeros((n_mels, len(freq)))
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m+1], pts[m+2]
        for k, f in enumerate(freq):
            if lo <= f <= c:   fbank[m,k] = (f-lo)/(c-lo+1e-10)
            elif c < f <= hi:  fbank[m,k] = (hi-f)/(hi-c+1e-10)
    log_mel = np.log(spec @ fbank.T + 1e-10)
    n_idx   = np.arange(1, n_mfcc+1)
    dct     = np.cos(np.pi * n_idx[:,None] * (2*np.arange(n_mels)[None,:]+1) / (2*n_mels))
    return (dct @ log_mel.T).T.mean(axis=0)

def build_training_set(task, shape=None, n=500):
    np.random.seed(1234)
    dur = 1.5
    t   = np.linspace(0, dur, int(FS*dur), endpoint=False)
    X, y = [], []

    if task == 'tension':
        for _ in range(n):
            T_r = np.random.uniform(2800, 4300)
            fr, mo, al, r, th, _ = get_spectral_basis(shape, T=T_r)
            sig, _ = synthesise_strike(fr, mo, al, r, th,
                r_s=np.random.uniform(0.1,0.9)*R_PHYS,
                theta_s=np.random.uniform(0,2*np.pi), t=t, snr_db=25)
            X.append(compute_mfcc(sig, FS)); y.append(T_r)

    elif task == 'position':
        fr, mo, al, r, th, _ = get_spectral_basis(shape)
        for _ in range(n):
            r_r = np.random.uniform(0.0, 0.95)
            sig, _ = synthesise_strike(fr, mo, al, r, th,
                r_s=r_r*R_PHYS, theta_s=np.random.uniform(0,2*np.pi),
                t=t, snr_db=25)
            X.append(compute_mfcc(sig, FS)); y.append(r_r)

    elif task == 'skin':
        fr, mo, _, r, th, _ = get_spectral_basis(shape)
        for eta2 in np.logspace(np.log10(8.8e-6), np.log10(4.5e-5), n):
            d   = ChendaDamping(eta1=0.80, eta2=eta2)
            al  = np.array([d.decay_rate(2*np.pi*f) for f in fr])
            sig, _ = synthesise_strike(fr, mo, al, r, th,
                r_s=0.3*R_PHYS, theta_s=0.0, t=t, snr_db=25)
            X.append(compute_mfcc(sig, FS, win_dur=0.5))
            y.append(d.T60(2*np.pi*fr[0]))

    elif task == 'shape':
        for i, s in enumerate(SHAPES_LIST):
            for _ in range(n//3):
                T_r = np.random.uniform(3000, 4100)
                fr, mo, al, r, th, _ = get_spectral_basis(s, T=T_r)
                sig, _ = synthesise_strike(fr, mo, al, r, th,
                    r_s=np.random.uniform(0.1,0.9)*R_PHYS,
                    theta_s=np.random.uniform(0,2*np.pi), t=t, snr_db=25)
                X.append(compute_mfcc(sig, FS)); y.append(i)

    return np.array(X), np.array(y)

def train_mlp(X, y, hidden=(64,32), classifier=False):
    sc  = StandardScaler()
    Xsc = sc.fit_transform(X)
    if classifier:
        m = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500,
                          random_state=0, early_stopping=True)
    else:
        m = MLPRegressor(hidden_layer_sizes=hidden, max_iter=500,
                         random_state=0, early_stopping=True)
    m.fit(Xsc, y.astype(int) if classifier else y)
    return m, sc

# =============================================================================
# B3 — HPS (Harmonic Product Spectrum)
# =============================================================================

def hps_f0(signal, fs, n_harmonics=5, f_lo=60.0, f_hi=350.0):
    N    = min(len(signal), 16384)
    mag  = np.abs(rfft(signal[:N] * np.hanning(N)))
    freq = rfftfreq(N, 1.0/fs)
    hps  = mag.copy()
    for h in range(2, n_harmonics+1):
        ds  = mag[::h]
        hps = hps[:len(ds)] * ds
    freq_h = freq[:len(hps)]
    mask   = (freq_h >= f_lo) & (freq_h <= f_hi)
    return float(freq_h[mask][np.argmax(hps[mask])]) if mask.any() \
           else float(freq[np.argmax(mag)])

# =============================================================================
# EXISTING BASELINES (FFT-PT and NMF)
# =============================================================================

def fft_peak_f1(signal, fs, f_lo=60.0, f_hi=350.0):
    mag  = np.abs(rfft(signal))
    freq = rfftfreq(len(signal), 1.0/fs)
    mask = (freq >= f_lo) & (freq <= f_hi)
    return float(freq[mask][np.argmax(mag[mask])]) if mask.any() \
           else float(freq[np.argmax(mag)])

def nmf_decompose(signal, fs, K=18, n_fft=1024, hop=256, n_iter=200):
    _, _, Zxx = stft(signal, fs=fs, nperseg=n_fft, noverlap=n_fft-hop)
    V    = np.abs(Zxx) + 1e-10
    freq = np.linspace(0, fs/2, V.shape[0])
    np.random.seed(123)
    W = np.random.rand(V.shape[0], K) + 0.01
    H = np.random.rand(K, V.shape[1]) + 0.01
    for _ in range(n_iter):
        WH = W @ H + 1e-10
        H *= (W.T @ (V/WH)) / (W.T @ np.ones_like(V) + 1e-10)
        WH = W @ H + 1e-10
        W *= ((V/WH) @ H.T) / (np.ones_like(V) @ H.T + 1e-10)
        n  = np.sqrt((W**2).sum(0)) + 1e-10
        W /= n; H *= n[:,None]
    cents = [float(np.sum(freq * W[:,k]) / (np.sum(W[:,k])+1e-10)) for k in range(K)]
    return np.array(cents), W, H

def nmf_f1(signal, fs, f_lo=60, f_hi=350):
    cents, _, _ = nmf_decompose(signal, fs)
    band = sorted([c for c in cents if f_lo < c < f_hi])
    return float(band[0]) if band else 150.0

def nmf_t60(H, fs, hop=256, k_idx=0):
    act   = H[k_idx,:]
    t_arr = np.arange(len(act)) * hop / fs
    noise = act.max() / 31.6
    above = np.where(act > noise*2)[0]
    if len(above) < 10: return 1.0
    i0, i1 = above[0], above[-1]
    if i1 <= i0+5: return 1.0
    e   = np.log(np.maximum(act[i0:i1], 1e-12))
    t   = t_arr[i0:i1]
    fin = np.isfinite(e)
    if fin.sum() < 5: return 1.0
    return np.log(1000.0) / max(-np.polyfit(t[fin], e[fin], 1)[0], 1e-3)

# =============================================================================
# TASK 1 — TENSION ESTIMATION
# =============================================================================

def task1():
    print("\n" + "="*70 + "\n  TASK 1: TENSION ESTIMATION\n" + "="*70)
    results = {}
    for shape in SHAPES_LIST:
        dur = 1.5
        t   = np.linspace(0, dur, int(FS*dur), endpoint=False)
        np.random.seed(42)
        freqs_ref, modes_ref, alphas_ref, r, th, _ = get_spectral_basis(shape, T=T_REF)
        f1_ref = freqs_ref[0]

        print(f"\n  {shape}  (f1_ref={f1_ref:.1f} Hz)")
        print(f"  Training MFCC+MLP...", end='', flush=True)
        Xtr, ytr = build_training_set('tension', shape, n=500)
        mlp, sc  = train_mlp(Xtr, ytr)
        print(" done")

        errs = {m: [] for m in ['esprit','mfcc_mlp','hps','fft_pt','nmf']}
        print(f"  {'T':>6}  {'ESPRIT':>8}  {'MLP':>8}  {'HPS':>8}  {'FFT-PT':>8}  {'NMF':>8}")
        print("  " + "─"*58)

        for T_true in T_TENSIONS:
            fr_t, mo_t, al_t, _, _, _ = get_spectral_basis(shape, T=T_true)
            sig, _ = synthesise_strike(fr_t, mo_t, al_t, r, th,
                                       r_s=0.4*R_PHYS, theta_s=0.0, t=t)
            def pct(f): return abs(T_REF*(f/f1_ref)**2 - T_true)/T_true*100

            e_esp = pct(esprit_f1(sig, FS))
            e_hps = pct(hps_f0(sig, FS))
            e_fft = pct(fft_peak_f1(sig, FS))
            e_nmf = pct(nmf_f1(sig, FS))
            T_mlp = float(mlp.predict(sc.transform(compute_mfcc(sig,FS).reshape(1,-1)))[0])
            e_mlp = abs(T_mlp - T_true)/T_true*100

            for k,v in zip(errs, [e_esp,e_mlp,e_hps,e_fft,e_nmf]):
                errs[k].append(v)
            print(f"  {T_true:>6}  {e_esp:>8.2f}  {e_mlp:>8.2f}  {e_hps:>8.2f}"
                  f"  {e_fft:>8.2f}  {e_nmf:>8.2f}")

        means = {k: float(np.mean(v)) for k,v in errs.items()}
        print(f"  {'Mean':>6}  {means['esprit']:>8.2f}  {means['mfcc_mlp']:>8.2f}"
              f"  {means['hps']:>8.2f}  {means['fft_pt']:>8.2f}  {means['nmf']:>8.2f}")
        results[shape] = means
    return results

# =============================================================================
# TASK 2 — STRIKE POSITION
# =============================================================================

def task2():
    print("\n" + "="*70 + "\n  TASK 2: STRIKE POSITION\n" + "="*70)
    results = {}
    for shape in SHAPES_LIST:
        dur = 1.5
        t   = np.linspace(0, dur, int(FS*dur), endpoint=False)
        np.random.seed(77)
        freqs_ref, modes_ref, alphas_ref, r, th, _ = get_spectral_basis(shape)

        print(f"\n  {shape}")
        print(f"  Training MFCC+MLP...", end='', flush=True)
        Xtr, ytr = build_training_set('position', shape, n=500)
        mlp_p, sc_p = train_mlp(Xtr, ytr)
        print(" done")

        errs = {m: [] for m in ['esprit','mfcc_mlp','hps','fft_pt','nmf']}
        print(f"  {'r/R':>6}  {'ESPRIT':>8}  {'MLP':>8}  {'HPS':>8}  {'FFT-PT':>8}  {'NMF':>8}")
        print("  " + "─"*58)

        for r_frac in R_POSITIONS:
            np.random.seed(77)
            sig, _ = synthesise_strike(freqs_ref, modes_ref, alphas_ref, r, th,
                                       r_s=r_frac*R_PHYS, theta_s=0.0, t=t, snr_db=25.0)

            # MFCC+MLP: direct regression
            r_mlp = float(np.clip(mlp_p.predict(
                sc_p.transform(compute_mfcc(sig,FS).reshape(1,-1)))[0], 0, 0.97))
            e_mlp = abs(r_mlp - r_frac) * 100

            # ESPRIT: harmonic ratio heuristic (no mode shapes)
            fq   = esprit_freqs(sig, FS, n_modes=36)
            band = fq[(fq >= 60) & (fq <= 500)]
            r_esp = float(np.clip((band[1]/(band[0]+1e-5)-1.5)/0.6, 0, 0.97)) \
                    if len(band) >= 2 else 0.5
            e_esp = abs(r_esp - r_frac) * 100

            # HPS heuristic
            mag  = np.abs(rfft(sig[:16384]))
            freq = rfftfreq(16384, 1.0/FS)
            f1h  = hps_f0(sig, FS)
            a1   = float(mag[np.argmin(np.abs(freq-f1h))])
            i2   = np.argmin(np.abs(freq-f1h*1.59))
            a2   = float(mag[i2]) if i2 < len(mag) else 0.0
            r_hps = float(np.clip((a2/(a1+1e-10))*1.2, 0, 0.97))
            e_hps = abs(r_hps - r_frac) * 100

            # FFT-PT heuristic
            mag2  = np.abs(rfft(sig))
            freq2 = rfftfreq(len(sig), 1.0/FS)
            f1f   = fft_peak_f1(sig, FS)
            a1f   = float(mag2[np.argmin(np.abs(freq2-f1f))])
            i2f   = np.argmin(np.abs(freq2-f1f*1.59))
            a2f   = float(mag2[i2f]) if i2f < len(mag2) else 0.0
            r_fft = float(np.clip((a2f/(a1f+1e-10))*1.2, 0, 0.97))
            e_fft = abs(r_fft - r_frac) * 100

            # NMF heuristic
            cents_n, _, Hn = nmf_decompose(sig, FS, n_iter=100)
            sidx  = np.argsort(cents_n)
            lo_e  = Hn[sidx[:3],:].sum()
            hi_e  = Hn[sidx[3:],:].sum()
            r_nmf = float(np.clip(hi_e/(lo_e+hi_e+1e-10)*1.5, 0, 0.97))
            e_nmf = abs(r_nmf - r_frac) * 100

            for k,v in zip(errs, [e_esp,e_mlp,e_hps,e_fft,e_nmf]):
                errs[k].append(v)
            print(f"  {r_frac:>6.2f}  {e_esp:>8.1f}  {e_mlp:>8.1f}  {e_hps:>8.1f}"
                  f"  {e_fft:>8.1f}  {e_nmf:>8.1f}")

        means = {k: float(np.mean(v)) for k,v in errs.items()}
        print(f"  {'Mean':>6}  {means['esprit']:>8.1f}  {means['mfcc_mlp']:>8.1f}"
              f"  {means['hps']:>8.1f}  {means['fft_pt']:>8.1f}  {means['nmf']:>8.1f}")
        results[shape] = means
    return results

# =============================================================================
# TASK 3 — SKIN CONDITION
# =============================================================================

def task3():
    print("\n" + "="*70 + "\n  TASK 3: SKIN CONDITION\n" + "="*70)
    results = {}
    for shape in SHAPES_LIST:
        dur = 2.0
        t   = np.linspace(0, dur, int(FS*dur), endpoint=False)
        np.random.seed(55)
        freqs_ref, modes_ref, _, r, th, _ = get_spectral_basis(
            shape, damping=ChendaDamping())
        f1_ref  = freqs_ref[0]
        T60_ref = ChendaDamping().T60(2*np.pi*f1_ref)

        print(f"\n  {shape}  f1={f1_ref:.1f} Hz  T60_ref={T60_ref:.3f} s")
        print(f"  Training MFCC+MLP...", end='', flush=True)
        Xtr, ytr = build_training_set('skin', shape, n=300)
        mlp_sk, sc_sk = train_mlp(Xtr, ytr, hidden=(32,16))
        print(" done")

        ci_lists = {m: [] for m in ['esprit','mfcc_mlp','hps','fft_pt','nmf','true']}
        print(f"  {'Skin':>12}  {'True':>6}  {'ESPRIT':>8}  {'MLP':>8}"
              f"  {'HPS':>8}  {'FFT-PT':>8}  {'NMF':>8}")
        print("  " + "─"*72)

        for skin_name, damp in SKINS.items():
            al_s   = np.array([damp.decay_rate(2*np.pi*f) for f in freqs_ref])
            sig, _ = synthesise_strike(freqs_ref, modes_ref, al_s, r, th,
                                       r_s=0.3*R_PHYS, theta_s=0.0, t=t)

            T60_gt = damp.T60(2*np.pi*f1_ref)
            def ci(t60): return min(100, max(0, (T60_ref-t60)/T60_ref*100))

            CI_gt  = ci(T60_gt)
            CI_esp = ci(measure_t60(sig, FS, esprit_f1(sig, FS)))
            CI_hps = ci(measure_t60(sig, FS, hps_f0(sig, FS)))
            CI_fft = ci(measure_t60(sig, FS, fft_peak_f1(sig, FS)))

            cents_n, _, Hn = nmf_decompose(sig, FS, n_iter=150)
            sidx   = np.argsort(cents_n)
            k_best = next((k for k in sidx if 60 < cents_n[k] < 350), sidx[0])
            CI_nmf = ci(nmf_t60(Hn, FS, k_idx=k_best))

            T60_mlp = float(np.clip(mlp_sk.predict(
                sc_sk.transform(compute_mfcc(sig,FS,win_dur=0.5).reshape(1,-1)))[0], 0.05, 5.0))
            CI_mlp = ci(T60_mlp)

            for k,v in zip(ci_lists,
                           [CI_esp,CI_mlp,CI_hps,CI_fft,CI_nmf,CI_gt]):
                ci_lists[k].append(v)

            print(f"  {skin_name:>12}  {CI_gt:>6.1f}  {CI_esp:>8.1f}  {CI_mlp:>8.1f}"
                  f"  {CI_hps:>8.1f}  {CI_fft:>8.1f}  {CI_nmf:>8.1f}")

        def mae(m): return float(np.mean(
            [abs(p-q) for p,q in zip(ci_lists[m], ci_lists['true'])]))
        def mono(m): lst=ci_lists[m]; return all(lst[i]<=lst[i+1] for i in range(len(lst)-1))

        print(f"\n  MAE   ESPRIT:{mae('esprit'):.1f}  MLP:{mae('mfcc_mlp'):.1f}"
              f"  HPS:{mae('hps'):.1f}  FFT-PT:{mae('fft_pt'):.1f}  NMF:{mae('nmf'):.1f}")
        print(f"  Mono  ESPRIT:{mono('esprit')}  MLP:{mono('mfcc_mlp')}"
              f"  HPS:{mono('hps')}  FFT-PT:{mono('fft_pt')}  NMF:{mono('nmf')}")

        results[shape] = {m: {'mae': mae(m), 'monotone': mono(m)}
                          for m in ['esprit','mfcc_mlp','hps','fft_pt','nmf']}
    return results

# =============================================================================
# TASK 4 — SHAPE IDENTIFICATION
# =============================================================================

def task4():
    print("\n" + "="*70 + "\n  TASK 4: SHAPE IDENTIFICATION  (with tension variation)\n" + "="*70)
    dur    = 1.5
    TRIALS = 5

    print("\n  Training MFCC+MLP classifier...", end='', flush=True)
    Xtr, ytr = build_training_set('shape', n=600)
    clf, sc_sh = train_mlp(Xtr, ytr, hidden=(64,32), classifier=True)
    print(" done")

    # Reference fingerprints for ESPRIT and frequency-based methods
    ref_esp = {}
    ref_f1  = {}
    print("  Building reference fingerprints...", end='', flush=True)
    for shape in SHAPES_LIST:
        t = np.linspace(0, dur, int(FS*dur), endpoint=False)
        np.random.seed(200)
        fr, mo, al, r, th, _ = get_spectral_basis(shape)
        sig, _ = synthesise_strike(fr, mo, al, r, th, r_s=0.4*R_PHYS, theta_s=0.0, t=t)
        fp = esprit_freqs(sig, FS, n_modes=36)
        ref_esp[shape] = np.sort(fp[fp > 60])[:10]
        ref_f1[shape]  = fr[0]
    print(" done")

    results = {}
    print(f"\n  {'Shape':>12}  {'ESPRIT':>8}  {'MLP':>8}  {'HPS':>8}  {'FFT-PT':>8}  {'NMF':>8}")
    print("  " + "─"*62)

    for true_shape in SHAPES_LIST:
        counts = {m: 0 for m in ['esprit','mfcc_mlp','hps','fft_pt','nmf']}

        for trial in range(TRIALS):
            T_test = T_TENSIONS[trial % len(T_TENSIONS)]
            fr, mo, al, r, th, _ = get_spectral_basis(true_shape, T=T_test)
            t = np.linspace(0, dur, int(FS*dur), endpoint=False)
            np.random.seed(300 + trial)
            sig, _ = synthesise_strike(fr, mo, al, r, th, r_s=0.4*R_PHYS, theta_s=0.0, t=t)

            # ESPRIT
            fp  = np.sort(esprit_freqs(sig, FS, n_modes=36))
            fp  = fp[fp > 60][:10]
            best_e = min(ref_esp, key=lambda s:
                         norm(ref_esp[s][:len(fp)] - fp[:len(ref_esp[s])]))
            if best_e == true_shape: counts['esprit'] += 1

            # MFCC+MLP
            p = int(clf.predict(sc_sh.transform(compute_mfcc(sig,FS).reshape(1,-1)))[0])
            if SHAPES_LIST[np.clip(p,0,2)] == true_shape: counts['mfcc_mlp'] += 1

            # HPS
            f1h = hps_f0(sig, FS)
            if min(ref_f1, key=lambda s: abs(ref_f1[s]-f1h)) == true_shape:
                counts['hps'] += 1

            # FFT-PT
            f1f = fft_peak_f1(sig, FS)
            if min(ref_f1, key=lambda s: abs(ref_f1[s]-f1f)) == true_shape:
                counts['fft_pt'] += 1

            # NMF
            cents_n, _, _ = nmf_decompose(sig, FS, n_iter=100)
            nmf_f1v = float(sorted([c for c in cents_n if c > 60])[0]) \
                      if any(c > 60 for c in cents_n) else 150.0
            if min(ref_f1, key=lambda s: abs(ref_f1[s]-nmf_f1v)) == true_shape:
                counts['nmf'] += 1

        accs = {m: counts[m]/TRIALS*100 for m in counts}
        results[true_shape] = accs
        print(f"  {true_shape:>12}  {accs['esprit']:>8.0f}%  {accs['mfcc_mlp']:>7.0f}%"
              f"  {accs['hps']:>8.0f}%  {accs['fft_pt']:>7.0f}%  {accs['nmf']:>7.0f}%")

    ov = {m: float(np.mean([results[s][m] for s in SHAPES_LIST]))
          for m in ['esprit','mfcc_mlp','hps','fft_pt','nmf']}
    results['overall'] = ov
    print(f"  {'Overall':>12}  {ov['esprit']:>8.0f}%  {ov['mfcc_mlp']:>7.0f}%"
          f"  {ov['hps']:>8.0f}%  {ov['fft_pt']:>7.0f}%  {ov['nmf']:>7.0f}%")
    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "█"*70)
    print("  EXTENDED BASELINES: ESPRIT | MFCC+MLP | HPS | FFT-PT | NMF")
    print("█"*70)

    r1 = task1()
    r2 = task2()
    r3 = task3()
    r4 = task4()

    methods = ['esprit', 'mfcc_mlp', 'hps', 'fft_pt', 'nmf']

    t1_mean = {m: float(np.mean([r1[s][m] for s in SHAPES_LIST])) for m in methods}
    t2_mean = {m: float(np.mean([r2[s][m] for s in SHAPES_LIST])) for m in methods}
    t3_mae  = {m: float(np.mean([r3[s][m]['mae'] for s in SHAPES_LIST])) for m in methods}
    t4_acc  = r4['overall']

    pgsd = {'T1': 2.62, 'T1_oval': 0.32, 'T2': 0.3, 'T3': 0.0, 'T4': 100.0}

    print("\n\n" + "█"*70)
    print("  SUMMARY — paste the JSON below back to Claude")
    print("█"*70)

    final = {
        'T1_tension_mean_pct'  : {**t1_mean, 'PGSD': pgsd['T1'], 'PGSD_Oval': pgsd['T1_oval']},
        'T2_position_mean_pctR': {**t2_mean, 'PGSD': pgsd['T2']},
        'T3_skin_mae'          : {**t3_mae,  'PGSD': pgsd['T3']},
        'T4_shape_acc_pct'     : {**t4_acc,  'PGSD': pgsd['T4']},
        'raw_T1': {s: r1[s] for s in SHAPES_LIST},
        'raw_T2': {s: r2[s] for s in SHAPES_LIST},
        'raw_T3': {s: r3[s] for s in SHAPES_LIST},
        'raw_T4': r4,
    }

    print("\n" + "─"*70)
    print(json.dumps(final, indent=2))
    print("─"*70)
    print("\nDone. Copy the JSON above and paste it back to Claude.")

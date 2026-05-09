# Physics-Guided Spectral Decomposition for Acoustic Analysis of Chenda
## ISMIR 2026 — Code Repository

Official code for the paper:  
**"Physics-Guided Spectral Decomposition for Acoustic Analysis of Chenda and Shell Geometry Variants"**

---

## Repository Layout

```
chenda-pgsd-ismir2026/
├── src/
│   ├── chenda_spectral.py          Fourier-Chebyshev eigensolver + density models
│   ├── chenda_pgsd_v3.py           PGSD v3 framework (calibrated to real recordings)
│   └── generate_eigenmodes.py      Reproduce Figures 3-5 (eigenmode plots)
│
├── notebooks/
│   └── chenda_pgsd_analysis.ipynb  Full analysis: PGSD + FFT + Sinusoidal + NMF
│
├── paper/
│   ├── main.tex                    LaTeX source
│   ├── mybib.bib                   Bibliography
│   ├── section_real_recordings.tex Section 5 (real recordings)
│   └── chenda_pgsd_ISMIR2026.pdf   Compiled PDF
│
├── figures/
│   ├── eigenmodes_cylinder.png     Figure 3
│   ├── eigenmodes_hourglass.png    Figure 4
│   └── eigenmodes_oval.png         Figure 5
│
└── audio/
    ├── synthesised/                PGSD-synthesised Chenda strikes
    ├── real_recordings/            Place your beats dataset here (beat_1.wav … beat_50.wav)
    └── reconstructed/
        ├── beats/                  Original beats (reference)
        ├── pgsd/                   PGSD reconstructions  (beat_N_pgsd_recon.wav)
        ├── fft/                    FFT-PT reconstructions (beat_N_fft_recon.wav)
        ├── sinusoidal/             Sinusoidal reconstructions (beat_N_sinusoidal_recon.wav)
        └── nmf/                    NMF reconstructions  (beat_N_nmf_recon.wav)
```

---

## Dataset

**50 real Chenda beats** (`beat_1.wav` … `beat_50.wav`)  
Kaggle dataset: `kevinbenty/beats-chenda`

Place the files in `audio/real_recordings/` before running the notebook locally,  
or keep the Kaggle paths unchanged for Kaggle execution.

---

## Reconstruction Results (50 beats, mean R²)

| Method | Mean R² | Notes |
|--------|---------|-------|
| **PGSD (ours)** | **0.701** | Physics model, 30 modes, T=4284 N/m |
| Sinusoidal Model | 0.594 | STFT peak tracking + iSTFT |
| FFT Peak Tracking | 0.499 | Top-30 spectral peaks retained |
| NMF | −0.210 | Griffin-Lim phase reconstruction artefacts |

PGSD additionally recovers physical parameters inaccessible to all baselines:
tension (T̂ ≈ 3235–7591 N/m across beats), strike position, and shell shape.

---

## Quick Start — Local

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/chenda-pgsd-ismir2026.git
cd chenda-pgsd-ismir2026

# 2. Install
conda create -n chenda python=3.10
conda activate chenda
pip install numpy scipy matplotlib scikit-learn soundfile librosa jupyter

# 3. Reproduce eigenmode figures (Figures 3-5)
cd src
python generate_eigenmodes.py

# 4. Run full analysis notebook
cd ..
jupyter notebook notebooks/chenda_pgsd_analysis.ipynb
```

---

## Quick Start — Kaggle

1. Add dataset `kevinbenty/beats-chenda` (the 50 recordings)
2. Add dataset `kevinbenty/privatechenda` (the two `.py` model files)
3. Open `notebooks/chenda_pgsd_analysis.ipynb` — no path changes needed
4. Run All

---

## Five Implementation Bugs Fixed

All five cause R² < 0.01 in otherwise correct physics models:

| # | Bug | Fix |
|---|-----|-----|
| 1 | Chebyshev grid descends; interpolator requires ascending | Flip `r_cheb[::-1]` |
| 2 | Grid excludes r = 0; fill_value = 0 zeroes centre strikes | Linear extrapolation to r = 0 |
| 3 | Only cos(mθ) degenerate member excited | Signal-level template matching for position |
| 4 | Bandpass hardcoded at reference f₁ regardless of shape | Shape-specific f₁ ± 18% |
| 5 | Fixed λ = 1e-6 over-regularises when frequencies mismatch | λ = λ_rel · ‖ΦᵀΦ‖_F |

---

## Physical Model Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| T_REAL | 4284 N/m | Mean f₁ = 222.6 Hz across real recordings |
| η₁ | 1.087 | hand_small T60 = 0.755 s at f₁ = 222 Hz |
| η₂ | 8.8 × 10⁻⁶ | Sathej & Adhikari (2008) |
| N modes | 30 | Analysis |
| R | 0.125 m | Membrane radius |
| σ_ref | 1.4 kg/m² | Reference surface density |

---

## Citation

```bibtex
@inproceedings{chenda_pgsd_ismir2026,
  title     = {Physics-Guided Spectral Decomposition for Acoustic Analysis
               of {Chenda} and Shell Geometry Variants},
  booktitle = {Proc.\ 27th Int.\ Society for Music Information Retrieval
               Conf.\ (ISMIR)},
  year      = {2026},
  note      = {Under review --- anonymous submission}
}
```

---

## Acknowledgements

Physical model follows Sathej & Adhikari (2008) arXiv:0809.1320.  
Recordings captured with assistance from performers at Kalamandalam, Kerala.

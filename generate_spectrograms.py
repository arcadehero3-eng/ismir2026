"""
src/generate_spectrograms.py
=============================
Generates mel-spectrograms for all synthesised audio files
and saves them to figures/spectrograms/.

Usage:
    python src/generate_spectrograms.py

Output:
    figures/spectrograms/<shape>_<condition>.png
    One PNG per WAV file in audio/synthesised/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram as scipy_spectrogram

AUDIO_DIR  = "audio/synthesised"
OUT_DIR    = "figures/spectrograms"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
NPERSEG   = 512
NOVERLAP  = 448
FMAX_PLOT = 1200    # Hz — only plot up to this frequency
CMAP      = "magma"


def make_spectrogram(wav_path, out_path, title=None):
    fs, data = wavfile.read(wav_path)

    # Convert to float
    if data.dtype == np.int16:
        sig = data.astype(float) / 32768.0
    elif data.dtype == np.int32:
        sig = data.astype(float) / 2147483648.0
    else:
        sig = data.astype(float)

    # Mono
    if sig.ndim == 2:
        sig = sig.mean(axis=1)

    # Compute spectrogram
    f, t, Sxx = scipy_spectrogram(
        sig, fs=fs,
        nperseg=NPERSEG, noverlap=NOVERLAP,
        window="hann", scaling="spectrum")

    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    # Trim to FMAX_PLOT
    fmask  = f <= FMAX_PLOT
    f_plot = f[fmask]
    S_plot = Sxx_db[fmask, :]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3), facecolor="white")
    ax.set_facecolor("white")

    pcm = ax.pcolormesh(t, f_plot, S_plot,
                        shading="gouraud", cmap=CMAP,
                        vmin=S_plot.max() - 60,
                        vmax=S_plot.max())

    cbar = fig.colorbar(pcm, ax=ax, pad=0.01)
    cbar.set_label("Power (dB)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.tick_params(labelsize=9)

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def friendly_title(filename):
    """Convert v3_cylinder_T3500.wav → Cylinder  T = 3500 N/m"""
    name = os.path.splitext(filename)[0]           # v3_cylinder_T3500
    parts = name.replace("v3_", "").split("_")     # ['cylinder', 'T3500']
    shape = parts[0].capitalize()
    rest  = " ".join(parts[1:])
    rest  = rest.replace("T", "T = ").replace("pos", "pos = ")
    rest  = rest.replace("skin", "skin: ")
    return f"{shape}  {rest}"


# ── Batch process all WAVs ────────────────────────────────────────────────────
total = 0
for shape_dir in ["cylinder", "hourglass", "oval"]:
    dir_path = os.path.join(AUDIO_DIR, shape_dir)
    if not os.path.isdir(dir_path):
        continue
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".wav"):
            continue
        wav_path = os.path.join(dir_path, fname)
        out_name = fname.replace(".wav", "_spectrogram.png")
        out_path = os.path.join(OUT_DIR, out_name)
        title    = friendly_title(fname)
        make_spectrogram(wav_path, out_path, title=title)
        print(f"  {fname:45s} → {out_name}")
        total += 1

print(f"\nGenerated {total} spectrograms → {OUT_DIR}/")


# ── Bonus: 3×5 comparison grid for tension series ────────────────────────────
def make_tension_grid():
    """One figure showing all 5 tensions for all 3 shapes side by side."""
    tensions  = [3000, 3200, 3500, 3800, 4100]
    shapes    = ["cylinder", "hourglass", "oval"]
    colors    = {"cylinder": "#1D9E75", "hourglass": "#378ADD", "oval": "#E24B4A"}

    fig, axes = plt.subplots(3, 5, figsize=(15, 7), facecolor="white")
    fig.suptitle("Chenda Spectrograms — Tension Variation (3000–4100 N/m)",
                 fontsize=12, fontweight="bold")

    for r, shape in enumerate(shapes):
        for c, T in enumerate(tensions):
            ax = axes[r, c]
            wav = os.path.join(AUDIO_DIR, shape, f"v3_{shape}_T{T}.wav")
            if not os.path.exists(wav):
                ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes)
                continue
            fs, data = wavfile.read(wav)
            sig = data.astype(float) / 32768.0
            f, t, Sxx = scipy_spectrogram(sig, fs=fs,
                nperseg=NPERSEG, noverlap=NOVERLAP, window="hann")
            Sxx_db = 10*np.log10(Sxx+1e-12)
            fmask = f <= FMAX_PLOT
            ax.pcolormesh(t, f[fmask], Sxx_db[fmask],
                          shading="gouraud", cmap=CMAP,
                          vmin=Sxx_db[fmask].max()-55,
                          vmax=Sxx_db[fmask].max())
            ax.set_facecolor("white")
            if r == 0:
                ax.set_title(f"T={T}", fontsize=9)
            if c == 0:
                ax.set_ylabel(shape.capitalize(), fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_yticks([200, 600, 1000])

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "tension_comparison_grid.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nTension comparison grid → {out}")


make_tension_grid()

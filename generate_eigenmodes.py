"""
generate_eigenmodes.py
======================
Produces three publication-quality figures (one per shell geometry):
  - spectral_modes_cylinder.png
  - spectral_modes_hourglass.png
  - spectral_modes_oval.png

Layout: 10 rows × 2 columns = 20 eigenmodes per figure
Colormap: RdBu_r  (red = +displacement, blue = -displacement, white = 0)
Background: white throughout
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import os, sys

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load spectral solver ──────────────────────────────────────────────────────
src = open("/mnt/user-data/outputs/chenda_spectral.py").read()
exec(compile(src[:src.rfind("\nif __name__")], "spectral", "exec"), globals())

# ── Parameters ────────────────────────────────────────────────────────────────
N_MODES   = 20          # 10 rows × 2 columns
NR, NTH   = 25, 24      # solver grid
T_REF     = 3500.0
R_PHYS    = 0.125
SIGMA_MEM = 1.4

# Polar plot grid (fine, for interpolation)
N_POLAR_R  = 80
N_POLAR_TH = 180

# ── Helper: polar → Cartesian interpolated mode ──────────────────────────────

def interpolate_mode(psi_raw, r_cheb, th_cheb):
    """
    psi_raw : (Nr, Nth)  on Chebyshev × Fourier grid (r descending)
    Returns Z : (N_POLAR_R, N_POLAR_TH)  on fine uniform polar grid
    """
    from scipy.interpolate import RegularGridInterpolator

    # Flip r to ascending
    r_asc   = r_cheb[::-1].copy()
    psi_asc = psi_raw[::-1, :].copy()

    # Extend to r = 0 via linear extrapolation
    r0, r1   = r_asc[0], r_asc[1]
    v0, v1   = psi_asc[0, :], psi_asc[1, :]
    v_centre = v0 - r0 * (v1 - v0) / (r1 - r0)
    r_full   = np.concatenate([[0.0], r_asc])
    psi_full = np.vstack([v_centre[None, :], psi_asc])

    # Wrap theta for periodic interpolation
    th_ext   = np.append(th_cheb, th_cheb[0] + 2 * np.pi)
    psi_ext  = np.hstack([psi_full, psi_full[:, :1]])

    interp = RegularGridInterpolator(
        (r_full, th_ext), psi_ext,
        method="linear", bounds_error=False, fill_value=0.0)

    r_fine  = np.linspace(0.0, 1.0,        N_POLAR_R)
    th_fine = np.linspace(0.0, 2 * np.pi,  N_POLAR_TH, endpoint=False)
    RR, TH  = np.meshgrid(r_fine, th_fine, indexing="ij")

    pts = np.column_stack([RR.ravel(), TH.ravel()])
    Z   = interp(pts).reshape(N_POLAR_R, N_POLAR_TH)
    return Z, r_fine, th_fine


def polar_to_xy(Z, r_fine, th_fine):
    """Convert polar Z array to Cartesian X, Y, Z for pcolormesh."""
    TH, RR = np.meshgrid(th_fine, r_fine)
    X = RR * np.cos(TH)
    Y = RR * np.sin(TH)
    return X, Y, Z


# ── Main plot function ────────────────────────────────────────────────────────

def plot_eigenmodes(shape_name, save_path):
    print(f"\nSolving eigenmodes for {shape_name} ...")
    cfg = SHAPES[shape_name]
    freqs, modes, r_cheb, th, rho = solve_eigenmodes(
        Nr=NR, Nth=NTH,
        density_fn=cfg["density_fn"],
        density_kw=cfg["density_kw"],
        n_modes=N_MODES + 4,          # solve a few extra
        R_phys=R_PHYS, T=T_REF,
        sigma_mem=SIGMA_MEM)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # 10 rows × 2 cols of mode plots
    # + 1 extra row at top for the title / colour legend

    ROWS, COLS = 10, 2
    fig_w = 5.0          # inches wide (ISMIR column width ~3.5in; double = 7in)
    ax_size = fig_w / COLS
    fig_h = ax_size * ROWS + 1.0   # +1 for header

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # GridSpec: row 0 = header, rows 1..ROWS = mode panels
    gs = gridspec.GridSpec(
        ROWS + 1, COLS,
        figure=fig,
        hspace=0.10,
        wspace=0.06,
        top=0.97, bottom=0.01,
        left=0.01, right=0.99,
        height_ratios=[0.55] + [1.0] * ROWS,
    )

    # ── Header row ────────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor("white")
    ax_hdr.axis("off")

    # Main title
    ax_hdr.text(
        0.5, 0.82,
        f"Eigenmodes of {shape_name} Shell",
        ha="center", va="top",
        fontsize=11, fontweight="bold",
        transform=ax_hdr.transAxes,
        color="black",
    )

    # Colour legend (small gradient bar)
    cbar_ax = ax_hdr.inset_axes([0.20, 0.05, 0.60, 0.32])
    cb_data = np.linspace(-1, 1, 256).reshape(1, -1)
    cbar_ax.imshow(cb_data, aspect="auto", cmap="RdBu_r",
                   extent=[-1, 1, 0, 1])
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks([-1, 0, 1])
    cbar_ax.set_xticklabels(
        ["−max\n(inward)", "0\n(nodal)", "+max\n(outward)"],
        fontsize=6.5, color="black")
    cbar_ax.tick_params(axis="x", length=2, pad=1)
    for spine in cbar_ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.5)
    cbar_ax.set_title("Displacement amplitude", fontsize=7,
                       color="black", pad=3)

    # ── Mode panels ───────────────────────────────────────────────────────────
    for idx in range(N_MODES):
        row = idx // COLS + 1    # +1 because row 0 is header
        col = idx  % COLS

        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("white")
        ax.set_aspect("equal")

        psi_raw = modes[idx]
        Z, r_fine, th_fine = interpolate_mode(psi_raw, r_cheb, th)
        X, Y, Zxy = polar_to_xy(Z, r_fine, th_fine)

        # Symmetric colour limits
        vmax = np.abs(Zxy).max()
        if vmax < 1e-10:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        # Filled polar plot
        ax.pcolormesh(X, Y, Zxy,
                      cmap="RdBu_r", norm=norm,
                      shading="gouraud", rasterized=True)

        # Membrane boundary circle
        theta_ring = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta_ring), np.sin(theta_ring),
                color="black", linewidth=0.7, zorder=5)

        # Nodal lines (zero contour)
        TH_c, RR_c = np.meshgrid(th_fine, r_fine)
        Xc = RR_c * np.cos(TH_c)
        Yc = RR_c * np.sin(TH_c)
        ax.contour(Xc, Yc, Zxy,
                   levels=[0.0],
                   colors=["#333333"],
                   linewidths=[0.5],
                   zorder=6)

        # Frequency label
        freq_hz = freqs[idx]
        ax.text(0.03, 0.97,
                f"$f_{{{idx+1}}}$",
                ha="left", va="top",
                fontsize=6.5, fontweight="bold",
                transform=ax.transAxes, color="black",
                zorder=7)
        ax.text(0.97, 0.97,
                f"{freq_hz:.0f} Hz",
                ha="right", va="top",
                fontsize=6, transform=ax.transAxes,
                color="#333333", zorder=7)

        # Clean up axes
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        print(f"  Mode {idx+1:2d}: {freq_hz:.1f} Hz", end="")
        if (idx + 1) % 5 == 0:
            print()

    print()
    fig.savefig(save_path, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  → Saved: {save_path}")


# ── Generate all three shapes ─────────────────────────────────────────────────

for shape in ["Cylinder", "Hourglass", "Oval"]:
    out = os.path.join(OUTPUT_DIR, f"spectral_modes_{shape.lower()}.png")
    plot_eigenmodes(shape, out)

print("\nAll three figures generated.")

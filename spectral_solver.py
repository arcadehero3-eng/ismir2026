"""
chenda_spectral.py
==================
Full 2D spectral eigenmode solver for non-uniform circular membranes.

Theory sources:
  [1] Sathej & Adhikari (2008), arXiv:0809.1320
      "The eigenspectra of Indian musical drums"
      → Fourier-Chebyshev spectral collocation
      → Non-uniform density ρ(r,θ) via smooth tanh loading
      → Generalised eigenvalue problem  L Ψ = -λ² B Ψ

  [2] Howle & Trefethen (2001), J. Comp. Appl. Math. 135:23-40
      "Eigenvalues and musical instruments"
      → Eigenvalues in complex plane (freq = Im λ, decay = -Re λ)
      → Air loading shifts m=0 modes only
      → Ideal membrane: η = AJm(z)cos(mθ), zeros of Jm

Physical model
--------------
Circular membrane of unit radius (normalised), non-uniform areal density ρ(r,θ),
uniform tension T, clamped boundary u(r=1)=0.

Equation of motion (Sathej eq. 3):
    ρ ü = T ∇² u

Eigenvalue problem (Sathej eq. 5):
    -ω² ρ(r,θ) Ψ(r,θ) = T ∇² Ψ(r,θ)

Normalise by setting T/ρ_ref = 1 → eigenvalue λ = ω/ω_ref

In matrix form (Sathej eq. 10):
    L · Ψ = -λ² B · Ψ

where L is the Fourier-Chebyshev Laplacian matrix and B = diag(ρ).

Fourier-Chebyshev Laplacian (Sathej eq. 7):
    L = (D1 + RE1)⊗Il + (D2 + RE2)⊗Ir + R²⊗Dθ²

where:
  D1, D2  = Chebyshev matrices for ∂²r
  E1, E2  = Chebyshev matrices for r⁻¹∂r
  Dθ²     = Fourier matrix for r⁻²∂²θ
  R       = diag(r⁻¹)
  Il, Ir  = block identity matrices (Fornberg prescription)

Density profiles
----------------
Three shell shapes change the CAVITY (not membrane) - but following Sathej,
we model the drum head itself as non-uniform. For the Chenda:

  Uniform (baseline) : σ=1, no loading → pure Bessel modes
  Cylinder-loaded    : central loading concentric (like dayan)
  Hourglass-loaded   : eccentric loading displaces density centroid
  Oval-loaded        : ring loading at intermediate radius

Shape enters via the density function ρ(r,θ;shape_params).
"""

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "/mnt/user-data/outputs"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CHEBYSHEV DIFFERENTIATION MATRICES  (Trefethen, Spectral Methods in MATLAB)
# ══════════════════════════════════════════════════════════════════════════════

def cheb(N):
    """
    Chebyshev spectral differentiation matrix of size (N+1)×(N+1).
    Returns (D, x) where x are the Chebyshev nodes on [-1,1].
    Trefethen (2000), Program 6.
    """
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])
    x = np.cos(np.pi * np.arange(N+1) / N)
    c = np.ones(N+1)
    c[0] = 2.0; c[N] = 2.0
    c *= (-1)**np.arange(N+1)
    X = np.tile(x, (N+1, 1))
    dX = X - X.T
    D = np.outer(c, 1.0/c) / (dX + np.eye(N+1))
    D -= np.diag(D.sum(axis=1))
    return D, x


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FOURIER DIFFERENTIATION MATRIX  (periodic, Trefethen Program 1)
# ══════════════════════════════════════════════════════════════════════════════

def fourier_diff2(Nth):
    """
    Second-derivative Fourier matrix for Nth equally-spaced points on [0,2π).
    Returns Dth2 of size Nth × Nth.
    """
    h = 2 * np.pi / Nth
    col = np.zeros(Nth)
    col[0] = -np.pi**2 / (3 * h**2) - 1.0/6.0
    k = np.arange(1, Nth)
    col[1:] = -0.5 * (-1)**k / np.sin(h * k / 2)**2
    Dth2 = la.toeplitz(col)
    return Dth2


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FOURIER-CHEBYSHEV LAPLACIAN  (Sathej eq. 7, Fornberg prescription)
#
#  The Cholesky expansion on [0,1] uses the symmetry extension trick:
#  map r ∈ [0,1] to s ∈ [-1,1] via  r = (s+1)/2  — BUT Fornberg's
#  prescription for the DISK uses the full [-1,1] range with only the
#  "right half" (s>0, i.e. r>0) being the physical domain, and enforces
#  parity conditions automatically.
#
#  Here we follow the simpler and equivalent approach of Sathej:
#  Nr Chebyshev nodes on r∈[0,1] (half-interval), Nθ Fourier nodes.
#
#  Grid: r_j = cos(jπ/2Nr), j=1..Nr   (interior nodes, r=0 excluded)
#        θ_k = 2πk/Nθ,       k=0..Nθ-1
# ══════════════════════════════════════════════════════════════════════════════

def build_laplacian(Nr, Nth):
    """
    Build the Fourier-Chebyshev Laplacian matrix for the unit disk.

    Uses the Sathej (2008) formulation:
        ∇² = ∂²r + r⁻¹∂r + r⁻²∂²θ

    represented as:
        L = (D1 + RE1)⊗Il + (D2 + RE2)⊗Ir + R²⊗Dθ²

    Parameters
    ----------
    Nr  : number of Chebyshev radial nodes (odd, interior only)
    Nth : number of Fourier angular nodes  (even)

    Returns
    -------
    L    : Laplacian matrix  (Nr*Nth) × (Nr*Nth)
    r    : radial grid (Nr,)
    th   : angular grid (Nth,)
    """
    # ── Chebyshev matrices on [0,1] ────────────────────────────────────────
    # Use 2*Nr+1 point grid on [-1,1], take only interior right half
    D_full, x_full = cheb(2 * Nr)         # (2Nr+1) × (2Nr+1)

    # Interior nodes: indices 1..Nr (right half, corresponding to r in (0,1])
    # x_full = cos(jπ/2Nr), j=0..2Nr  → x=1 at j=0, x=0 at j=Nr, x=-1 at j=2Nr
    # Physical r = x (we use only right half j=1..Nr-1, i.e. r ∈ (0,1))
    idx = np.arange(1, Nr)               # Nr-1 interior nodes
    r_nodes = x_full[idx]                # r values in (0,1)

    # Build ∂r and ∂²r on interior (boundary stripped)
    # Full first-deriv matrix
    Dr_full = D_full
    Dr2_full = Dr_full @ Dr_full

    # Map: on the full grid ∂/∂x = ∂/∂r (since r=x)
    # Keep only interior-to-interior block
    D1r = Dr2_full[np.ix_(idx, idx)]     # ∂²r  (Nr-1)×(Nr-1)
    D1  = Dr_full[np.ix_(idx, idx)]      # ∂r   (Nr-1)×(Nr-1)

    Nr_int = len(r_nodes)
    r = r_nodes                          # actual r values

    # Diagonal R matrix: diag(1/r_j)
    Rmat = np.diag(1.0 / r)             # (Nr_int) × (Nr_int)
    Rmat2 = np.diag(1.0 / r**2)

    # ── Fourier second-derivative matrix ──────────────────────────────────
    th = np.linspace(0, 2*np.pi, Nth, endpoint=False)
    Dth2 = fourier_diff2(Nth)           # (Nth) × (Nth)

    # ── Block identity matrices (Sathej eqs. 8,9) ─────────────────────────
    # Il and Ir encode the Fornberg parity prescription.
    # For ∂²r term: Il = I_Nth (standard)
    # For r⁻¹∂r term: Ir = block-swap to couple r and π-r harmonics
    # Following Trefethen MATLAB code: use simple identity for both
    # (valid for the half-interval approach)
    Il = np.eye(Nth)
    Ir = np.eye(Nth)

    # ── Build full Laplacian via Kronecker products ────────────────────────
    # L = ∂²r ⊗ Il  +  R·(∂r) ⊗ Ir  +  R² ⊗ Dθ²/r²
    # i.e.  L_full[i*Nth+k, j*Nth+l] =
    #         D1r[i,j]*Il[k,l]  +  Rmat[i,i]*D1[i,j]*Ir[k,l]  +  Rmat2[i,i]*Dth2[k,l]*δ_ij

    size = Nr_int * Nth

    # Term 1: ∂²r ⊗ Il
    L = np.kron(D1r, Il)

    # Term 2: diag(1/r) · ∂r ⊗ Ir  =  (Rmat @ D1) ⊗ Ir
    L += np.kron(Rmat @ D1, Ir)

    # Term 3: diag(1/r²) ⊗ Dθ²  (block-diagonal: each radial node gets its Dθ²)
    for i in range(Nr_int):
        row0 = i * Nth
        col0 = i * Nth
        L[row0:row0+Nth, col0:col0+Nth] += Rmat2[i, i] * Dth2

    return L, r, th


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DENSITY PROFILES  (Sathej eqs. 1,2 + shape extensions)
#
#  Base Sathej density (concentric loading):
#    ρ(r,θ) = 1 + (σ²-1)/2 · [1 - tanh((R(r,θ)-k)/ξ)]
#    R(r,θ) = sqrt((r cosθ - ε)² + (r sinθ)²)
#
#  σ = density ratio (central/outer), k = radius ratio, ξ = smoothness, ε = eccentricity
#
#  Shape-specific adaptations for Chenda:
#    Cylinder  : concentric loading (ε=0)  → σ_opt=2.57, k=0.492, ξ=0.091
#    Hourglass : ring loading at r=k_ring  → loading concentrated in annulus
#    Oval      : eccentric loading (ε>0)   → breaks circular symmetry
# ══════════════════════════════════════════════════════════════════════════════

def density_sathej(r_grid, th_grid, sigma=2.57, k=0.492, xi=0.091, eps=0.0):
    """
    Smooth non-uniform density from Sathej & Adhikari eq. (1).

    ρ(r,θ) = 1 + (σ²-1)/2 · [1 - tanh((R(r,θ)-k)/ξ)]
    R(r,θ) = sqrt((r cosθ - ε)² + (r sinθ)²)

    Parameters
    ----------
    r_grid, th_grid : 2D meshgrid arrays (Nr_int × Nth)
    sigma  : sqrt of density ratio ρ_centre/ρ_rim  (σ=1 → uniform)
    k      : normalised loading radius (ratio of loaded to total area ~ k²)
    xi     : smoothness of transition
    eps    : eccentricity of loading centre (0=concentric)
    """
    R_field = np.sqrt((r_grid * np.cos(th_grid) - eps)**2 +
                      (r_grid * np.sin(th_grid))**2)
    rho = 1.0 + (sigma**2 - 1.0) / 2.0 * (1.0 - np.tanh((R_field - k) / xi))
    return rho


def density_hourglass(r_grid, th_grid, k_ring=0.55, width=0.08, sigma=2.5):
    """
    Ring loading: heavier annulus at r ≈ k_ring.
    Models the Chenda hourglass shape where the waist restricts internal
    cross-section, effectively concentrating coupling at intermediate r.
    """
    R_field = r_grid
    # Ring profile: bell-shaped loading centred at k_ring
    rho = 1.0 + (sigma - 1.0) * np.exp(-((R_field - k_ring) / width)**2)
    return rho


def density_oval(r_grid, th_grid, sigma=2.57, k=0.492, xi=0.091, eps=0.18):
    """
    Eccentric loading (ε>0): Sathej bayan model.
    Loading patch displaced from centre by distance ε.
    Breaks circular symmetry → lifts mode degeneracy.
    """
    return density_sathej(r_grid, th_grid, sigma=sigma, k=k, xi=xi, eps=eps)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GENERALISED EIGENVALUE SOLVER
#
#  Solve:  L · Ψ = -λ² B · Ψ
#  where:
#    L  = Laplacian matrix  (size × size)
#    B  = diag(ρ) density matrix  (size × size)
#    λ² = eigenvalues = (ω/ω_ref)²
#    Ψ  = eigenvectors (flattened mode shapes)
#
#  We solve as standard eigenvalue problem:
#    (-B⁻¹ L) · Ψ = λ² Ψ
#
#  Take only eigenvalues with positive real part (physical modes).
# ══════════════════════════════════════════════════════════════════════════════

def solve_eigenmodes(Nr=21, Nth=20, density_fn=None, density_kw=None,
                     n_modes=25, f_ref=None, R_phys=0.125, T=3500.0, sigma_mem=1.4):
    """
    Full spectral eigenmode solver.

    Parameters
    ----------
    Nr, Nth    : Chebyshev radial and Fourier angular resolution
    density_fn : callable(r_grid, th_grid, **density_kw) → ρ array
    density_kw : dict of keyword args to density_fn
    n_modes    : number of modes to return
    f_ref      : reference frequency to convert λ to Hz (if None, computed from T, sigma_mem, R_phys)
    R_phys     : physical membrane radius (m)
    T          : tension (N/m)
    sigma_mem  : surface density (kg/m²)

    Returns
    -------
    freqs      : array of modal frequencies (Hz), sorted ascending
    modes      : list of 2D mode shape arrays (Nr_int × Nth), one per mode
    r, th      : radial and angular grids
    rho_2d     : 2D density field on grid
    """
    if density_fn is None:
        density_fn = density_sathej
    if density_kw is None:
        density_kw = {}

    # ── Build Laplacian and grid ───────────────────────────────────────────
    L, r, th = build_laplacian(Nr, Nth)   # r in (0,1), th in [0,2π)
    Nr_int = len(r)
    size   = Nr_int * Nth

    # ── Build 2D density grid and flatten ─────────────────────────────────
    R2, TH2 = np.meshgrid(r, th, indexing='ij')   # (Nr_int × Nth)
    rho_2d  = density_fn(R2, TH2, **density_kw)   # (Nr_int × Nth)
    rho_flat = rho_2d.flatten()                    # length size

    # ── Mass matrix B = diag(ρ) ────────────────────────────────────────────
    B = np.diag(rho_flat)
    B_inv = np.diag(1.0 / rho_flat)

    # ── Solve eigenvalue problem  (-B⁻¹ L) v = λ² v ──────────────────────
    A = -B_inv @ L
    raw_vals, raw_vecs = la.eig(A)

    # Take real positive eigenvalues (physical modes)
    # λ² should be real positive; imaginary parts are numerical noise
    real_vals = np.real(raw_vals)
    good = real_vals > 0.1   # discard near-zero and negative (numerical artefacts)
    real_vals = real_vals[good]
    raw_vecs  = raw_vecs[:, good]

    # Sort ascending
    order     = np.argsort(real_vals)
    real_vals = real_vals[order]
    raw_vecs  = raw_vecs[:, order]

    # ── Convert eigenvalues to Hz ──────────────────────────────────────────
    # On unit disk, λ = ω_normalised.
    # Physical frequency: f = λ · (c_mem / (2π R_phys))
    # where c_mem = sqrt(T / sigma_mem) is the membrane wave speed.
    c_mem = np.sqrt(T / sigma_mem)
    f_scale = c_mem / (2 * np.pi * R_phys)

    # λ from eigenvalue:  λ² = ω_norm²  →  ω_norm = sqrt(λ²)  →  f = ω_norm/(2π) * f_scale
    # Actually: λ² are the eigenvalues of (-B⁻¹L) where L is in normalised coords.
    # Normalised ω = sqrt(eigenvalue), physical f = normalised_ω × c_mem/(2πR)
    omega_norm = np.sqrt(real_vals)
    freqs_hz   = omega_norm * f_scale / (2 * np.pi)
    # Correction: the Laplacian is already in r/R units, so f = sqrt(λ)*c/(2πR)
    freqs_hz   = np.sqrt(real_vals) * c_mem / (2 * np.pi * R_phys)

    # ── Extract mode shapes ────────────────────────────────────────────────
    modes = []
    for i in range(min(n_modes, raw_vecs.shape[1])):
        psi_flat = np.real(raw_vecs[:, i])
        psi_2d   = psi_flat.reshape(Nr_int, Nth)
        # Normalise to ±1
        mx = np.max(np.abs(psi_2d))
        if mx > 1e-12:
            psi_2d /= mx
        modes.append(psi_2d)

    return freqs_hz[:n_modes], modes[:n_modes], r, th, rho_2d


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SHAPE CONFIGURATIONS  (three Chenda shell shapes)
# ══════════════════════════════════════════════════════════════════════════════

SHAPES = {
    'Cylinder': {
        'density_fn': density_sathej,
        'density_kw': {'sigma': 2.57, 'k': 0.492, 'xi': 0.091, 'eps': 0.0},
        'color': '#5AB4E8',
        'desc': 'Concentric loading  σ=2.57  k=0.492  ξ=0.091  ε=0',
        'theory': 'Sathej dayan model (ε=0): radially symmetric, m=0 pairs degenerate'
    },
    'Hourglass': {
        'density_fn': density_hourglass,
        'density_kw': {'k_ring': 0.55, 'width': 0.08, 'sigma': 2.5},
        'color': '#E8A838',
        'desc': 'Ring loading at r=0.55  σ_ring=2.5  width=0.08',
        'theory': 'Ring loading: lifts nodal-circle mode frequencies, concentrates coupling'
    },
    'Oval': {
        'density_fn': density_oval,
        'density_kw': {'sigma': 2.57, 'k': 0.492, 'xi': 0.091, 'eps': 0.18},
        'color': '#5ECF82',
        'desc': 'Eccentric loading  σ=2.57  k=0.492  ξ=0.091  ε=0.18',
        'theory': 'Sathej bayan model (ε>0): breaks circular symmetry, lifts degeneracy'
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOT: 20 MODE SHAPES FOR ONE SHAPE (full 2D on polar grid)
# ══════════════════════════════════════════════════════════════════════════════

def plot_mode_grid(shape_name, freqs, modes, r, th, rho_2d, color):
    """
    Plot 20 mode shapes in a 4×5 grid on a filled polar disk,
    exactly like Sathej Fig. 8 / Howle Fig. 15 but with colour fill.
    """
    n_show = min(20, len(modes))
    cols, rows = 5, 4

    fig, axes = plt.subplots(rows, cols, figsize=(18, 15))
    fig.patch.set_facecolor('#0A0F1E')
    axes = axes.flatten()

    cfg = SHAPES[shape_name]
    fig.suptitle(
        f'{shape_name} Chenda — First {n_show} Eigenmodes\n'
        f'{cfg["desc"]}\n'
        f'Theory: {cfg["theory"]}',
        fontsize=11, color='white', fontweight='bold', y=0.99,
        va='top'
    )

    # Build full 2D polar display grid
    r_disp  = np.linspace(0, 1, 160)
    th_disp = np.linspace(0, 2*np.pi, 200)
    R_disp, TH_disp = np.meshgrid(r_disp, th_disp)
    X_disp = R_disp * np.cos(TH_disp)
    Y_disp = R_disp * np.sin(TH_disp)
    th_c   = np.linspace(0, 2*np.pi, 400)

    norm_col = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    for i in range(n_show):
        ax = axes[i]
        ax.set_facecolor('#0A0F1E')
        ax.set_xlim(-1.22, 1.22)
        ax.set_ylim(-1.22, 1.22)
        ax.set_aspect('equal')
        ax.axis('off')

        psi_2d = modes[i]   # shape: (Nr_int × Nth)

        # Interpolate from (r, th) grid to display grid using 2D interp
        # psi_2d[i_r, i_th] → interpolate onto r_disp, th_disp
        from scipy.interpolate import RegularGridInterpolator
        # th grid must be periodic: append first column
        th_ext  = np.append(th, th[0] + 2*np.pi)
        psi_ext = np.hstack([psi_2d, psi_2d[:, :1]])
        interp  = RegularGridInterpolator(
            (r, th_ext), psi_ext,
            method='linear', bounds_error=False, fill_value=0.0
        )
        pts = np.column_stack([R_disp.ravel(), TH_disp.ravel() % (2*np.pi)])
        Z_disp = interp(pts).reshape(R_disp.shape)

        # Mask outside unit disk
        Z_disp[R_disp > 1.0] = np.nan

        # Draw filled contours
        ax.contourf(X_disp, Y_disp, Z_disp, levels=64,
                    cmap='RdBu_r', norm=norm_col, extend='both')

        # Nodal lines
        ax.contour(X_disp, Y_disp, Z_disp, levels=[0],
                   colors='white', linewidths=0.9, alpha=0.8)

        # Boundary circle
        ax.plot(np.cos(th_c), np.sin(th_c), color='white', lw=1.6, alpha=0.9)

        # Mode number and frequency label
        f_col = color if i < 6 else '#CCCCCC'
        ax.set_title(
            f'#{i+1}   {freqs[i]:.1f} Hz',
            fontsize=9, color=f_col, fontweight='bold', pad=3
        )

    # Hide unused axes
    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)

    # Legend for colours
    fig.text(0.5, 0.005,
             'Red = +displacement   Blue = −displacement   '
             'White lines = nodal lines (zero)   '
             f'Computed via Fourier-Chebyshev spectral collocation  '
             f'[Sathej & Adhikari 2008]',
             ha='center', color='#9CA3AF', fontsize=8.5)

    plt.tight_layout(rect=[0, 0.018, 1, 0.96])
    fname = os.path.join(OUTPUT_DIR, f'spectral_modes_{shape_name.lower()}.png')
    plt.savefig(fname, dpi=130, bbox_inches='tight', facecolor='#0A0F1E')
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PLOT: EIGENVALUE SPECTRUM IN COMPLEX PLANE  (Howle & Trefethen style)
# ══════════════════════════════════════════════════════════════════════════════

def plot_complex_spectrum(all_results):
    """
    Plot eigenvalues in the complex plane (freq vs decay rate),
    following Howle & Trefethen Fig. 15-17 style.
    For undamped membrane: decay rate = 0, eigenvalues on imaginary axis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#0A0F1E')
    fig.suptitle(
        'Chenda Eigenvalue Spectra in the Complex Plane\n'
        'Style: Howle & Trefethen (2001) — Im(λ) = frequency, −Re(λ) = decay rate\n'
        'For undamped membrane: all eigenvalues on imaginary axis (zero decay)',
        fontsize=11, color='white', fontweight='bold', y=0.99
    )

    for ax, (shape_name, (freqs, modes, r, th, rho)) in zip(axes, all_results.items()):
        ax.set_facecolor('#111827')
        col = SHAPES[shape_name]['color']

        # All eigenvalues on imaginary axis (no damping model)
        decay = np.zeros_like(freqs)
        ax.scatter(decay, freqs, color=col, s=60, zorder=5,
                   edgecolors='white', linewidths=0.5)

        # Harmonic reference lines
        f1 = freqs[0] if len(freqs) > 0 else 100
        for n in range(1, 9):
            ax.axhline(n * f1, color='white', lw=0.4, alpha=0.15, linestyle='--')

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, max(freqs) * 1.1)
        ax.axvline(0, color='white', lw=0.8, alpha=0.4)
        ax.set_title(f'{shape_name}\n{SHAPES[shape_name]["desc"]}',
                     color=col, fontsize=10, fontweight='bold')
        ax.set_xlabel('Decay rate  (−Re λ)', color='white', fontsize=9)
        ax.set_ylabel('Frequency (Hz)  [Im λ]', color='white', fontsize=9)
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#2A3550')

        # Annotate first 6 eigenvalues with frequency value
        for j, f in enumerate(freqs[:8]):
            ax.annotate(f'{f:.0f}', (0.02, f), color='#E0E0E0',
                        fontsize=7.5, va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fname = os.path.join(OUTPUT_DIR, 'spectral_complex_plane.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight', facecolor='#0A0F1E')
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  PLOT: DENSITY PROFILES COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_density_profiles(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0A0F1E')
    fig.suptitle(
        'Non-Uniform Density Fields  ρ(r,θ)\n'
        'Sathej & Adhikari (2008) eq.(1): ρ = 1 + (σ²−1)/2 · [1 − tanh((R(r,θ)−k)/ξ)]',
        fontsize=11, color='white', fontweight='bold'
    )

    th_c = np.linspace(0, 2*np.pi, 400)
    r_disp  = np.linspace(0, 1, 200)
    th_disp = np.linspace(0, 2*np.pi, 200)
    R_disp, TH_disp = np.meshgrid(r_disp, th_disp)
    X_disp = R_disp * np.cos(TH_disp)
    Y_disp = R_disp * np.sin(TH_disp)

    for ax, (shape_name, (freqs, modes, r, th, rho)) in zip(axes, all_results.items()):
        ax.set_facecolor('#0D1520')
        col = SHAPES[shape_name]['color']

        # Interpolate rho onto display grid
        from scipy.interpolate import RegularGridInterpolator
        th_ext  = np.append(th, th[0] + 2*np.pi)
        rho_ext = np.hstack([rho, rho[:, :1]])
        interp  = RegularGridInterpolator(
            (r, th_ext), rho_ext,
            method='linear', bounds_error=False, fill_value=1.0
        )
        pts = np.column_stack([R_disp.ravel(), TH_disp.ravel() % (2*np.pi)])
        rho_disp = interp(pts).reshape(R_disp.shape)
        rho_disp[R_disp > 1.0] = np.nan

        im = ax.contourf(X_disp, Y_disp, rho_disp, levels=40,
                         cmap='hot', extend='both')
        plt.colorbar(im, ax=ax, label='ρ(r,θ)', fraction=0.046, pad=0.04)
        ax.contour(X_disp, Y_disp, rho_disp, levels=6,
                   colors='white', linewidths=0.5, alpha=0.4)
        ax.plot(np.cos(th_c), np.sin(th_c), color='white', lw=1.5)

        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'{shape_name}\n{SHAPES[shape_name]["desc"]}',
                     color=col, fontsize=10, fontweight='bold')

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'spectral_density_profiles.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight', facecolor='#0A0F1E')
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  COMPARISON TABLE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_table(all_results):
    """
    Table: row = mode number, columns = shape name + frequency.
    Matches style of Sathej Table I.
    """
    n_modes = 20
    shape_names = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor('#0A0F1E')
    ax.set_facecolor('#0A0F1E')
    ax.axis('off')

    fig.suptitle(
        'Chenda Eigenfrequency Comparison — All Three Shell Shapes\n'
        'Fourier-Chebyshev Spectral Collocation  |  '
        'Theory: Sathej & Adhikari (2008) + Howle & Trefethen (2001)',
        fontsize=12, color='white', fontweight='bold', y=0.99
    )

    # Normalise by mode 1 of cylinder (Sathej Table I style)
    cyl_f1 = all_results['Cylinder'][0][0]

    col_labels = ['#', 'Cylinder\nf (Hz)', 'Cylinder\nf/f₁',
                         'Hourglass\nf (Hz)', 'Hourglass\nf/f₁',
                         'Oval\nf (Hz)', 'Oval\nf/f₁']
    col_x = [0.03, 0.18, 0.33, 0.46, 0.60, 0.73, 0.87]
    col_colors = ['white', '#5AB4E8', '#5AB4E8', '#E8A838', '#E8A838', '#5ECF82', '#5ECF82']

    # Header
    y_top = 0.93
    for cx, lbl, cc in zip(col_x, col_labels, col_colors):
        ax.text(cx, y_top, lbl, transform=ax.transAxes,
                ha='left', va='top', fontsize=9.5, fontweight='bold', color=cc)
    ax.plot([0, 1], [0.91, 0.91], color='#4A5A6A', lw=1.5,
            transform=ax.transAxes, clip_on=False)

    row_h = (0.88) / (n_modes + 0.5)

    for i in range(n_modes):
        y = 0.905 - (i + 1) * row_h
        bg = '#0D1520' if i % 2 == 0 else '#111827'
        ax.add_patch(plt.Rectangle((0, y - row_h*0.1), 1, row_h*1.0,
                     transform=ax.transAxes, color=bg, zorder=0))

        ax.text(col_x[0], y + row_h*0.45, f'{i+1}',
                transform=ax.transAxes, ha='left', va='center',
                fontsize=9, color='#6B7280')

        for si, sname in enumerate(shape_names):
            freqs = all_results[sname][0]
            if i < len(freqs):
                f = freqs[i]
                ratio = f / cyl_f1
                scol = SHAPES[sname]['color']

                # Highlight if differs significantly from cylinder
                cyl_f = all_results['Cylinder'][0][i] if i < len(all_results['Cylinder'][0]) else f
                diff_pct = abs(f - cyl_f) / cyl_f * 100

                fw = 'bold' if diff_pct > 2.0 and sname != 'Cylinder' else 'normal'
                fc = '#FF9944' if diff_pct > 5.0 and sname != 'Cylinder' else scol

                ax.text(col_x[1 + si*2], y + row_h*0.45, f'{f:.1f}',
                        transform=ax.transAxes, ha='left', va='center',
                        fontsize=9.5, color=fc, fontweight=fw)
                ax.text(col_x[2 + si*2], y + row_h*0.45, f'{ratio:.3f}',
                        transform=ax.transAxes, ha='left', va='center',
                        fontsize=9, color=scol, alpha=0.8)

        ax.plot([0, 1], [y - row_h*0.1, y - row_h*0.1], color='#1E2A3A', lw=0.5,
                transform=ax.transAxes, clip_on=False)

    # Footer note
    ax.text(0.5, 0.005,
            'Frequencies normalised by f₁(Cylinder)  |  '
            'Orange = >5% deviation from cylinder  |  '
            'Bold = >2% deviation',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=8, color='#6B7280', style='italic')

    fname = os.path.join(OUTPUT_DIR, 'spectral_comparison_table.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight', facecolor='#0A0F1E')
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Resolution: Nr Chebyshev × Nth Fourier
    # Sathej used Nr=65, Nth=30 (concentric) and Nr=65, Nth=56 (eccentric)
    # We use Nr=25, Nth=24 for speed while keeping good accuracy
    Nr  = 25
    Nth = 24

    R_phys   = 0.125   # m
    T        = 3500.0  # N/m
    sigma_mem = 1.4    # kg/m²

    all_results = {}

    print("\n" + "="*60)
    print("  CHENDA SPECTRAL EIGENMODE SOLVER")
    print("  Fourier-Chebyshev Collocation (Sathej & Adhikari 2008)")
    print("="*60)

    for shape_name, cfg in SHAPES.items():
        print(f"\n  Computing {shape_name}...")
        freqs, modes, r, th, rho = solve_eigenmodes(
            Nr=Nr, Nth=Nth,
            density_fn=cfg['density_fn'],
            density_kw=cfg['density_kw'],
            n_modes=22,
            R_phys=R_phys, T=T, sigma_mem=sigma_mem
        )
        all_results[shape_name] = (freqs, modes, r, th, rho)

        print(f"  {'Mode':<6} {'Freq (Hz)':>10}  {'f/f1':>8}")
        print(f"  {'-'*28}")
        f1 = freqs[0]
        for i, f in enumerate(freqs[:20]):
            print(f"  {i+1:<6} {f:>10.2f}  {f/f1:>8.3f}")

    # ── Generate all figures ───────────────────────────────────────────────
    print("\n  Generating figures...")

    for shape_name, (freqs, modes, r, th, rho) in all_results.items():
        print(f"\n  Plotting {shape_name} mode grid...")
        plot_mode_grid(shape_name, freqs, modes, r, th, rho,
                       SHAPES[shape_name]['color'])

    print("\n  Plotting complex eigenvalue spectrum...")
    plot_complex_spectrum(all_results)

    print("\n  Plotting density profiles...")
    plot_density_profiles(all_results)

    print("\n  Plotting comparison table...")
    plot_comparison_table(all_results)

    print("\n  All done.")

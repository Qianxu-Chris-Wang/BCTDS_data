#!/usr/bin/env python3
"""
ringdown_fft_all_pulses.py  – 3 × 2 figure
• rows  = pulse widths   (20 ns, 50 ns, 200 ns)
• col 1 = ⟨σ⁺σ⁻⟩ vs time  (full drive-frequency sweep)
• col 2 = FFT of the ring-down part (t > pulse)

Adds:
  • Global column labels “a” (left) and “b” (right) on the first row.
  • Row labels “i / ii / iii” (white, top-right of every panel).
  • Slim, full-height colour-bars that hug each column.
Everything else (solver, parallel sweep, styling) unchanged.
"""

# ───────────────────────── imports ───────────────────────────────────────
import os, datetime, numpy as np, matplotlib
matplotlib.use("Agg")                     # comment-out for interactive use
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# –––– Hamiltonian helper functions (you already have these) ––––––
from hamiltonian_generator import (
    run_simulation_for_frequency,
    build_spin_spin_interactions_random_distribution,
    build_spin_spin_interactions_r_cube
)

# ─────────────────────── matplotlib style ───────────────────────────────
plt.rcParams.update({
    "font.size": 40,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ─────────────────────── user knobs / constants ─────────────────────────
PULSES_NS           = [20, 50, 200]        # row pulse widths
f_min, f_max        = 3.0, 5.0             # GHz sweep
FREQ_AXIS           = np.linspace(f_min, f_max, 400)
T_MAX_NS, N_T       = 1600, 1000
MAX_WORKERS         = 80

μs    = 1e3
tlist = np.linspace(0, T_MAX_NS, N_T)      # ns
dt_ns = tlist[1] - tlist[0]

# reproducible disorder / interactions
np.random.seed(42)
N_TLS     = 4
init_freqs = np.random.uniform(f_min, f_max, N_TLS)

random_dipole = True
J = 0.05

if random_dipole:
    H_int = build_spin_spin_interactions_random_distribution(
        N_TLS, -J, J, alpha_x=1.0, alpha_y=0.0, alpha_z=0.0
    )
else:
    pos = np.random.uniform(-1, 1, (N_TLS, 3))
    dip = np.random.uniform(-1, 1, (N_TLS, 3))
    dip /= np.linalg.norm(dip, axis=1, keepdims=True)
    H_int = build_spin_spin_interactions_r_cube(
        N_TLS, pos, dip, alpha_x=1.0, alpha_y=0.0, alpha_z=0.0
    )

GAMMA, GAMMA_PHI, DRIVE_AMPL = 0.001, 0.0, 0.4

# output dir
root = "../plots/ringdowns_final_layout"
os.makedirs(root, exist_ok=True)

# ─────────────────────── figure canvas ───────────────────────────────────
fig, axes = plt.subplots(
    nrows=len(PULSES_NS), ncols=2,
    figsize=(22, 14),
    gridspec_kw=dict(hspace=0.40, wspace=0.45)
)

# will store one imshow handle of each column for colour-bars
im_ring_example, im_fft_example = None, None

roman = ["i", "ii", "iii"]   # row labels

# ───────────────────────── main loop ─────────────────────────────────────
for row, pulse_ns in enumerate(PULSES_NS):

    # parallel sweep over drive frequencies
    pop_tr = np.zeros((len(FREQ_AXIS), len(tlist)))
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(FREQ_AXIS))) as pool:
        jobs = {
            pool.submit(
                run_simulation_for_frequency, f_drv, tlist,
                init_freqs, H_int,
                GAMMA, GAMMA_PHI,
                DRIVE_AMPL, pulse_ns
            ): i
            for i, f_drv in enumerate(FREQ_AXIS)
        }
        for fut in tqdm(as_completed(jobs), total=len(jobs),
                        desc=f"pulse {pulse_ns:4.1f} ns"):
            pop_tr[jobs[fut]] = fut.result()

    # FFT of ring-down part
    mask_tail   = tlist > 0            # ignore drive pulse
    pop_tail    = pop_tr[:, mask_tail]
    Nt_tail     = pop_tail.shape[1]
    fft_freq_MHz = np.fft.rfftfreq(Nt_tail, d=dt_ns) * 1e3
    fft_view_MHz = 300
    fft_mask     = fft_freq_MHz <= fft_view_MHz
    fft_freq_plot = fft_freq_MHz[fft_mask]
    pop_fft_log   = np.log10(np.abs(np.fft.rfft(pop_tail, axis=1)).clip(1e-12))

    # axes
    ax_r, ax_fft = axes[row, 0], axes[row, 1]

    # (1) population vs time
    im_r = ax_r.imshow(
        pop_tr.T, origin="lower", aspect="auto",
        extent=[FREQ_AXIS[0], FREQ_AXIS[-1], tlist[0]/μs, tlist[-1]/μs],
        cmap="inferno"
    )
    ax_r.axhline(pulse_ns/μs, ls="--", color="white")
    for f0 in init_freqs:
        ax_r.axvline(f0, ls="--", color="white", alpha=0.5, lw=2.0)
    ax_r.text(0.13, 0.85, f"{pulse_ns} ns",
              transform=ax_r.transAxes, ha="right", va="bottom",
              fontsize=22, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # (2) FFT map
    im_f = ax_fft.imshow(
        pop_fft_log[:, fft_mask].T, origin="lower", aspect="auto",
        extent=[FREQ_AXIS[0], FREQ_AXIS[-1], fft_freq_plot[0], fft_freq_plot[-1]],
        cmap="inferno"
    )
    for f0 in init_freqs:
        ax_fft.axvline(f0, ls="--", color="white", alpha=0.5)

    # labels & ticks
    ax_r.set_ylabel("Time [$\\mu$s]", fontsize=30)
    ax_fft.set_ylabel("FFT Freq. [MHz]", fontsize=30)
    if row == len(PULSES_NS)-1:
        ax_r.set_xlabel("Drive Frequency [GHz]", fontsize=30)
        ax_fft.set_xlabel("Drive Frequency [GHz]", fontsize=30)
    for a in (ax_r, ax_fft):
        a.tick_params(axis='both', which='major', labelsize=28)
    ax_fft.set_ylim(0, fft_view_MHz)

    # ── Roman row label (white, top-right) ────────────────────────────
    for a in (ax_r, ax_fft):
        a.text(0.98, 0.98, rf"\textbf{{{roman[row]}}}",
               transform=a.transAxes, fontsize=28, fontweight="bold",
               ha="right", va="top", color="white")

    # ── Column labels “a” / “b” (first row only) ─────────────────────
    if row == 0:
        ax_r.text(-0.12, 1.05, r"\textbf{a}", transform=ax_r.transAxes,
                  fontsize=42, fontweight="bold", ha="left", va="bottom")
        ax_fft.text(-0.12, 1.05, r"\textbf{b}", transform=ax_fft.transAxes,
                    fontsize=42, fontweight="bold", ha="left", va="bottom")

    # save imshow handles for colour-bars
    if im_ring_example is None:
        im_ring_example = im_r
    if im_fft_example is None:
        im_fft_example  = im_f

# ───────────────────── canvas tightening & colour-bars ───────────────────
fig.tight_layout(pad=0.0)
fig.subplots_adjust(left=0.05, right=0.90, top=0.94, bottom=0.10,
                    wspace=0.35, hspace=0.32)

bbox_left  = Bbox.union([ax.get_position() for ax in axes[:, 0]])
bbox_right = Bbox.union([ax.get_position() for ax in axes[:, 1]])
cbar_w, gap = 0.012, 0.018

# left colour-bar
cax1 = fig.add_axes([bbox_left.x1 + gap, bbox_left.y0,
                     cbar_w, bbox_left.y1 - bbox_left.y0])
cbar1 = fig.colorbar(im_ring_example, cax=cax1)
cbar1.ax.set_ylabel(r"$\langle \sigma^{+}\sigma^{-} \rangle$",
                    fontsize=30, labelpad=16)
cbar1.ax.tick_params(labelsize=28)

# right colour-bar
cax2 = fig.add_axes([bbox_right.x1 + gap, bbox_right.y0,
                     cbar_w, bbox_right.y1 - bbox_right.y0])
cbar2 = fig.colorbar(im_fft_example, cax=cax2)
# cbar2.ax.set_ylabel(r"$\ln$ FFT [arb.]", fontsize=30, labelpad=16)
# cbar2.ax.set_ylabel(r"$\log_{10}$ FFT [amp.]", fontsize=30, labelpad=16)
cbar2.ax.set_ylabel(r"Log$_{10}$ (FFT$(\langle \sigma^{+}\sigma^{-} \rangle))$ [arb.]", fontsize=30, labelpad=20)


cbar2.ax.tick_params(labelsize=28)

# ───────────────────────── save & exit ───────────────────────────────────
outfile = os.path.join(
    root, f"NTLS_{N_TLS}_ringdown_fft_all_pulses_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
)
fig.savefig(outfile, dpi=180)
plt.close(fig)
print("✓ Figure saved →", outfile)

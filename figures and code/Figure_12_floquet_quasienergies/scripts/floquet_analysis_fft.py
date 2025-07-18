#!/usr/bin/env python3
"""
floquet_ringdown_fft_layout.py
──────────────────────────────
Single-column, 3-row figure:

  (a)  ⟨σ⁺σ⁻⟩ ring-down map  (time × drive freq)
  (b)  log10 FFT{|ring-down|} vs FFT freq × drive freq
       (FFT taken over *ring-down only*: t > 0  [as in current working code])
  (c)  Floquet quasi-energies vs drive freq

Layout update:
  • Three stacked main panels share identical horizontal extent.
  • Separate right-hand column holds the two colorbars (panels a & b).
  • Bottom quasi-energy panel spans the full main width with no colorbar.
  • Larger axis-label fonts.

Physics & notation unchanged.
"""

import os, numpy as np, multiprocessing as mp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qutip import *
from matplotlib.ticker import MaxNLocator
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ────────── GLOBAL STYLE ──────────
plt.rcParams.update({
    "font.size": 45,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Light"],
    "text.usetex": True,
})

# ───────────────────────── parameters ──────────────────────────────────
omega1, omega2         = 3.5, 4.5          # TLS splittings [GHz]
j_coupling             = 0.05              # coupling (σˣ₁σˣ₂) [GHz]
drive_amp              = 0.10              # drive amplitude [arb.]
t_drive_ns, t_total_ns = 100.0, 400.0      # ns
dt_ns                  = 0.01              # time step [ns]
omega_d_vals           = np.linspace(3.0, 5.0, 300)   # drive sweep [GHz]
gamma_collective       = 0.002             # collective decay [ns⁻¹]

# FFT display cutoff (panel b)
FFT_VIEW_MHz = 300.0

# ───────────────────────── operators / static H ───────────────────────
sx1 = tensor(sigmax(), qeye(2));  sx2 = tensor(qeye(2), sigmax())
sz1 = tensor(sigmaz(), qeye(2));  sz2 = tensor(qeye(2), sigmaz())
sp1 = tensor(sigmap(),  qeye(2)); sp2 = tensor(qeye(2), sigmap())
sm1 = tensor(sigmam(),  qeye(2)); sm2 = tensor(qeye(2), sigmam())

sp_tot, sm_tot = sp1 + sp2, sm1 + sm2
pop_op = sp_tot * sm_tot                                    # ⟨σ⁺σ⁻⟩

h_static = 0.5*omega1*sz1 + 0.5*omega2*sz2 + j_coupling*sx1*sx2
psi0     = h_static.eigenstates()[1][0]

c_ops = [np.sqrt(gamma_collective) * sm_tot]
solver_opts = {"nsteps": 5000}

# time grids
t_ns = np.arange(0.0, t_total_ns, dt_ns)
t_us = t_ns / 1e3                                   # μs for plotting

# ───────────────────────── parallel worker ────────────────────────────
def run_single(task):
    idx, omega_d = task  # drive frequency [GHz]

    def drive_coeff(t, args):
        """Square cosine pulse of duration t_drive_ns (consistent w/ reference)."""
        if t <= t_drive_ns:
            return drive_amp * np.cos(omega_d * t)
        return 0.0

    h_full = [h_static, [sx1 + sx2, drive_coeff]]

    pop = mesolve(h_full, psi0, t_ns, c_ops, [pop_op],
                  options=solver_opts).expect[0].real

    # Floquet quasi-energies over one period (t_drive_ns)
    try:
        u = propagator(h_full, t_drive_ns, c_ops=[], options=solver_opts)
        phases = np.angle(np.linalg.eigvals(u.full()))
        folded = ((phases + np.pi) % (2*np.pi) - np.pi) / t_drive_ns   # GHz
        quasi  = np.sort(folded)
    except Exception:
        quasi = np.full(4, np.nan)

    return idx, pop, quasi

# ───────────────────────── simulate sweep ─────────────────────────────
n_freq = len(omega_d_vals)
heat   = np.zeros((t_ns.size, n_freq))
quasi  = np.zeros((n_freq, 4))

with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
    futures = [pool.submit(run_single, (i, w)) for i, w in enumerate(omega_d_vals)]
    for fut in tqdm(as_completed(futures), total=n_freq, desc="drive sweep"):
        idx, pop_row, q_row = fut.result()
        heat[:, idx]  = pop_row
        quasi[idx]    = q_row

# ───────────────────────── ring-down FFT (panel b data) ───────────────
# Take FFT over ring-down portion only: t > 0  (unchanged from current working code).
mask_tail   = t_ns > 0
pop_tail    = heat[mask_tail, :].T            # shape = (n_freq, Nt_tail)
Nt_tail     = pop_tail.shape[1]
dt_tail     = dt_ns                            # uniform sampling

# frequency axis in MHz
fft_freq_MHz = np.fft.rfftfreq(Nt_tail, d=dt_tail) * 1e3
fft_mask     = fft_freq_MHz <= FFT_VIEW_MHz
fft_freq_plot = fft_freq_MHz[fft_mask]

# compute rFFT along time axis
pop_fft = np.abs(np.fft.rfft(pop_tail, axis=1))  # (n_freq, N_fft)
# log scale for dynamic range; clip tiny to avoid -inf
pop_fft_log = np.log10(np.clip(pop_fft, 1e-12, None))[:, fft_mask]  # (n_freq, N_plot)

# We'll imshow with drive freq on x, FFT freq (MHz) on y; so transpose
fft_map_for_plot = pop_fft_log.T

# ───────────────────────── plotting layout ────────────────────────────
# Use a 3×2 GridSpec: column-0 = main panels; column-1 = colourbars (panels a & b).
# Bottom row spans only column-0; column-1 bottom is an empty axis (hidden) to
# preserve identical main widths.
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(22, 28))
gs  = gridspec.GridSpec(
    3, 2,
    height_ratios=[2.0, 1.5, 1.3],
    width_ratios=[1.0, 0.02],
    hspace=0.2, wspace=0.05
)

# main axes
ax_heat  = fig.add_subplot(gs[0, 0])   # panel (a)
ax_fft   = fig.add_subplot(gs[1, 0])   # panel (b)
ax_quasi = fig.add_subplot(gs[2, 0])   # panel (c)

# colourbar axes (panels a & b)
cax_heat = fig.add_subplot(gs[0, 1])
cax_fft  = fig.add_subplot(gs[1, 1])
# dummy bottom-right to keep grid rectangular; hide.
ax_dummy = fig.add_subplot(gs[2, 1])
ax_dummy.axis("off")

# ───────── panel (a) heat-map ─────────
im = ax_heat.imshow(
    heat,
    extent=[omega_d_vals[0], omega_d_vals[-1],
            t_us[0], t_us[-1]],
    origin="lower", aspect="auto", cmap="inferno"
)
ax_heat.axvline(omega1, ls="--", color="cyan", lw=4.0)  # ω₁ marker
ax_heat.axvline(omega2, ls="--", color="cyan", lw=4.0)  # ω₂ marker
ax_heat.axhline(t_drive_ns/1e3, ls='--', color='white', lw=2)
ax_heat.set_ylabel(r"Time [$\mu$s]", fontsize=52, labelpad=26)
ax_heat.tick_params(axis='both', which='major', labelsize=45, pad=16)
# labelled colourbar
cb_heat = fig.colorbar(im, cax=cax_heat)
cb_heat.set_label(r"$\langle \sigma^{+}\sigma^{-} \rangle$", fontsize=48, labelpad=20)
cb_heat.ax.tick_params(labelsize=45)
ax_heat.text(-0.12, 1.0, r"\textbf{a}", transform=ax_heat.transAxes,
             fontsize=70, fontweight="bold")
ax_heat.xaxis.set_major_locator(MaxNLocator(nbins=4))


# ───────── panel (b) ring-down FFT map ─────────
im_fft = ax_fft.imshow(
    fft_map_for_plot,
    extent=[omega_d_vals[0], omega_d_vals[-1],
            fft_freq_plot[0],    fft_freq_plot[-1]],
    origin="lower", aspect="auto", cmap="inferno"
)
ax_fft.set_ylabel("FFT Freq. [MHz]", fontsize=52, labelpad=26)
# ax_fft.set_xlabel(r"Drive Frequency [GHz]", fontsize=52, labelpad=20)
ax_fft.tick_params(axis='both', which='major', labelsize=45, pad=16)
cb_fft = fig.colorbar(im_fft, cax=cax_fft)
cb_fft.set_label(r"Log$_{10}$ (FFT$(\langle \sigma^{+}\sigma^{-} \rangle))$ [arb.]", fontsize=48, labelpad=20)
cb_fft.ax.tick_params(labelsize=45)
ax_fft.text(-0.12, 1.0, r"\textbf{b}", transform=ax_fft.transAxes,
            fontsize=70, fontweight="bold")
ax_fft.xaxis.set_major_locator(MaxNLocator(nbins=4))

# ───────── panel (c) quasi-energies (GHz→MHz) ─────────
for k in range(quasi.shape[1]):
    ax_quasi.plot(omega_d_vals, quasi[:, k]*1e3, lw=4.5, label=fr"$Q_{{{k}}}$")
ax_quasi.set_xlabel(r"Drive Frequency [GHz]", fontsize=52, labelpad=20)
ax_quasi.set_ylabel("Quasi-Energy [MHz]", fontsize=52, labelpad=26)
ax_quasi.set_xlim(omega_d_vals[0], omega_d_vals[-1])

# ax_quasi.legend(fontsize=36, ncol=2, loc="upper left", frameon=False)
handles, labels = ax_quasi.get_legend_handles_labels()

leg1 = ax_quasi.legend(handles[:2], labels[:2],
                       loc='upper left', fontsize=36,
                       frameon=False, borderaxespad=0.25)
ax_quasi.add_artist(leg1)  # keep first legend

ax_quasi.legend(handles[2:], labels[2:],
                loc='lower left', fontsize=36,
                frameon=False, borderaxespad=0.25)

ax_quasi.tick_params(axis='both', which='major', labelsize=45, pad=16)
ax_quasi.text(-0.12, 1.05, r"\textbf{c}", transform=ax_quasi.transAxes,
              fontsize=70, fontweight="bold")
ax_quasi.xaxis.set_major_locator(MaxNLocator(nbins=4))

# overall margins (room for big labels already in grid; modest padding)
fig.subplots_adjust(left=0.12, right=0.9, top=0.95, bottom=0.07,
                    hspace=0.25, wspace=0.05)

# save
os.makedirs("../plots", exist_ok=True)
outfile = "../plots/floquet_ringdown_fft_column.png"
fig.savefig(outfile, dpi=300)
plt.close(fig)
print("✓ Figure saved →", outfile)

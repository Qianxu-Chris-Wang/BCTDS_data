import numpy as np
import matplotlib.pyplot as plt
from qutip import (
    basis, qeye, destroy, tensor, mesolve, mcsolve, sesolve
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import datetime
import os
import matplotlib

# #### Preamble
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Light']
plt.rcParams['text.usetex'] = True

##############################################################################
# 1. General matplotlib settings
##############################################################################
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['legend.fontsize'] = 20
matplotlib.rcParams['figure.titlesize'] = 20

def run_simulation_for_frequency(
    freq, tlist, init_freqs, interactions,
    gamma, gamma_phi, drive_ampl, pulse_duration
):
    
    N_tls = len(init_freqs)
    
    # TLS operators
    I2 = qeye(2)
    sm_single = destroy(2)
    sx_single = sm_single + sm_single.dag()
    sy_single = -1j * (sm_single - sm_single.dag())
    sz_single = 2 * sm_single.dag() * sm_single - I2

    def tensor_op(op, k, N):
        return tensor([I2] * k + [op] + [I2] * (N - k - 1))

    sm = [tensor_op(sm_single, k, N_tls) for k in range(N_tls)]
    sx = [tensor_op(sx_single, k, N_tls) for k in range(N_tls)]
    sy = [tensor_op(sy_single, k, N_tls) for k in range(N_tls)]
    sz = [tensor_op(sz_single, k, N_tls) for k in range(N_tls)]

    Sx = sum(sx)
    Sy = sum(sy)
    Sm = sum(sm)

    T_ramp = 1.0
    def drive_coeff(t, args):
        if t < T_ramp:
            return max(0, drive_ampl * (1 - np.cos(np.pi * t / T_ramp)) / 2 * np.cos(freq * t))
        elif t < pulse_duration:
            return max(0, drive_ampl * np.cos(freq * t))
        else:
            return 0

    H_tls = sum(init_freqs[j]* sz[j] / 2 for j in range(N_tls))

    # H_int = sum(interactions[i, j] * (sx[i] * sx[j]) for i in range(N_tls) for j in range(i))  # Dipole-Dipole Interaction
    H_int = sum(interactions[i, j] for i in range(N_tls) for j in range(i))  # Dipole-Dipole Interaction
    
    H_drive = [[sum(sx), drive_coeff]]

    H = [H_tls + H_int] + H_drive
    c_ops = [np.sqrt(gamma) * Sm]  # Collective decay

    # ---------------------- Initial State ----------------------
    H_static = H_tls + H_int
    evals, evecs = H_static.eigenstates()
    psi0 = evecs[0]  # Ground state

    # -------- Solve for <Sm^dagger Sm> and <Sm> --------
    e_ops = [Sx, Sy, Sm.dag()*Sm]

    result = mesolve(H, psi0, tlist, c_ops, e_ops, progress_bar=None)

    expec_pop = result.expect[2]

    return expec_pop

# --------------------------------------------------
# Build spin–spin interactions (dipolar + alpha_x,y,z)
# --------------------------------------------------
def build_spin_spin_interactions_random_distribution(N_tls, J_min, J_max, 
                                 alpha_x=1.0, alpha_y=1.0, alpha_z=0.5):
    
    I2 = qeye(2)
    sm_single = destroy(2)
    sx_single = sm_single + sm_single.dag()
    sy_single = -1j*(sm_single - sm_single.dag())
    sz_single = 2*sm_single.dag()*sm_single - I2
    
    def tensor_op(op, k, N):
        return tensor([I2]*k + [op] + [I2]*(N - k - 1))
    
    sx_ops = [tensor_op(sx_single, i, N_tls) for i in range(N_tls)]
    sy_ops = [tensor_op(sy_single, i, N_tls) for i in range(N_tls)]
    sz_ops = [tensor_op(sz_single, i, N_tls) for i in range(N_tls)]
    
    J_mat = np.zeros((N_tls, N_tls), dtype=np.complex128)
    for i in range(N_tls):
        for j in range(i+1, N_tls):
            np.random.seed(42)
            J_rand = np.random.uniform(J_min, J_max)
            J_mat[i,j] = J_rand
            J_mat[j,i] = J_rand
    
    H_int = 0
    for i in range(N_tls):
        for j in range(i+1, N_tls):
            Jij = J_mat[i,j]
            H_int += alpha_x*Jij*(sx_ops[i]*sx_ops[j])
            H_int += alpha_y*Jij*(sy_ops[i]*sy_ops[j])
            H_int += alpha_z*Jij*(sz_ops[i]*sz_ops[j])
            # H_int += alpha_x*Jij
            # H_int += alpha_y*Jij
            # H_int += alpha_z*Jij
    return H_int


def build_spin_spin_interactions_r_cube(N_tls, positions, dipoles, 
                                 alpha_x=1.0, alpha_y=1.0, alpha_z=0.5):
    
    I2 = qeye(2)
    sm_single = destroy(2)
    sx_single = sm_single + sm_single.dag()
    sy_single = -1j*(sm_single - sm_single.dag())
    sz_single = 2*sm_single.dag()*sm_single - I2
    
    def tensor_op(op, k, N):
        return tensor([I2]*k + [op] + [I2]*(N - k - 1))
    
    sx_ops = [tensor_op(sx_single, i, N_tls) for i in range(N_tls)]
    sy_ops = [tensor_op(sy_single, i, N_tls) for i in range(N_tls)]
    sz_ops = [tensor_op(sz_single, i, N_tls) for i in range(N_tls)]
    
    J_mat = np.zeros((N_tls, N_tls), dtype=np.complex128)
    for i in range(N_tls):
        for j in range(i+1, N_tls):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-12:
                continue
            r_hat = r_vec / r
            d_i = dipoles[i]
            d_j = dipoles[j]
            base_j = (np.dot(d_i, d_j)
                      - 3*np.dot(d_i, r_hat)*np.dot(d_j, r_hat))/r**3
            J_mat[i,j] = base_j
            J_mat[j,i] = base_j
    
    H_int = 0
    for i in range(N_tls):
        for j in range(i+1, N_tls):
            Jij = J_mat[i,j]
            H_int += alpha_x*Jij*(sx_ops[i]*sx_ops[j])
            H_int += alpha_y*Jij*(sy_ops[i]*sy_ops[j])
            H_int += alpha_z*Jij*(sz_ops[i]*sz_ops[j])
            # H_int += alpha_x*Jij
            # H_int += alpha_y*Jij
            # H_int += alpha_z*Jij
    return H_int

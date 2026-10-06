import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------- parameters ----------

L = 6
U = np.array([0.0, 15.0, 15.0, 15.0, 15.0, 15.0])  # onsite interaction U_n
V = -1

EPS1_INITIAL = -15  # onsite energy of site 1 for t <= 0
EPS1_RESONANCE = 22.5 # our found value of resonance energy
eps1_compare = [5, 15]  # perturbations added to site 1 for t > 0, shown side by side
eps1_values = np.linspace(20, 25, 50)  # perturbations shown in the animation

t = np.linspace(0, 20, 1000)


# ---------- physics ----------

def build_H_matrix(site1_energy):  # two-electron Hamiltonian, Eq. (21) generalized to L sites
    eps = np.zeros(L)
    eps[0] = site1_energy
    H = np.zeros((L**2, L**2))
    for i in range(L**2):
        n_up, n_down = divmod(i, L)
        H[i, i] = eps[n_up] + eps[n_down] + (U[n_up] if n_up == n_down else 0)
        for j in range(L**2):
            m_up, m_down = divmod(j, L)
            if n_up == m_up and abs(n_down - m_down) == 1:  # down electron hops
                H[i, j] = V
            if n_down == m_down and abs(n_up - m_up) == 1:  # up electron hops
                H[i, j] = V

    return H


def solution(t, n, E1, lambda_basis, psi_0):  # Eq. (16): psi_n(t) for basis state n

    def inner_sum(lam):  # <lambda|psi(0)>, the inner sum of Eq. (16)
        eigenvector = lambda_basis[:, lam]
        overlap = 0
        for k in range(L**2):
            overlap += psi_0[k] * eigenvector[k]

        return overlap

    psi_n = np.zeros(len(t), dtype=complex)
    for lam in range(L**2):
        eigenvector = lambda_basis[:, lam]
        psi_n += np.exp(-1j * E1[lam] * t) * eigenvector[n] * inner_sum(lam)

    return psi_n


def densities(eps1):  # rho_up and rho_double per site after adding eps1 to site 1
    E1, lambda_basis = np.linalg.eigh(build_H_matrix(EPS1_INITIAL + eps1))  # H'|lambda> = E_lambda|lambda>

    probabilities = [abs(solution(t, i, E1, lambda_basis, psi_0)) ** 2 for i in range(L**2)]

    rho_up = [sum(probabilities[L*m + k] for k in range(L)) for m in range(L)]  # sum over the down electron
    rho_double = [probabilities[L*n + n] for n in range(L)]  # both electrons on site n

    return rho_up, rho_double


# ---------- plotting ----------

def plot_sites(fig, rho_up, rho_double):  # 3x2 grid in fig, one panel per site with rho_up and rho_double overlaid
    axes = fig.subplots(3, 2, sharex=True, sharey=True)
    lines = []
    for site, ax in enumerate(axes.flat):
        (line_up,) = ax.plot(t, rho_up[site], linewidth=2, label=r"$\rho_{n\uparrow}$")
        (line_double,) = ax.plot(t, rho_double[site], linewidth=2, linestyle="--", label=r"$\rho^{(2)}_n$")
        lines.append((line_up, line_double))
        ax.set_title(f"site {site + 1}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("time")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability density")

    return lines


def title(eps1):
    return rf"$\epsilon_1 = {eps1:.1f}$  (site 1: ${EPS1_INITIAL} \to {EPS1_INITIAL + eps1:.1f}$)"


# ---------- initial state: ground state of H(t <= 0) ----------

_, psi_0_eigenvectors = np.linalg.eigh(build_H_matrix(EPS1_INITIAL))
psi_0 = psi_0_eigenvectors[:, 0]


# ---------- densities for eps1 = 5 and 15 side by side ----------

fig = plt.figure(figsize=(16, 9), layout="constrained")
for subfig, eps1 in zip(fig.subfigures(1, 2), eps1_compare):
    plot_sites(subfig, *densities(eps1))
    subfig.suptitle(title(eps1))


# ---------- density for eps1 = 22.5 ----------

fig = plt.figure(figsize=(16, 9), layout="constrained")
fig.suptitle(title(EPS1_RESONANCE))
plot_sites(fig, *densities(EPS1_RESONANCE))


# ---------- animation over span of eps1 ----------

frames = [densities(e) for e in eps1_values]
fig_anim = plt.figure(figsize=(16, 9), layout="constrained")
lines = plot_sites(fig_anim, *frames[0])


def update_eps1(frame):  # swaps in the densities for eps1_values[frame]
    rho_up, rho_double = frames[frame]
    for site, (line_up, line_double) in enumerate(lines):
        line_up.set_ydata(rho_up[site])
        line_double.set_ydata(rho_double[site])
    fig_anim.suptitle(title(eps1_values[frame]))


ani = FuncAnimation(fig_anim, update_eps1, frames=len(eps1_values), interval=150)

plt.show()
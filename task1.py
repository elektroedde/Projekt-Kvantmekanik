"""Task 1: one electron in a chain of L sites.

For t <= 0 the chain is uniform (H0, Eq. 8) and the electron is in its ground state.
At t = 0 the perturbation eps1 is added to site 1, H' = H0 + eps1|1><1| (Eq. 9).
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------- parameters ----------

L = 6
V = -1.0
eps1_values = [2.0, -2.0]  # perturbations added to site 1 for t > 0, one panel each

t = np.linspace(0, 20, 1000)


# ---------- physics ----------

def solution(t, n, E1, lambda_basis, psi_0):  # Eq. (16): psi_n(t) for site n

    def inner_sum(lam):  # <lambda|psi(0)>, the inner sum of Eq. (16)
        eigenvector = lambda_basis[:, lam]
        overlap = 0
        for k in range(L):
            overlap += psi_0[k] * eigenvector[k]

        return overlap

    psi_n = np.zeros(len(t), dtype=complex)
    for lam in range(L):
        eigenvector = lambda_basis[:, lam]
        psi_n += np.exp(-1j * E1[lam] * t) * eigenvector[n] * inner_sum(lam)

    return psi_n


# ---------- initial state: ground state of H0 ----------

H0 = V * (np.eye(L, k=1) + np.eye(L, k=-1))  # Eq. (12)

_, psi_0_eigenvectors = np.linalg.eigh(H0)  # eigenvectors are the columns
psi_0 = psi_0_eigenvectors[:, 0]  # ground state of H0


# ---------- plot: one panel per eps1 ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, eps1 in zip(axes, eps1_values):
    H1 = H0.copy()
    H1[0, 0] = eps1  # H' = H0 + eps1|1><1|
    E1, lambda_basis = np.linalg.eigh(H1)  # H'|lambda> = E_lambda|lambda>

    for n in range(L):
        ax.plot(t, np.abs(solution(t, n, E1, lambda_basis, psi_0)) ** 2, label=f"site {n + 1}")

    ax.set_xlabel("time")
    ax.set_title(rf"$\epsilon_1 = {eps1:+g}$")
    ax.legend(loc="upper right")
axes[0].set_ylabel("Probability density")

fig.tight_layout()
plt.show()

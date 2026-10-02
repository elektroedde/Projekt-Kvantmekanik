import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 6
U = np.array([0.0, 15.0, 15.0, 15.0, 15.0, 15.0]) 
V = -1

def build_H_matrix(eps1):
    eps = np.zeros(L)
    eps[0] = eps1
    H = np.zeros((L**2, L**2))
    for i in range(L**2):
        n_up, n_down = divmod(i, L)
        H[i, i] = eps[n_up] + eps[n_down] + (U[n_up] if n_up == n_down else 0)   
        for j in range(L**2):
            m_up, m_down = divmod(j, L)   
            if n_up == m_up and abs(n_down - m_down) == 1:
                H[i, j] = V
       
            if n_down == m_down and abs(n_up - m_up) == 1:
                H[i, j] = V  

    return H

H0 = build_H_matrix(-15)
eps1 = 5  # value of eps1 for t > 0



t = np.linspace(0, 20, 10000)




E0, psi_0_eigenvectors = np.linalg.eigh(
    H0
)

psi_0 = psi_0_eigenvectors[
    :, 0
]


def solution(t, n, E1, lambda_basis):  # calculates equation 16, E1 and lambda_basis are the eigenvalues/vectors of H'

    def inner_sum(lam):  # calculates the inner sum of equation 16

        eigenvector = lambda_basis[:, lam]
        sum = 0
        for n in range(L**2):
            sum += psi_0[n] * eigenvector[n]

        return sum

    psi_net = np.zeros(len(t), dtype=complex)
    for lam in range(L**2):

        eigenvector = lambda_basis[:, lam]

        psi_net += np.exp(-1j * E1[lam] * t) * eigenvector[n] * inner_sum(lam)

    return psi_net


def densities(eps1):  # rho_up and rho_double for a quench eps1: -15 -> eps1
    E1, lambda_basis = np.linalg.eigh(build_H_matrix(eps1))  # H'|lambda>=E_lambda|lambda>

    wave_function_dataset = [abs(solution(t, i, E1, lambda_basis)) ** 2 for i in range(L**2)]

    rho_up = [sum(wave_function_dataset[L*m+k] for k in range(L)) for m in range(L)]
    rho_double = [wave_function_dataset[L*n+n] for n in range(L)]

    return rho_up, rho_double


def plot_animated(dataset):  # animates the probability density in time across space

    fig, ax = plt.subplots(figsize=(10, 6))

    n_lanes = len(dataset)
    n_frames = len(dataset[0])

    lanes = np.arange(n_lanes)

    # Initial values
    densities = [data[0] for data in dataset]

    (line,) = ax.plot(lanes, densities, "o-", linewidth=2, markersize=7)

    ax.set_xlabel("Lane")
    ax.set_ylabel("Probability density")

    ax.set_xlim(-0.5, n_lanes - 0.5)
    ax.set_ylim(np.min(dataset), 0.5)

    ax.set_xticks(lanes)

    def update(frame):
        densities = [data[frame] for data in dataset]

        # print(sum(densities)) double check that normalization is kept

        line.set_data(lanes, densities)
        ax.set_title(f"Time = {frame}")

        return (line,)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=5, blit=True)

    plt.tight_layout()
    plt.show()

    return ani


rho_up, rho_double = densities(eps1)

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
for site, ax in enumerate(axes.flat):  # one panel per site with rho_up and rho_double on top of each other
    ax.plot(t, rho_up[site], linewidth=2, label=r"$\rho_{n\uparrow}$")
    ax.plot(t, rho_double[site], linewidth=2, linestyle="--", label=r"$\rho^{(2)}_n$")
    ax.set_title(f"site {site + 1}")
    ax.legend()
for ax in axes[-1]:
    ax.set_xlabel("time")
for ax in axes[:, 0]:
    ax.set_ylabel("Probability density")
plt.tight_layout()


eps1_values = np.linspace(5, 25, 50)
frames = [densities(eps1) for eps1 in eps1_values]

fig_anim, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
lines = []
for site, ax in enumerate(axes.flat):
    (line_up,) = ax.plot(t, frames[0][0][site], linewidth=2, label=r"$\rho_{n\uparrow}$")
    (line_double,) = ax.plot(t, frames[0][1][site], linewidth=2, linestyle="--", label=r"$\rho^{(2)}_n$")
    lines.append((line_up, line_double))
    ax.set_title(f"site {site + 1}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
for ax in axes[-1]:
    ax.set_xlabel("time")
for ax in axes[:, 0]:
    ax.set_ylabel("Probability density")
plt.tight_layout(rect=(0, 0, 1, 0.96))


def update_eps1(frame):
    rho_up_frame, rho_double_frame = frames[frame]
    for site, (line_up, line_double) in enumerate(lines):
        line_up.set_ydata(rho_up_frame[site])
        line_double.set_ydata(rho_double_frame[site])
    fig_anim.suptitle(rf"$\epsilon_1$: $-15 \to {eps1_values[frame]:.1f}$ at $t = 0$")


ani = FuncAnimation(fig_anim, update_eps1, frames=len(eps1_values), interval=150)  # keep a reference or it stops

plt.show()
#plot_animated(rho_up).save("wavefunctions.mp4", writer="ffmpeg", fps=30)
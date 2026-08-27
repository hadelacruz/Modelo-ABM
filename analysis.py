
import numpy as np
import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import CityModel

MAX_STEPS = 150   # horizonte fijo T de la simulacion (Task 2.1)
M_RUNS = 100
B_BOOT = 2000
Z_95 = 1.959963984540054  # z_{0.025}


# ----------------------------------------------------------------------
# Task 2.1.a
# ----------------------------------------------------------------------
def run_simulation(policy_active, seed, max_steps=MAX_STEPS):
    model = CityModel(N=150, policy_active=policy_active, seed=seed)
    for _ in range(max_steps):
        if model.all_arrived():
            break
        model.step()
    return float(model.congestion_map.mean())


# ----------------------------------------------------------------------
# Task 2.1.b / c
# ----------------------------------------------------------------------
def collect_runs(m_runs=M_RUNS, max_steps=MAX_STEPS, base_seed=0):
    Y_sin = np.empty(m_runs)
    Y_con = np.empty(m_runs)
    for i in range(m_runs):
        seed = base_seed + i
        Y_sin[i] = run_simulation(policy_active=False, seed=seed, max_steps=max_steps)
        Y_con[i] = run_simulation(policy_active=True, seed=seed, max_steps=max_steps)
    return Y_sin, Y_con


def summarize(Y, label):
    mu_hat = Y.mean()
    sd_hat = Y.std(ddof=1)
    cv = sd_hat / mu_hat
    # M* = (z * CV / error_relativo)^2   (formula de tamano de muestra
    # para error relativo objetivo del 5% sobre la media, ver clase)
    m_star = (Z_95 * cv / 0.05) ** 2
    print(f"  [{label}]")
    print(f"    media hat_mu_Y        = {mu_hat:.5f}")
    print(f"    desv. estandar hat_sd = {sd_hat:.5f}")
    print(f"    coef. de variacion CV = {cv:.5f}")
    print(f"    M* (error rel. < 5%)  = {m_star:.2f}  -> redondeando: {int(np.ceil(m_star))}")
    return mu_hat, sd_hat, cv, m_star


# ----------------------------------------------------------------------
# Task 2.2.a
# ----------------------------------------------------------------------
def bootstrap_ci(Y, B=B_BOOT, seed=2024):
    """Bootstrap no parametrico (percentiles) sobre la media de Y."""
    rng = np.random.RandomState(seed)
    n = len(Y)
    boot_means = np.empty(B)
    for b in range(B):
        sample = rng.choice(Y, size=n, replace=True)
        boot_means[b] = sample.mean()
    mu_boot = boot_means.mean()
    se_boot = boot_means.std(ddof=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return mu_boot, se_boot, (ci_lo, ci_hi), boot_means


# ----------------------------------------------------------------------
# Task 2.2.b - curva de convergencia del error estandar bootstrap
# ----------------------------------------------------------------------
def bootstrap_se_convergence(Y, m_grid=range(10, 101, 10), B=B_BOOT, seed=2025):
    rng = np.random.RandomState(seed)
    ses = []
    for m in m_grid:
        sub = Y[:m]
        boot_means = np.empty(B)
        for b in range(B):
            sample = rng.choice(sub, size=m, replace=True)
            boot_means[b] = sample.mean()
        ses.append(boot_means.std(ddof=1))
    return list(m_grid), ses


# ----------------------------------------------------------------------
# Task 2.2.c - bootstrap PAREADO de Delta Y = Y_sin - Y_con
# ----------------------------------------------------------------------
def bootstrap_delta_ci(Y_sin, Y_con, B=B_BOOT, seed=2026):
    rng = np.random.RandomState(seed)
    n = len(Y_sin)
    delta_obs = Y_sin - Y_con
    boot_deltas = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        boot_deltas[b] = Y_sin[idx].mean() - Y_con[idx].mean()
    delta_hat = delta_obs.mean()
    se_delta = boot_deltas.std(ddof=1)
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    return delta_hat, se_delta, (ci_lo, ci_hi), boot_deltas


def main():
    print("=" * 70)
    print("Task 2.1 - M=100 corridas por escenario")
    print("=" * 70)
    Y_sin, Y_con = collect_runs()
    np.save("Y_sin.npy", Y_sin)
    np.save("Y_con.npy", Y_con)

    plt.figure(figsize=(7, 5))
    bins = np.linspace(min(Y_sin.min(), Y_con.min()), max(Y_sin.max(), Y_con.max()), 20)
    plt.hist(Y_sin, bins=bins, alpha=0.6, label="Sin politica", color="#d62728", edgecolor="k")
    plt.hist(Y_con, bins=bins, alpha=0.6, label="Con politica", color="#2ca02c", edgecolor="k")
    plt.xlabel("Y = indice de congestion promedio (veh/celda) al paso T")
    plt.ylabel("Frecuencia (de 100 corridas)")
    plt.title("Distribucion de Y por escenario (M=100 corridas c/u)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_histograma_congestion.png", dpi=150)
    plt.close()

    print("\nEstadisticos (Task 2.1.c):")
    stats_sin = summarize(Y_sin, "sin_politica")
    stats_con = summarize(Y_con, "con_politica")

    print()
    print("=" * 70)
    print("Task 2.2.a - Bootstrap (B=2000) por escenario")
    print("=" * 70)
    mu_boot_sin, se_boot_sin, ci_sin, _ = bootstrap_ci(Y_sin)
    mu_boot_con, se_boot_con, ci_con, _ = bootstrap_ci(Y_con)
    print(f"  sin_politica: mu_boot={mu_boot_sin:.5f}  se_boot={se_boot_sin:.5f}  "
          f"IC95%=({ci_sin[0]:.5f}, {ci_sin[1]:.5f})")
    print(f"  con_politica: mu_boot={mu_boot_con:.5f}  se_boot={se_boot_con:.5f}  "
          f"IC95%=({ci_con[0]:.5f}, {ci_con[1]:.5f})")

    print()
    print("=" * 70)
    print("Task 2.2.b - Curva de convergencia del error estandar bootstrap")
    print("=" * 70)
    m_grid, se_sin = bootstrap_se_convergence(Y_sin, seed=2025)
    _, se_con = bootstrap_se_convergence(Y_con, seed=2027)
    for m, s1, s2 in zip(m_grid, se_sin, se_con):
        print(f"  M={m:3d}  SE_boot(sin)={s1:.5f}   SE_boot(con)={s2:.5f}")

    plt.figure(figsize=(7, 5))
    plt.plot(m_grid, se_sin, marker="o", label="Sin politica", color="#d62728")
    plt.plot(m_grid, se_con, marker="s", label="Con politica", color="#2ca02c")
    plt.xlabel("Numero de corridas acumuladas (M)")
    plt.ylabel("Error estandar bootstrap de la media")
    plt.title("Convergencia del error estandar bootstrap")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_convergencia_bootstrap.png", dpi=150)
    plt.close()

    print()
    print("=" * 70)
    print("Task 2.2.c - Delta Y = Y_sin - Y_con (bootstrap pareado)")
    print("=" * 70)
    delta_hat, se_delta, ci_delta, _ = bootstrap_delta_ci(Y_sin, Y_con)
    print(f"  Delta_hat_Y = {delta_hat:.5f}")
    print(f"  SE_boot(Delta_Y) = {se_delta:.5f}")
    print(f"  IC 95% (percentiles) = ({ci_delta[0]:.5f}, {ci_delta[1]:.5f})")
    print(f"  El IC incluye 0? {'SI' if ci_delta[0] <= 0 <= ci_delta[1] else 'NO'}")


if __name__ == "__main__":
    main()

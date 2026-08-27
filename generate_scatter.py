import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import generate_commuter_attributes, SIGMA

N = 150
SEED = 42


def main():
    rng = np.random.RandomState(SEED)
    incomes, distances = generate_commuter_attributes(N, rng)

    r = np.corrcoef(incomes, distances)[0, 1]
    cov = np.cov(incomes, distances, ddof=1)[0, 1]

    print(f"N = {N}, semilla = {SEED}")
    print(f"Media muestral: ingreso={incomes.mean():.2f}  distancia={distances.mean():.4f}")
    print(f"SD muestral:    ingreso={incomes.std(ddof=1):.2f}  distancia={distances.std(ddof=1):.4f}")
    print(f"Covarianza muestral: {cov:.4f}")
    print(f"Correlacion muestral r: {r:.4f}")
    print(f"Signo covarianza teorica (SIGMA_12={SIGMA[0,1]}): "
          f"{'consistente' if np.sign(cov) == np.sign(SIGMA[0,1]) else 'INCONSISTENTE'}")

    plt.figure(figsize=(6, 5))
    plt.scatter(incomes, distances, alpha=0.6, edgecolor="k", linewidth=0.3)
    plt.xlabel("Ingreso mensual (Q)")
    plt.ylabel("Distancia al trabajo (km)")
    plt.title(f"Ingreso vs distancia (N={N}) \u2014 r = {r:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_scatter_ingreso_distancia.png", dpi=150)
    print("Figura guardada en fig_scatter_ingreso_distancia.png")


if __name__ == "__main__":
    main()


import numpy as np
from model import CityModel, MU, N_COMMUTERS


def test_scheduler():
    print("=" * 70)
    print("Task 1.3.a - Prueba de scheduler (RandomActivation)")
    print("=" * 70)
    # N pequeno y politica activa en False para que ningun CommuterAgent
    # llegue a destino/desaparezca durante las 3 corridas de prueba, y asi
    # el conteo de activaciones por paso sea comparable.
    model = CityModel(N=10, policy_active=False, seed=123)

    from agents import CommuterAgent
    activation_log = []          # se llena en cada step() de cada agente
    original_step = CommuterAgent.step

    def logging_step(self):
        activation_log.append(self.unique_id)
        original_step(self)

    CommuterAgent.step = logging_step
    try:
        orders = []
        for t in range(3):
            activation_log.clear()
            model.step()
            orders.append(list(activation_log))
            print(f"  Paso {t}: orden real de activacion -> {orders[-1]}")
    finally:
        CommuterAgent.step = original_step  # restaurar el metodo original

    distinct_orders = len({tuple(o) for o in orders})
    each_agent_once = all(len(o) == len(set(o)) == 10 for o in orders)
    passed = distinct_orders > 1 and each_agent_once
    print(f"  Ordenes distintos observados: {distinct_orders} de {len(orders)} pasos")
    print(f"  Los 10 agentes se activan exactamente una vez por paso: {each_agent_once}")
    print(f"  RESULTADO: {'PASA' if passed else 'FALLA'}")
    print()
    return passed


def test_initialization():
    print("=" * 70)
    print("Task 1.3.b - Prueba de inicializacion (media y desviacion estandar)")
    print("=" * 70)
    model = CityModel(N=N_COMMUTERS, policy_active=False, seed=42)
    incomes, distances = model.incomes, model.distances

    mean_income, std_income = incomes.mean(), incomes.std(ddof=1)
    mean_distance, std_distance = distances.mean(), distances.std(ddof=1)

    print(f"  Ingreso:   media muestral = {mean_income:10.2f}  (mu = {MU[0]:.2f})   "
          f"sd muestral = {std_income:10.2f} (sigma teorica = {np.sqrt(4_000_000):.2f})")
    print(f"  Distancia: media muestral = {mean_distance:10.4f}  (mu = {MU[1]:.2f})  "
          f"sd muestral = {std_distance:10.4f} (sigma teorica = {np.sqrt(4):.2f})")

    # Margen razonable: dentro de +-3 errores estandar de la media (Teorema
    # Central del Limite, N=150) respecto de mu.
    se_income = np.sqrt(4_000_000 / N_COMMUTERS)
    se_distance = np.sqrt(4 / N_COMMUTERS)
    ok_income = abs(mean_income - MU[0]) < 3 * se_income
    ok_distance = abs(mean_distance - MU[1]) < 3 * se_distance
    passed = ok_income and ok_distance
    print(f"  |media_ingreso - mu_ingreso|     = {abs(mean_income - MU[0]):.2f}  "
          f"< 3*SE ({3*se_income:.2f})? {ok_income}")
    print(f"  |media_distancia - mu_distancia| = {abs(mean_distance - MU[1]):.4f} "
          f"< 3*SE ({3*se_distance:.4f})? {ok_distance}")
    print(f"  RESULTADO: {'PASA' if passed else 'FALLA'}")
    print()
    return passed


def test_conservation():
    print("=" * 70)
    print("Task 1.3.c - Prueba de conservacion (agentes activos decrece monotonamente)")
    print("=" * 70)
    model = CityModel(N=60, policy_active=True, seed=7)
    counts = [model.n_active_commuters()]
    for t in range(80):
        if model.all_arrived():
            break
        model.step()
        counts.append(model.n_active_commuters())

    non_increasing = all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    strictly_decreased_at_some_point = counts[-1] < counts[0]
    print(f"  Conteo de commuters activos por paso: {counts}")
    print(f"  Secuencia no creciente (nunca sube) en {len(counts)} pasos: {non_increasing}")
    print(f"  Bajo de {counts[0]} a {counts[-1]} activos: {strictly_decreased_at_some_point}")
    # La prueba pedida es de MONOTONICIDAD (nunca crece), no que necesariamente
    # llegue a 0: con la regla "espera si esta ocupada" (sin negociacion de
    # intercambio de posiciones) puede emerger un embotellamiento (deadlock)
    # cuando dos o mas agentes quedan bloqueados mutuamente para siempre. Esto
    # es una propiedad EMERGENTE valida del modelo (ver Task 3.1 - Emergencia),
    # no una violacion de la monotonicidad: el conteo de activos simplemente
    # deja de bajar, nunca sube.
    if counts[-1] > 0:
        print("  NOTA: el conteo se estanca en un valor > 0 -> embotellamiento "
              "(deadlock) emergente por bloqueo mutuo de celdas; ver discusion "
              "en el informe. La monotonicidad (nunca sube) se mantiene igual.")
    passed = non_increasing and strictly_decreased_at_some_point
    print(f"  RESULTADO: {'PASA' if passed else 'FALLA'}")
    print()
    return passed


if __name__ == "__main__":
    r1 = test_scheduler()
    r2 = test_initialization()
    r3 = test_conservation()
    print("=" * 70)
    print("RESUMEN Task 1.3")
    print("=" * 70)
    print(f"  a. Scheduler:      {'PASA' if r1 else 'FALLA'}")
    print(f"  b. Inicializacion: {'PASA' if r2 else 'FALLA'}")
    print(f"  c. Conservacion:   {'PASA' if r3 else 'FALLA'}")

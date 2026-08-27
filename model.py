
import numpy as np
import mesa
from mesa.time import RandomActivation
from mesa.space import MultiGrid

from agents import CommuterAgent, VehicleAgent

GRID_SIZE = 20
N_COMMUTERS = 150

# Parametros de la normal multivariada (ver enunciado, Task 1.1)
MU = np.array([8500.0, 6.5])
SIGMA = np.array([
    [4_000_000.0, -800.0],
    [-800.0,          4.0],
])


def generate_commuter_attributes(N, rng):
    L = np.linalg.cholesky(SIGMA)      # SIGMA = L @ L.T
    Z = rng.normal(size=(N, 2))        # Z_i ~ N(0, I_2), i = 1..N
    X = MU + Z @ L.T                   # X_i = MU + L @ Z_i  (version vectorizada)
    incomes, distances = X[:, 0], X[:, 1]
    return incomes, distances


class CityModel(mesa.Model):

    def __init__(self, N=N_COMMUTERS, width=GRID_SIZE, height=GRID_SIZE,
                 policy_active=False, seed=None):
        super().__init__()
        if seed is not None:
            self.reset_randomizer(seed)   # semilla del self.random de Mesa (scheduler, grid)
        self._rng = np.random.RandomState(seed)  # rng propio para Cholesky (Task 1.1a)

        self.num_agents = N
        self.policy_active = policy_active
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)  # Task 1.1c
        self.congestion_map = np.zeros((width, height), dtype=int)  # Task 1.2b

        incomes, distances = generate_commuter_attributes(N, self._rng)
        self.incomes = incomes
        self.distances = distances

        for i in range(N):
            start = self._random_cell(width, height)
            dest = self._random_cell(width, height)
            while dest == start:
                dest = self._random_cell(width, height)
            agent = CommuterAgent(i, self, incomes[i], distances[i], start, dest)
            self.schedule.add(agent)
            self.grid.place_agent(agent, start)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Congestion_index": lambda m: float(m.congestion_map.mean()),
                "Active_commuters": lambda m: sum(
                    1 for a in m.schedule.agents if isinstance(a, CommuterAgent)
                ),
            }
        )
        # Registra el estado inicial (paso 0) para las pruebas de conservacion
        self._update_congestion_map()
        self.datacollector.collect(self)

    def _random_cell(self, width, height):
        return (self.random.randrange(width), self.random.randrange(height))

    def step(self):
        self.schedule.step()
        self._update_congestion_map()
        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    # Task 1.2b
    # ------------------------------------------------------------------
    def _update_congestion_map(self):
        self.congestion_map[:, :] = 0
        for agent in self.schedule.agents:
            if isinstance(agent, VehicleAgent):
                x, y = agent.pos
                self.congestion_map[x, y] += 1

    def n_active_commuters(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, CommuterAgent))

    def all_arrived(self):
        return self.n_active_commuters() == 0

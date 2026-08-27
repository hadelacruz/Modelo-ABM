
import mesa


class VehicleAgent(mesa.Agent):
    """Vehiculo en circulacion asociado a un CommuterAgent que usa automovil."""

    def __init__(self, unique_id, model, owner):
        super().__init__(unique_id, model)
        self.owner = owner  # CommuterAgent dueno de este vehiculo

    def step(self):
        # El vehiculo no decide movimiento propio: CommuterAgent.step() mueve
        # a su dueno y a su vehiculo a la vez, para que ambos queden siempre
        # en la misma celda. Se mantiene en el schedule solo para que el
        # conteo de congestion (Task 1.2b) lo contabilice como agente aparte.
        pass


class CommuterAgent(mesa.Agent):
    """Persona que se desplaza diariamente desde un origen hasta un destino."""

    def __init__(self, unique_id, model, income, distance, start_pos, dest_pos):
        super().__init__(unique_id, model)
        self.income = income        # Q/mes (ingreso mensual)
        self.distance = distance    # km al trabajo
        self.destination = dest_pos
        self.mode = None            # se fija la primera vez que corre step()
        self.vehicle = None         # VehicleAgent asociado si mode == 'automovil'

    # ------------------------------------------------------------------
    # Task 1.1b
    # ------------------------------------------------------------------
    def choose_transport(self):
        if self.distance <= 3.0 and self.model.policy_active:
            self.mode = "bicicleta"
        else:
            self.mode = "automovil"

    def _spawn_vehicle_if_needed(self):
        if self.mode == "automovil" and self.vehicle is None:
            vid = f"veh_{self.unique_id}"
            self.vehicle = VehicleAgent(vid, self.model, owner=self)
            self.model.schedule.add(self.vehicle)
            self.model.grid.place_agent(self.vehicle, self.pos)

    def _manhattan_step_towards(self, target):
        """Un paso de un camino Manhattan minimo hacia target: corrige primero
        el eje (x u y) con mayor diferencia restante."""
        x, y = self.pos
        tx, ty = target
        dx, dy = tx - x, ty - y
        if dx == 0 and dy == 0:
            return self.pos
        if abs(dx) >= abs(dy) and dx != 0:
            return (x + (1 if dx > 0 else -1), y)
        return (x, y + (1 if dy > 0 else -1))

    # ------------------------------------------------------------------
    # Task 1.2a / Task 1.2c
    # ------------------------------------------------------------------
    def step(self):
        if self.mode is None:
            # Primera activacion: elige modo (Task 1.1b) y, si aplica, crea
            # su VehicleAgent (Task 1.2b).
            self.choose_transport()
            self._spawn_vehicle_if_needed()

        if self.pos == self.destination:
            self._remove()
            return

        next_cell = self._manhattan_step_towards(self.destination)

        # Si la celda objetivo esta ocupada por otro CommuterAgent, se espera
        # en la posicion actual (Task 1.2a).
        occupants = self.model.grid.get_cell_list_contents([next_cell])
        occupied_by_commuter = any(
            isinstance(a, CommuterAgent) and a is not self for a in occupants
        )

        if not occupied_by_commuter:
            self.model.grid.move_agent(self, next_cell)
            if self.vehicle is not None:
                self.model.grid.move_agent(self.vehicle, next_cell)

        if self.pos == self.destination:
            self._remove()

    def _remove(self):
        """Task 1.2c: al llegar a destino, el CommuterAgent (y su vehiculo, si
        tiene) se eliminan tanto del schedule como del grid."""
        if self.vehicle is not None:
            if self.vehicle in self.model.schedule.agents:
                self.model.schedule.remove(self.vehicle)
            self.model.grid.remove_agent(self.vehicle)
            self.vehicle = None
        self.model.schedule.remove(self)
        self.model.grid.remove_agent(self)

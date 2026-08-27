# Laboratorio 4

Modelo ABM en Mesa de un sistema de bicicletas compartidas: se compara la
congestión vehicular de un centro urbano (cuadrícula 20×20) con y sin la
política de bicicletas, usando 100 corridas por escenario y bootstrap.

## Estructura del proyecto

```
.
├── Lab4_Bicicletas_Compartidas.ipynb   # INFORME + CODIGO. Correr "Run All" reproduce todo.
├── agents.py              # CommuterAgent y VehicleAgent (Task 1.1b, 1.2a, 1.2c)
├── model.py               # CityModel: Cholesky, scheduling, congestion_map (Task 1.1a/c, 1.2b)
├── lab4_tests.py          # Pruebas de verificacion (Task 1.3)
├── generate_scatter.py    # Genera fig_scatter_ingreso_distancia.png (Task 1.1a) fuera del notebook
├── analysis.py            # Analisis estadistico completo (Task 2.1 y 2.2)
├── requirements.txt
├── .gitignore
└── README.md
```

## Requisitos

- Python 3.10+
- **Mesa 2.4.x** (no Mesa 3.x — la API clásica `RandomActivation` /
  `SimultaneousActivation` que pide el enunciado fue removida en Mesa 3)
- numpy, matplotlib
- jupyter / notebook (para abrir y correr el `.ipynb`)

Instalación:

```bash
pip install -r requirements.txt
```

## Cómo correr

**Todo desde el notebook (recomendado):**

```bash
jupyter notebook Lab4_Bicicletas_Compartidas.ipynb
# luego: Kernel -> Restart & Run All
```

o, sin abrir la interfaz, para ejecutarlo y guardarlo con los resultados
desde la terminal:

```bash
jupyter execute --inplace Lab4_Bicicletas_Compartidas.ipynb
```

Tarda unos ~25 segundos en total (la mayor parte es correr las 200
simulaciones de la Task 2.1: 100 corridas × 2 escenarios).

**Por partes, si solo se necesita una sección sin Jupyter:**

```bash
python3 generate_scatter.py   # Task 1.1a: scatter + correlacion muestral
python3 lab4_tests.py         # Task 1.3: pruebas de verificacion
python3 analysis.py           # Task 2: M=100 corridas x 2 escenarios + bootstrap
```

Todo es reproducible (semillas fijas); correrlo de nuevo debe dar
exactamente los mismos números y figuras reportados en el notebook.

## Notas de diseño

- Se usa **Mesa 2.4.0** por compatibilidad con `RandomActivation`/`SimultaneousActivation`.
- Comparación de escenarios con **números aleatorios comunes** (misma
  semilla en ambos escenarios por corrida) para reducir varianza y permitir
  un bootstrap **pareado** de `ΔY`.
- El horizonte de simulación es fijo (`T=150` pasos) porque el modelo
  puede caer en un **embotellamiento (deadlock) emergente** — ver Task 1.3.c
  y la sección de limitaciones en el notebook.
- `analysis.py` solo fuerza el backend `Agg` de matplotlib cuando se corre
  como script de consola (`if __name__ == "__main__"`); si se importa desde
  el notebook, respeta el backend inline ya configurado (de lo contrario las
  gráficas de la Task 2 no se verían dentro del notebook).

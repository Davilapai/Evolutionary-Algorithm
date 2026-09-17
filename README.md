# Algoritmo evolutivo para N reinas

Implementación experimental de un algoritmo genético en Python para explorar soluciones al problema de las N reinas. Cada individuo se representa como una permutación: el índice representa una columna y el valor almacenado representa la fila de la reina en esa columna.

El proyecto se encuentra en desarrollo y contiene dos partes relacionadas, pero todavía no integradas completamente: el ciclo evolutivo en `genetic_algorithm/` y el motor de evaluación en `engine/`.

## Estructura

```text
engine/
└── Engine.py                 # Validación y evaluación de estados de N reinas
genetic_algorithm/
├── Individual.py             # Individuos, generación y mutación
└── Simulation.py             # Población y ciclo evolutivo
docs/
└── Taller_Investigativo.tex  # Documento del taller
requirements.txt              # Dependencias de Python
```

## Requisitos

- Python 3.9 o posterior
- NumPy

Instala las dependencias desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
```

## Ejecución

La simulación incluida puede ejecutarse desde la raíz con:

```bash
python genetic_algorithm/Simulation.py
```

Actualmente utiliza estos parámetros definidos al final de `Simulation.py`:

- Tamaño del tablero: `100`
- Población: `1000` individuos
- Generaciones: `200`
- Factor de mutación: `0.001`

Durante la ejecución se imprime el mejor estado encontrado en cada generación y, al final, el estado de la población.

## Funcionamiento

1. Se crea una población de permutaciones válidas.
2. Se evalúan y ordenan los individuos según su `fitness`.
3. Se seleccionan parejas entre aproximadamente dos tercios de la población.
4. Se generan descendientes mediante cruce parcialmente mapeado (PMX), conservando la estructura de permutación.
5. Se generan individuos adicionales a partir de padres seleccionados para completar la población.
6. Se aplica mutación por intercambio de posiciones.
7. Se reemplaza la población y se repite el proceso durante el número de generaciones configurado.

El motor de `engine/Engine.py` calcula las colisiones entre diagonales usando conteo de pares en tiempo lineal respecto al número de reinas. Su función de fitness parte del número máximo de pares posibles y resta las colisiones.

## Estado actual y pendientes

El repositorio es un prototipo. La simulación todavía usa una función de prueba en `Individual.fitness()` y no el fitness calculado por `Engine`. Por ello, las salidas actuales no deben interpretarse como soluciones optimizadas para N reinas hasta conectar ambas partes.

También es necesario revisar la firma de `Engine.calculate_cost` antes de invocarla como método de instancia, además de agregar pruebas para validar estados, costos y soluciones sin conflictos diagonales.

## Autoría y uso de IA

Este README fue actualizado con asistencia de inteligencia artificial (IA) a partir del contenido actual del repositorio. El código y la documentación forman parte de un trabajo en desarrollo y deben ser revisados, comprendidos y validados por sus autores.

# Algoritmo evolutivo

Implementación experimental de un algoritmo genético en Python. El proyecto representa cada individuo como una permutación de valores y aplica operaciones evolutivas sobre una población: evaluación, selección de padres, cruce, mutación y reemplazo.

## Estructura

```text
genetic_algorithm/
├── Individual.py     # Representación, generación y mutación de individuos
├── Simulation.py     # Población y ciclo de simulación
└── requirements.txt  # Dependencias de Python
```

## Requisitos

- Python 3.9 o posterior
- NumPy

Instala la dependencia desde la carpeta del algoritmo:

```bash
cd genetic_algorithm
python -m pip install -r requirements.txt
```

## Ejecución

Desde `genetic_algorithm/`, ejecuta:

```bash
python Simulation.py
```

La simulación de ejemplo crea una población de 100 individuos para un problema de tamaño 6, ejecuta 100 generaciones y utiliza un factor de mutación de `0.01`.

## Funcionamiento

1. Se inicializa una población con individuos legales representados por permutaciones.
2. Se evalúa y ordena la población según el valor de `fitness`.
3. Se seleccionan parejas usando aproximadamente dos tercios de la población.
4. Se generan descendientes mediante un cruce parcialmente mapeado (PMX), que conserva la integridad de las permutaciones.
5. Se aplica mutación por intercambio de posiciones.
6. Se completa la población y se repite el ciclo durante el número de generaciones indicado.

## Estado actual

El proyecto es un prototipo en desarrollo. La función `Individual.fitness()` todavía devuelve `0`, por lo que la evaluación real del problema objetivo debe implementarse antes de interpretar los resultados como una solución optimizada. La función `random_Individual_ilegal` se conserva para experimentar con individuos que no son permutaciones válidas.

## Autoría y uso de IA

Este README fue elaborado con asistencia de inteligencia artificial (IA) a partir del contenido actual del repositorio. La implementación del proyecto también puede encontrarse en proceso de desarrollo y debe ser revisada y validada por sus autores.

#!/usr/bin/env python3
"""
multi_experiment_runner.py - Generador autónomo de datos para experimentación multidimensional (N-Reinas).

Motor de simulación puro, de alto rendimiento y sin dependencias de graficación.
Genera conjuntos de datos consolidados en disco estructurados en formatos CSV y JSON.

========================================================================================
¿QUÉ ES EL PARÁMETRO '--points'?
========================================================================================
El parámetro '--points' define el número de muestras o valores discretos en los que se
divide el intervalo continuo de factor de mutación [min_mut, max_mut].

En el código se calcula internamente mediante:
    mutation_values = [round(float(x), 6) for x in np.linspace(min_mut, max_mut, points)]

Ejemplo:
    Si se configuran --min-mut 0.001, --max-mut 0.20 y --points 20 (valores por defecto),
    np.linspace genera 20 valores de mutación uniformemente distribuidos:
    [0.001000, 0.011474, 0.021947, 0.032421, 0.042895, ..., 0.200000]

Cada uno de esos valores de mutación se evalúa para CADA configuración analizada
(ya sea para cada tamaño de población M o para cada dimensión de tablero N).

========================================================================================
¿CÓMO SE CALCULA EL TOTAL DE EXPERIMENTOS A GENERAR?
========================================================================================
La cantidad total de simulaciones que ejecutará el script se calcula con la fórmula:

    Total de Experimentos = (Cantidad de configuraciones) * (points)

1. MODO POBLACIÓN (--mode population, N=25 fijo por defecto o con -n):
   La cantidad de tamaños de población evaluados se define con:
       |M| = floor((m_max - m_min) / m_step) + 1  (o len(m_list) si es lista manual)
   
   Total = |M| * points

   Ejemplo para ~1,000 experimentos:
       --m-min 20 --m-max 500 --m-step 10  --> |M| = (500 - 20)/10 + 1 = 49 tamaños de población
       --points 20                         --> 20 valores de mutación por cada población
       Total = 49 * 20 = 980 experimentos.

2. MODO TABLERO (--mode board, m=100 fija por defecto o con --m):
   La cantidad de tamaños de tablero evaluados se define con:
       |N| = floor((n_max - n_min) / n_step) + 1  (o len(n_list) si es lista manual)
   
   Total = |N| * points

   Ejemplo para ~1,000 experimentos:
       --n-min 8 --n-max 100 --n-step 2    --> |N| = (100 - 8)/2 + 1 = 47 tamaños de tablero
       --points 20 o 21                    --> 20 o 21 valores de mutación por cada tablero
       Total = 47 * 20 = 940 experimentos.

========================================================================================
OPTIMIZACIONES DE RENDIMIENTO:
========================================================================================
  --patience K:
      Parada temprana por estancamiento (Early Stopping). Si el mejor fitness no mejora
      tras K generaciones consecutivas, la corrida se detiene de inmediato.
      Evita desperdiciar miles de generaciones en mutaciones destructivas o mínimos locales.
      Ej: --patience 1000

  --resume:
      Reanuda una ejecución previa leyendo el CSV de salida y saltando automáticamente
      las configuraciones que ya fueron completadas. Permite cancelar con Ctrl+C y retomar.

========================================================================================
MODALIDADES Y ARCHIVOS DE SALIDA DIFERENCIADOS:
========================================================================================
  1. Modo Tablero ('board'):
     Varía N en un rango o lista con m fija (default m=100) y 'points' mutaciones.
     Salidas por defecto:
       - data/results_board.csv
       - data/multi_metrics_board.json
       - data/multi_data_board.json

  2. Modo Población ('population'):
     Fija el tablero en N=25 (default o con -n) y varía m en un rango con steps o lista manual,
     evaluando 'points' mutaciones para cada m.
     Salidas por defecto:
       - data/results_population.csv
       - data/multi_metrics_population.json
       - data/multi_data_population.json

========================================================================================
EJEMPLOS DE USO:
========================================================================================
    Reanudar ejecución acelerada con Early Stopping (patience=1000):
        python3 multi_experiment_runner.py --mode board --n-min 8 --n-max 100 --n-step 2 --points 20 --max-runs 10000 --m 250 --resume --patience 1000

    Modo Población (~1,000 experimentos: N=25 fijo, M de 20 a 500, paso 10, 20 mutaciones):
        python3 multi_experiment_runner.py --mode population --m-min 20 --m-max 500 --m-step 10 --points 20
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Configurar rutas para importar módulos del proyecto con independencia del directorio de ejecución
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "genetic_algorithm").exists():
    PROJECT_ROOT = SCRIPT_DIR
else:
    PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genetic_algorithm.Individual import Individual  # noqa: E402


def crossover_pmx(p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
    """Cruce parcialmente asignado (PMX) preservando la permutación legal sin duplicados."""
    l = random.randint(0, n - 1)
    r = random.choice([i for i in range(1, n + 1) if i != l])
    if l > r:
        l, r = r, l

    son = np.empty(n, dtype=int)
    son[l:r] = p1[l:r]

    p1_slice = set(p1[l:r])
    p1_pos = {int(p1[idx]): idx for idx in range(l, r)}

    for i in (*range(0, l), *range(r, n)):
        candidate = int(p2[i])
        while candidate in p1_slice:
            candidate = int(p2[p1_pos[candidate]])
        son[i] = candidate

    return son


def run_trial(args_tuple: Tuple[int, int, int, int, float, Optional[int], Optional[int]]) -> Dict[str, Any]:
    """
    Ejecuta una corrida del algoritmo genético y registra la evolución generacional del fitness.
    Soporta parada por convergencia (0 conflictos) y parada temprana por estancamiento (patience).
    """
    idx, n, m, max_runs, mutation_factor, seed, patience = args_tuple

    if seed is not None:
        random.seed(seed + idx)
        np.random.seed((seed + idx) % (2**32))

    max_fitness = (n * (n - 1)) // 2
    t0 = time.perf_counter()

    # 1. Inicialización
    population = [Individual.random_Individual_legal(n) for _ in range(m)]

    found_generation = None
    generations_log = []
    best_state = None
    best_conflicts = max_fitness
    best_fit = 0.0

    best_seen_conflicts = max_fitness
    stagnation_counter = 0

    for gen in range(max_runs):
        # 2. Evaluación
        fitness_list = np.array([ind.fitness() for ind in population])
        order = np.argsort(fitness_list)[::-1]
        population = [population[i] for i in order]
        fitness_sorted = fitness_list[order]

        best_fit = float(fitness_sorted[0])
        mean_fit = float(np.mean(fitness_sorted))
        best_conflicts = int(max_fitness - best_fit)
        best_state = [int(v) for v in population[0].state]

        # Registro generacional de fitness para métricas
        generations_log.append({
            "generation": int(gen),
            "best_fitness": best_fit,
            "mean_fitness": round(mean_fit, 2),
            "worst_fitness": float(fitness_sorted[-1]),
            "best_conflicts": best_conflicts,
        })

        # Comprobar solución óptima (0 colisiones diagonales)
        if best_conflicts == 0:
            found_generation = int(gen)
            break

        # Comprobar parada temprana por estancamiento (patience)
        if best_conflicts < best_seen_conflicts:
            best_seen_conflicts = best_conflicts
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            if patience is not None and stagnation_counter >= patience:
                break

        # 3. Selección de padres
        parents = []
        for i in range(0, 2 * m // 3 - 1, 2):
            parents.append((population[i], population[i + 1]))

        # 4. Cruce PMX
        new_population = []
        for p1, p2 in parents:
            new_population.append(Individual(crossover_pmx(p1.state, p2.state, n)))
            new_population.append(Individual(crossover_pmx(p2.state, p1.state, n)))

        needed = m - 2 - len(new_population)
        for _ in range(needed):
            x = random.randint(0, (m - 1) // 2)
            y = random.randint(0, (m - 1) // 2)
            new_population.append(Individual(crossover_pmx(population[x].state, population[y].state, n)))

        # 5. Mutación por intercambio (swap)
        new_population = [ind.mutate(mutation_factor) for ind in new_population]

        # 6. Elitismo determinista (conservar los dos mejores intactos)
        new_population.append(population[0])
        new_population.append(population[1])

        # 7. Reemplazo
        population = new_population

    total_time_s = time.perf_counter() - t0

    return {
        "idx": int(idx),
        "n": int(n),
        "poblacion": int(m),
        "mutacion": float(mutation_factor),
        "generacion_encontrada": found_generation,
        "exito": bool(best_conflicts == 0),
        "mejor_fitness": best_fit,
        "max_fitness": int(max_fitness),
        "conflictos_finales": int(best_conflicts),
        "generaciones_totales": int(len(generations_log)),
        "tiempo_s": round(total_time_s, 4),
        "best_state": best_state,
        "generations": generations_log,
    }


def save_consolidated_data(
    results_list: List[Dict[str, Any]],
    max_runs: int,
    csv_path: str,
    metrics_path: str,
    data_path: str,
    mode: str = "board",
) -> None:
    """
    Guarda los datos consolidados en disco de forma atómica y ordenada.
    Utiliza archivos temporales .tmp y reemplazo atómico para evitar corrupción de datos.
    Preserva y combina métricas previas si ya existían en disco.
    """
    if not results_list:
        return

    clean_results = [r for r in results_list if r is not None]
    clean_results.sort(key=lambda x: x["idx"])

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    # 1. Guardar CSV ordenado
    csv_tmp = csv_path + ".tmp"
    with open(csv_tmp, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "indice",
            "n",
            "poblacion",
            "mutacion",
            "generacion_encontrada",
            "exito",
            "mejor_fitness",
            "max_fitness",
            "conflictos_finales",
            "tiempo_s",
        ])
        for r in clean_results:
            gen_str = str(r["generacion_encontrada"]) if r["generacion_encontrada"] is not None else ""
            writer.writerow([
                r["idx"],
                r["n"],
                r["poblacion"],
                f"{r['mutacion']:.6f}",
                gen_str,
                r["exito"],
                r["mejor_fitness"],
                r["max_fitness"],
                r["conflictos_finales"],
                r["tiempo_s"],
            ])
    os.replace(csv_tmp, csv_path)

    # 2. Cargar métricas existentes si las hay para combinar sin pérdida
    metrics_dict = {}
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_dict = json.load(f)
        except Exception:
            metrics_dict = {}

    data_dict = {}
    if os.path.isfile(data_path):
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
        except Exception:
            data_dict = {}

    for r in clean_results:
        if mode == "population":
            key = f"n_{r['n']}_pop_{r['poblacion']}_mut_{r['mutacion']:.4f}"
        else:
            key = f"n_{r['n']}_mut_{r['mutacion']:.4f}"

        # Solo actualizar generaciones si la corrida actual contiene trazas
        if key not in metrics_dict or r.get("generations"):
            metrics_dict[key] = {
                "config": {
                    "n": r["n"],
                    "m": r["poblacion"],
                    "runs": max_runs,
                    "mutation_factor": r["mutacion"],
                    "max_fitness": r["max_fitness"],
                },
                "summary": {
                    "optimo_encontrado": r["exito"],
                    "generacion_del_mejor": r["generacion_encontrada"],
                    "mejor_fitness": r["mejor_fitness"],
                    "conflictos_finales": r["conflictos_finales"],
                    "tiempo_total_s": r["tiempo_s"],
                },
                "generations": r["generations"],
            }

        if key not in data_dict or r.get("best_state"):
            data_dict[key] = {
                "n": r["n"],
                "population": r["poblacion"],
                "mutation_factor": r["mutacion"],
                "best_solution": r["best_state"],
                "best_fitness": r["mejor_fitness"],
                "conflicts": r["conflictos_finales"],
                "optimo_encontrado": r["exito"],
                "generacion": r["generacion_encontrada"],
            }

    metrics_tmp = metrics_path + ".tmp"
    with open(metrics_tmp, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    os.replace(metrics_tmp, metrics_path)

    data_tmp = data_path + ".tmp"
    with open(data_tmp, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    os.replace(data_tmp, data_path)


def run_interactive_wizard() -> Dict[str, Any]:
    """
    Asistente guiado interactivo para configurar y ejecutar la experimentación de datos.
    """
    print("\n" + "=" * 72)
    print("      ASISTENTE DE EXPERIMENTACIÓN MULTIDIMENSIONAL (N-REINAS)      ")
    print("=" * 72)
    print("Seleccione la modalidad de experimentación que desea realizar:")
    print("  [1] Variar tamaño de tablero (N de 8 a 64, población fija)")
    print("  [2] Variar tamaño de población (M con N=25 fijo y función de steps)")

    choice = input("\nIngrese opción [1 o 2, por defecto 2]: ").strip()
    mode = "board" if choice == "1" else "population"

    cfg: Dict[str, Any] = {"mode": mode}

    if mode == "population":
        print("\n--- CONFIGURACIÓN: VARIACIÓN DE POBLACIÓN (N FIJO) ---")
        n_in = input("Tamaño de tablero fijo N [default: 25]: ").strip()
        cfg["fixed_n"] = int(n_in) if n_in else 25

        print("\n¿Cómo desea definir los tamaños de población?")
        print("  [1] Rango con función de pasos/steps (m-min, m-max, m-step)")
        print("  [2] Lista explícita de valores (ej: 20 40 60 80 100 150 200)")
        p_type = input("Opción [1 o 2, default: 1]: ").strip()

        if p_type == "2":
            list_in = input("Ingrese poblaciones separadas por espacio [ej: 20 40 60 80 100 150 200]: ").strip()
            if list_in:
                cfg["m_list"] = [int(x) for x in list_in.split()]
            else:
                cfg["m_list"] = [20, 40, 60, 80, 100, 150, 200]
            cfg["m_min"] = None
            cfg["m_max"] = None
            cfg["m_step"] = None
        else:
            min_in = input("Población mínima [default: 20]: ").strip()
            max_in = input("Población máxima [default: 200]: ").strip()
            step_in = input("Paso (step) entre poblaciones [default: 20]: ").strip()
            cfg["m_min"] = int(min_in) if min_in else 20
            cfg["m_max"] = int(max_in) if max_in else 200
            cfg["m_step"] = int(step_in) if step_in else 20
            cfg["m_list"] = None
    else:
        print("\n--- CONFIGURACIÓN: VARIACIÓN DE TABLEROS (M FIJO) ---")
        m_in = input("Población fija m [default: 100]: ").strip()
        cfg["m"] = int(m_in) if m_in else 100

        print("\n¿Cómo desea definir las dimensiones de tablero?")
        print("  [1] Rango con pasos (n-min, n-max, n-step)")
        print("  [2] Lista explícita de dimensiones (ej: 8 12 16 20 24 28 32)")
        n_type = input("Opción [1 o 2, default: 1]: ").strip()
        if n_type == "2":
            list_in = input("Ingrese dimensiones separadas por espacio [ej: 8 12 16 20 24 28 32]: ").strip()
            cfg["n_list"] = [int(x) for x in list_in.split()] if list_in else [8, 12, 16, 20, 24, 28, 32]
            cfg["n_min"] = None
            cfg["n_max"] = None
            cfg["n_step"] = None
        else:
            min_in = input("Dimensión mínima N [default: 8]: ").strip()
            max_in = input("Dimensión máxima N [default: 50]: ").strip()
            step_in = input("Paso (step) [default: 1]: ").strip()
            cfg["n_min"] = int(min_in) if min_in else 8
            cfg["n_max"] = int(max_in) if max_in else 50
            cfg["n_step"] = int(step_in) if step_in else 1
            cfg["n_list"] = None

    print("\n--- PARÁMETROS GENERALES ---")
    pts_in = input("Puntos de mutación continuos a evaluar [default: 20]: ").strip()
    cfg["points"] = int(pts_in) if pts_in else 20

    runs_in = input("Tope de generaciones por corrida [default: 5000]: ").strip()
    cfg["max_runs"] = int(runs_in) if runs_in else 5000

    pat_in = input("Paciencia para parada temprana por estancamiento (patience) [default: 1000, 0 para desactivar]: ").strip()
    if pat_in:
        pat_val = int(pat_in)
        cfg["patience"] = pat_val if pat_val > 0 else None
    else:
        cfg["patience"] = 1000

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generador Autónomo de Datos de Experimentación Multidimensional (N-Reinas)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Modo general
    parser.add_argument(
        "--mode",
        type=str,
        choices=["board", "population"],
        default=None,
        help="Modo de experimentación: 'board' (variar tablero N con m fija) o 'population' (variar m con N=25 fijo).",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Lanza el asistente interactivo en terminal para configurar los parámetros paso a paso.",
    )

    # Parámetros para modo tablero ('board')
    parser.add_argument(
        "--n-list",
        type=int,
        nargs="+",
        default=None,
        help="Lista explícita de dimensiones de tablero (ej: 8 12 16 20 24 28 32 36 40 50)",
    )
    parser.add_argument("--n-min", type=int, default=8, help="Tamaño mínimo de tablero")
    parser.add_argument("--n-max", type=int, default=50, help="Tamaño máximo de tablero")
    parser.add_argument("--n-step", type=int, default=1, help="Paso entre tamaños de tablero si no se pasa --n-list")
    parser.add_argument("--m", type=int, default=100, help="Tamaño de población fija en modo 'board'")

    # Parámetros para modo población ('population')
    parser.add_argument(
        "-n", "--n", "--fixed-n",
        dest="fixed_n",
        type=int,
        default=25,
        help="Tamaño de tablero fijo para el modo 'population' (default: 25)",
    )
    parser.add_argument(
        "--m-min",
        type=int,
        default=20,
        help="Tamaño mínimo de población en modo 'population'",
    )
    parser.add_argument(
        "--m-max",
        type=int,
        default=200,
        help="Tamaño máximo de población en modo 'population'",
    )
    parser.add_argument(
        "--m-step",
        type=int,
        default=20,
        help="Paso (step) entre tamaños de población consecutivos",
    )
    parser.add_argument(
        "--m-list",
        type=int,
        nargs="+",
        default=None,
        help="Lista explícita de tamaños de población (ej: 20 40 60 80 100 150 200)",
    )

    # Parámetros de mutación y ejecución
    parser.add_argument(
        "--points",
        type=int,
        default=20,
        help="Cantidad de valores discretos de mutación evaluados (np.linspace entre min-mut y max-mut). Total corridas = configuraciones * points",
    )
    parser.add_argument("--min-mut", type=float, default=0.001, help="Mutación mínima del rango continuo")
    parser.add_argument("--max-mut", type=float, default=0.20, help="Mutación máxima del rango continuo")
    parser.add_argument("--max-runs", type=int, default=5000, help="Tope de generaciones por corrida")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Generaciones consecutivas sin mejora para parada temprana por estancamiento (ej: 1000).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Omite configuraciones que ya existan completadas en el archivo CSV de salida y continúa.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Número de procesos paralelos (default: CPU count)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base aleatoria")

    # Archivos y directorios
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Directorio de destino para los archivos de datos",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default=None,
        help="Nombre del archivo CSV consolidado (default según modo: results_board.csv o results_population.csv)",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default=None,
        help="Nombre del archivo JSON de métricas (default según modo: multi_metrics_board.json o multi_metrics_population.json)",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default=None,
        help="Nombre del archivo JSON de soluciones (default según modo: multi_data_board.json o multi_data_population.json)",
    )

    args = parser.parse_args()

    # Si se solicitó modo interactivo, invocar asistente
    if args.interactive:
        wizard_cfg = run_interactive_wizard()
        mode = wizard_cfg["mode"]
        if mode == "population":
            args.fixed_n = wizard_cfg.get("fixed_n", 25)
            args.m_list = wizard_cfg.get("m_list")
            args.m_min = wizard_cfg.get("m_min", 20)
            args.m_max = wizard_cfg.get("m_max", 200)
            args.m_step = wizard_cfg.get("m_step", 20)
        else:
            args.m = wizard_cfg.get("m", 100)
            args.n_list = wizard_cfg.get("n_list")
            args.n_min = wizard_cfg.get("n_min", 8)
            args.n_max = wizard_cfg.get("n_max", 50)
            args.n_step = wizard_cfg.get("n_step", 1)
        args.points = wizard_cfg.get("points", 20)
        args.max_runs = wizard_cfg.get("max_runs", 1000)
        args.patience = wizard_cfg.get("patience", 1000)
    else:
        # Detección automática de modo si no se pasa explícitamente --mode
        if args.mode is not None:
            mode = args.mode
        elif args.m_list is not None or "--m-min" in sys.argv or "--m-max" in sys.argv or "--m-step" in sys.argv or "--fixed-n" in sys.argv:
            mode = "population"
        else:
            mode = "board"

    # Multiprocessing en Linux
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    num_workers = args.workers or max(1, os.cpu_count() or 1)

    # Configurar valores de mutación (20 valores continuos por defecto)
    mutation_values = [round(float(x), 6) for x in np.linspace(args.min_mut, args.max_mut, args.points)]

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Asignar nombres diferenciados de archivo según la modalidad
    if mode == "population":
        csv_name = args.csv_name or "results_population.csv"
        metrics_name = args.metrics_name or "multi_metrics_population.json"
        data_name = args.data_name or "multi_data_population.json"

        # Determinar valores de población
        if args.m_list is not None and len(args.m_list) > 0:
            m_values = sorted(list(set(args.m_list)))
        else:
            m_vals = list(range(args.m_min, args.m_max + 1, args.m_step))
            if args.m_max not in m_vals:
                m_vals.append(args.m_max)
            m_values = sorted(m_vals)

        # Validar población mínima requerida para el operador de cruce
        m_values = [int(v) for v in m_values if v >= 4]
        if not m_values:
            print("[ERROR] Los tamaños de población deben ser mayores o iguales a 4.")
            sys.exit(1)

        fixed_n = args.fixed_n
        all_tasks = []
        task_idx = 0
        for m in m_values:
            for mut in mutation_values:
                all_tasks.append((task_idx, fixed_n, m, args.max_runs, mut, args.seed, args.patience))
                task_idx += 1

    else:  # mode == "board"
        csv_name = args.csv_name or "results_board.csv"
        metrics_name = args.metrics_name or "multi_metrics_board.json"
        data_name = args.data_name or "multi_data_board.json"

        if args.n_list is not None and len(args.n_list) > 0:
            n_values = sorted(list(set(args.n_list)))
        else:
            n_vals = list(range(args.n_min, args.n_max + 1, args.n_step))
            if args.n_max not in n_vals:
                n_vals.append(args.n_max)
            n_values = sorted(n_vals)

        fixed_n = None
        all_tasks = []
        task_idx = 0
        for n in n_values:
            for mut in mutation_values:
                all_tasks.append((task_idx, n, args.m, args.max_runs, mut, args.seed, args.patience))
                task_idx += 1

    csv_path = os.path.join(out_dir, csv_name)
    metrics_path = os.path.join(out_dir, metrics_name)
    data_path = os.path.join(out_dir, data_name)

    total_tasks = len(all_tasks)
    existing_results: List[Dict[str, Any]] = []
    completed_keys: Set[Tuple[int, int, float]] = set()

    # Si se solicitó --resume, leer registros previos
    if args.resume and os.path.isfile(csv_path):
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    gen_val = r["generacion_encontrada"].strip()
                    n_val = int(r["n"])
                    p_val = int(r["poblacion"])
                    mut_val = round(float(r["mutacion"]), 6)
                    completed_keys.add((n_val, p_val, mut_val))
                    existing_results.append({
                        "idx": int(r["indice"]),
                        "n": n_val,
                        "poblacion": p_val,
                        "mutacion": mut_val,
                        "generacion_encontrada": int(gen_val) if gen_val != "" else None,
                        "exito": r["exito"].lower() in ("true", "1"),
                        "mejor_fitness": float(r["mejor_fitness"]),
                        "max_fitness": int(r["max_fitness"]),
                        "conflictos_finales": int(r["conflictos_finales"]),
                        "tiempo_s": float(r["tiempo_s"]),
                        "best_state": None,
                        "generations": [],
                    })
        except Exception as e:
            print(f"[AVISO] Error al leer archivo CSV para reanudar ({e}). Se iniciará desde cero.")
            completed_keys = set()
            existing_results = []

    tasks = [
        t for t in all_tasks
        if (t[1], t[2], round(t[4], 6)) not in completed_keys
    ]

    already_done = total_tasks - len(tasks)

    print("=" * 72)
    print(f" GENERADOR AUTÓNOMO DE DATOS: EXPERIMENTACIÓN [MODO: {mode.upper()}] ")
    print("=" * 72)
    if mode == "population":
        print(f"  Tamaño de tablero fijo (n): {fixed_n}")
        print(f"  Poblaciones a evaluar (m):  {m_values} ({len(m_values)} tamaños)")
        if args.m_list is None:
            print(f"  Rango y pasos (steps):      min={args.m_min}, max={args.m_max}, step={args.m_step}")
    else:
        print(f"  Tamaños de tablero (n):     {n_values} ({len(n_values)} tableros)")
        print(f"  Población fija (m):         {args.m}")

    print(f"  Puntos de mutación:         {len(mutation_values)} valores en [{min(mutation_values):.4f}, {max(mutation_values):.4f}]")
    print(f"  Tope de generaciones:       {args.max_runs}")
    if args.patience:
        print(f"  Parada temprana (patience): {args.patience} generaciones sin mejora")
    if args.resume:
        print(f"  Modo reanudación (resume):  {already_done} completados previamente, {len(tasks)} pendientes")
    print(f"  Total de experimentos:      {total_tasks} corridas ({len(tasks)} por ejecutar)")
    print(f"  Procesos concurrentes:      {num_workers}")
    print(f"  Directorio de destino:      {out_dir}")
    print(f"  Archivo CSV:                {csv_path}")
    print(f"  Archivo métricas:           {metrics_path}")
    print(f"  Archivo soluciones:         {data_path}")
    print("=" * 72)

    if not tasks:
        print("\n[INFO] Todas las configuraciones ya fueron completadas en el CSV. Nada pendiente.")
        return

    # Si no es resume o el CSV no existe, inicializar con encabezado
    if not (args.resume and os.path.isfile(csv_path)):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "indice",
                "n",
                "poblacion",
                "mutacion",
                "generacion_encontrada",
                "exito",
                "mejor_fitness",
                "max_fitness",
                "conflictos_finales",
                "tiempo_s",
            ])

    completed = 0
    success_count = sum(1 for r in existing_results if r["exito"])
    all_results: List[Optional[Dict[str, Any]]] = list(existing_results)
    start_time = time.perf_counter()
    interrupted = False
    tasks_to_run_count = len(tasks)

    try:
        chunksize = max(1, tasks_to_run_count // (num_workers * 4))
        with mp.Pool(processes=num_workers) as pool:
            for res in pool.imap_unordered(run_trial, tasks, chunksize=chunksize):
                all_results.append(res)

                completed += 1
                if res["exito"]:
                    success_count += 1

                # Escritura progresiva en CSV para persistencia inmediata
                with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    gen_str = str(res["generacion_encontrada"]) if res["generacion_encontrada"] is not None else ""
                    writer.writerow([
                        res["idx"],
                        res["n"],
                        res["poblacion"],
                        f"{res['mutacion']:.6f}",
                        gen_str,
                        res["exito"],
                        res["mejor_fitness"],
                        res["max_fitness"],
                        res["conflictos_finales"],
                        res["tiempo_s"],
                    ])

                # Reporte periódico en consola
                step = max(1, tasks_to_run_count // 20)
                if completed % step == 0 or completed == tasks_to_run_count:
                    elapsed = time.perf_counter() - start_time
                    pct = (completed / tasks_to_run_count) * 100.0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (tasks_to_run_count - completed) / rate if rate > 0 else 0
                    current_success_pct = (success_count / (already_done + completed)) * 100.0
                    print(
                        f"[{completed:4d}/{tasks_to_run_count:4d}] {pct:5.1f}% | "
                        f"Éxitos totales: {success_count}/{already_done + completed} ({current_success_pct:5.1f}%) | "
                        f"Velocidad: {rate:5.1f} sim/s | "
                        f"ETA: {eta:4.1f}s"
                    )

    except KeyboardInterrupt:
        print("\n\n[AVISO] Proceso interrumpido por el usuario. Guardando datos completados hasta el momento...")
        interrupted = True
    finally:
        total_elapsed = time.perf_counter() - start_time
        valid_results = [r for r in all_results if r is not None]

        # Consolidar y ordenar archivos atómicamente
        save_consolidated_data(
            results_list=valid_results,
            max_runs=args.max_runs,
            csv_path=csv_path,
            metrics_path=metrics_path,
            data_path=data_path,
            mode=mode,
        )

        status_header = "EJECUCIÓN INTERRUMPIDA (DATOS SALVADOS)" if interrupted else "GENERACIÓN DE DATOS FINALIZADA"
        print("\n" + "=" * 72)
        print(f" {status_header} ")
        print("=" * 72)
        print(f"  Corridas completadas en esta sesión: {completed} / {tasks_to_run_count}")
        print(f"  Corridas totales consolidadas:       {len(valid_results)} / {total_tasks}")
        if valid_results:
            exitos = sum(1 for r in valid_results if r["exito"])
            print(f"  Soluciones óptimas totales:          {exitos} ({exitos / len(valid_results) * 100:.1f}%)")
        print(f"  Tiempo de sesión:                    {total_elapsed:.2f} s ({completed / total_elapsed if total_elapsed > 0 else 0:.1f} sim/s)")
        print(f"  CSV ordenado:                        {csv_path}")
        print(f"  Métricas generacionales:             {metrics_path}")
        print(f"  Soluciones guardadas:                {data_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()

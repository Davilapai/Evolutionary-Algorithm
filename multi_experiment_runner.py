#!/usr/bin/env python3
"""
multi_experiment_runner.py - Generador autónomo de datos para múltiples tamaños de tablero y mutaciones.

Ejecuta el algoritmo genético para diferentes tamaños de tablero (ej: N de 8 a 64)
y factores de mutación continuos (ej: 20 valores entre 0.001 y 0.20), diseñado como
un motor de simulación puro, de alto rendimiento y sin dependencias de graficación.

Genera de forma atómica y segura en el directorio de destino (por defecto data/):
  - results.csv: Tabla consolidada con todas las corridas y sus resultados.
  - multi_metrics.json: Trazas generacionales de fitness (mejor, promedio, colisiones).
  - multi_data.json: Estado cromosómico de las mejores soluciones alcanzadas.

Uso:
    Desde la raíz del proyecto, ejecuta una experimentación con los valores
    predeterminados:

        python3 multi_experiment_runner.py

    Para definir tamaños de tablero concretos y ajustar los parámetros:

        python3 multi_experiment_runner.py --n-list 8 12 16 20 --m 200 --max-runs 500

    También puedes usar un rango con --n-min, --n-max y --n-step. Consulta todas
    las opciones disponibles con:

        python3 multi_experiment_runner.py --help

Los archivos generados se guardan en data/ de forma predeterminada. Usa
--out-dir para cambiar el directorio de salida.
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
from typing import Any, Dict, List, Optional, Tuple

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


def run_trial(args_tuple: Tuple[int, int, int, int, float, Optional[int]]) -> Dict[str, Any]:
    """
    Ejecuta una corrida del algoritmo genético y registra la evolución generacional del fitness.
    """
    idx, n, m, max_runs, mutation_factor, seed = args_tuple

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
    m: int,
    max_runs: int,
    csv_path: str,
    metrics_path: str,
    data_path: str,
) -> None:
    """
    Guarda los datos consolidados en disco de forma atómica y ordenada.
    Utiliza archivos temporales .tmp y reemplazo atómico para evitar corrupción de datos.
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
                f"{r['mutacion']:.6f}",
                gen_str,
                r["exito"],
                r["mejor_fitness"],
                r["max_fitness"],
                r["conflictos_finales"],
                r["tiempo_s"],
            ])
    os.replace(csv_tmp, csv_path)

    # 2. Guardar métricas generacionales en JSON
    metrics_dict = {}
    data_dict = {}
    for r in clean_results:
        key = f"n_{r['n']}_mut_{r['mutacion']:.4f}"
        metrics_dict[key] = {
            "config": {
                "n": r["n"],
                "m": m,
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
        data_dict[key] = {
            "n": r["n"],
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


def main():
    parser = argparse.ArgumentParser(
        description="Generador Autónomo de Datos de Experimentación Multidimensional (N-Reinas)"
    )
    parser.add_argument(
        "--n-list",
        type=int,
        nargs="+",
        default=None,
        help="Lista explícita de dimensiones de tablero (ej: 8 12 16 20 24 28 32 36 40 50)",
    )
    parser.add_argument("--n-min", type=int, default=8, help="Tamaño mínimo de tablero (default: 8)")
    parser.add_argument("--n-max", type=int, default=50, help="Tamaño máximo de tablero (default: 50)")
    parser.add_argument("--n-step", type=int, default=1, help="Paso entre tamaños de tablero si no se pasa --n-list (default: 1)")
    parser.add_argument("--points", type=int, default=20, help="Número de valores de mutación (default: 20)")
    parser.add_argument("--min-mut", type=float, default=0.001, help="Mutación mínima (default: 0.001)")
    parser.add_argument("--max-mut", type=float, default=0.20, help="Mutación máxima (default: 0.20)")
    parser.add_argument("--m", type=int, default=100, help="Tamaño de la población (default: 100)")
    parser.add_argument("--max-runs", type=int, default=1000, help="Tope de generaciones por corrida (default: 1000)")
    parser.add_argument("--workers", type=int, default=None, help="Número de procesos paralelos (default: CPU count)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base aleatoria (default: 42)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Directorio de destino para los archivos de datos (default: ./data)",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="results.csv",
        help="Nombre del archivo CSV consolidado (default: results.csv)",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default="multi_metrics.json",
        help="Nombre del archivo JSON de métricas generacionales (default: multi_metrics.json)",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="multi_data.json",
        help="Nombre del archivo JSON de estados y soluciones (default: multi_data.json)",
    )
    args = parser.parse_args()

    # Multiprocessing en Linux
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    num_workers = args.workers or max(1, os.cpu_count() or 1)

    # Determinar tamaños de tablero a evaluar: rango continuo inclusivo [n_min, n_max]
    if args.n_list is not None and len(args.n_list) > 0:
        n_values = sorted(list(set(args.n_list)))
    else:
        n_vals = list(range(args.n_min, args.n_max + 1, args.n_step))
        if args.n_max not in n_vals:
            n_vals.append(args.n_max)
        n_values = sorted(n_vals)

    # Valores equidistantes de factor de mutación
    mutation_values = [round(float(x), 6) for x in np.linspace(args.min_mut, args.max_mut, args.points)]

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, args.csv_name)
    metrics_path = os.path.join(out_dir, args.metrics_name)
    data_path = os.path.join(out_dir, args.data_name)

    # Preparar tareas
    tasks = []
    task_idx = 0
    for n in n_values:
        for mut in mutation_values:
            tasks.append((task_idx, n, args.m, args.max_runs, mut, args.seed))
            task_idx += 1

    total_tasks = len(tasks)

    print("=" * 72)
    print(" GENERADOR AUTÓNOMO DE DATOS: EXPERIMENTACIÓN MULTIDIMENSIONAL ")
    print("=" * 72)
    print(f"  Tamaños de tablero (n):   {n_values} ({len(n_values)} tableros)")
    print(f"  Puntos de mutación:       {len(mutation_values)} valores en [{min(mutation_values):.4f}, {max(mutation_values):.4f}]")
    print(f"  Población (m):            {args.m}")
    print(f"  Tope de generaciones:     {args.max_runs}")
    print(f"  Total de experimentos:    {total_tasks} corridas")
    print(f"  Procesos concurrentes:    {num_workers}")
    print(f"  Directorio de destino:    {out_dir}")
    print(f"  Archivo CSV:              {csv_path}")
    print(f"  Archivo métricas:         {metrics_path}")
    print(f"  Archivo soluciones:       {data_path}")
    print("=" * 72)

    # Inicializar CSV progresivo con encabezado
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "indice",
            "n",
            "mutacion",
            "generacion_encontrada",
            "exito",
            "mejor_fitness",
            "max_fitness",
            "conflictos_finales",
            "tiempo_s",
        ])

    completed = 0
    success_count = 0
    all_results = [None] * total_tasks
    start_time = time.perf_counter()
    interrupted = False

    try:
        chunksize = max(1, total_tasks // (num_workers * 4))
        with mp.Pool(processes=num_workers) as pool:
            for res in pool.imap_unordered(run_trial, tasks, chunksize=chunksize):
                idx = res["idx"]
                all_results[idx] = res

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
                        f"{res['mutacion']:.6f}",
                        gen_str,
                        res["exito"],
                        res["mejor_fitness"],
                        res["max_fitness"],
                        res["conflictos_finales"],
                        res["tiempo_s"],
                    ])

                # Reporte periódico en consola
                step = max(1, total_tasks // 20)
                if completed % step == 0 or completed == total_tasks:
                    elapsed = time.perf_counter() - start_time
                    pct = (completed / total_tasks) * 100.0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_tasks - completed) / rate if rate > 0 else 0
                    success_pct = (success_count / completed) * 100.0
                    print(
                        f"[{completed:4d}/{total_tasks:4d}] {pct:5.1f}% | "
                        f"Éxitos: {success_count}/{completed} ({success_pct:5.1f}%) | "
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
            m=args.m,
            max_runs=args.max_runs,
            csv_path=csv_path,
            metrics_path=metrics_path,
            data_path=data_path,
        )

        status_header = "EJECUCIÓN INTERRUMPIDA (DATOS SALVADOS)" if interrupted else "GENERACIÓN DE DATOS FINALIZADA"
        print("\n" + "=" * 72)
        print(f" {status_header} ")
        print("=" * 72)
        print(f"  Corridas completadas:   {len(valid_results)} / {total_tasks}")
        if valid_results:
            exitos = sum(1 for r in valid_results if r["exito"])
            print(f"  Soluciones óptimas:     {exitos} ({exitos / len(valid_results) * 100:.1f}%)")
        print(f"  Tiempo total:           {total_elapsed:.2f} s ({len(valid_results) / total_elapsed if total_elapsed > 0 else 0:.1f} sim/s)")
        print(f"  CSV ordenado:           {csv_path}")
        print(f"  Métricas generacionales:{metrics_path}")
        print(f"  Soluciones guardadas:   {data_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()

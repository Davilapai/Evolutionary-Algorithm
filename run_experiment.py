
    python run_experiment.py

Equivale a:

    python run_experiment.py --n 100 --m 300 --runs 300 --mutation 0.007

Otras opciones:
    --seed 42       reproducibilidad (misma semilla -> misma corrida)
    --tag exp1      sufijo para no sobrescribir corridas anteriores
    --no-plots      solo métricas, sin figuras
"""

import argparse
import os
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "analysis"))

from runner import run_instrumented  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Experimento AG - N Reinas (sin tocar el código del taller)")
    p.add_argument("--n", type=int, default=100, help="tamaño del tablero")
    p.add_argument("--m", type=int, default=300, help="tamaño de la población")
    p.add_argument("--runs", type=int, default=300, help="número de generaciones (tope; puede terminar antes si encuentra el óptimo)")
    p.add_argument("--mutation", type=float, default=0.007, help="factor de mutación")
    p.add_argument("--seed", type=int, default=None, help="semilla aleatoria")
    p.add_argument("--tag", type=str, default="", help="sufijo para los archivos de salida")
    p.add_argument("--no-plots", action="store_true", help="no generar figuras")
    p.add_argument("--quiet", action="store_true", help="no imprimir cada generación")
    args = p.parse_args()

    sufijo = f"_{args.tag}" if args.tag else ""
    metrics_path = os.path.join(RAIZ, "data", f"metrics{sufijo}.json")
    figdir = os.path.join(RAIZ, "figures", args.tag) if args.tag else os.path.join(RAIZ, "figures")

    print(f"n={args.n}  m={args.m}  generaciones={args.runs}  "
          f"mutación={args.mutation}  semilla={args.seed}\n")

    recorder = run_instrumented(RAIZ, args.n, args.m, args.runs, args.mutation,
                                seed=args.seed, quiet=args.quiet)
    recorder.save(metrics_path)

    s = recorder.summary()
    print("\n===== RESUMEN =====")
    for k, v in s.items():
        print(f"{k:38s}: {v}")
    print(f"\nMétricas guardadas en: {metrics_path}")

    if not args.no_plots:
        sys.path.insert(0, os.path.join(RAIZ, "analysis"))
        from plots import generar_todo
        generar_todo(metrics_path, figdir)


if __name__ == "__main__":
    main()

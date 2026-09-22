"""
Uso:
    python run_scalability.py
    python run_scalability.py --n-values 20,50,100,150,200 --seed 42
"""

import argparse
import os
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "analysis"))

from scalability import run_sweep  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Barrido de escalabilidad - N Reinas")
    p.add_argument("--n-values", type=str, default="20,50,100,150,200",
                   help="lista de tamaños de tablero separados por coma")
    p.add_argument("--m", type=int, default=300, help="tamaño de la población (fijo para todas las corridas)")
    p.add_argument("--mutation", type=float, default=0.007, help="factor de mutación (fijo para todas las corridas)")
    p.add_argument("--runs", type=int, default=300, help="tope de generaciones por corrida")
    p.add_argument("--seed", type=int, default=None, help="semilla aleatoria (misma para cada n)")
    p.add_argument("--tag", type=str, default="", help="sufijo para no sobrescribir barridos anteriores")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    n_values = [int(v) for v in args.n_values.split(",") if v.strip()]
    sufijo = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(RAIZ, "data", f"scalability{sufijo}.json")
    figdir = os.path.join(RAIZ, "figures", args.tag) if args.tag else os.path.join(RAIZ, "figures")

    print(f"Tamaños a probar: {n_values}  (m={args.m}, mutación={args.mutation}, "
          f"tope de generaciones={args.runs})\n")

    run_sweep(RAIZ, n_values, m=args.m, mutation=args.mutation, max_runs=args.runs,
             seed=args.seed, quiet=True, out_path=out_path)

    if not args.no_plots:
        from plots import generar_escalabilidad
        generar_escalabilidad(out_path, figdir)


if __name__ == "__main__":
    main()

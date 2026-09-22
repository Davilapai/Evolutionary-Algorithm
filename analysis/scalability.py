
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from runner import run_instrumented


def run_sweep(raiz, n_values, m=300, mutation=0.007, max_runs=300,
             seed=None, quiet=True, out_path="data/scalability.json"):
    """
    Corre una instancia por cada n en n_values (mismo m y mutación para
    todas, así la comparación es justa) y guarda un resumen por n.
    """
    resultados = []
    for n in n_values:
        print(f"--- n={n} (m={m}, mutación={mutation}, hasta {max_runs} generaciones) ---")
        rec = run_instrumented(raiz, n, m, max_runs, mutation, seed=seed, quiet=quiet)
        s = rec.summary()
        fila = {
            "n": n,
            "m": m,
            "mutation_factor": mutation,
            "runs_configurados": max_runs,
            "tiempo_total_s": s["tiempo_total_s"],
            "generaciones_registradas": s["generaciones_registradas"],
            "generacion_del_mejor": s["generacion_del_mejor"],
            "optimo_encontrado": s["optimo_encontrado"],
            "best_conflicts": s["best_conflicts"],
            "evaluaciones_totales": s["evaluaciones_totales"],
            "tiempo_por_evaluacion_ms": (
                1000 * s["tiempo_total_s"] / s["evaluaciones_totales"]
                if s["evaluaciones_totales"] else None),
        }
        resultados.append(fila)
        print(f"   tiempo={s['tiempo_total_s']:.2f}s  "
              f"óptimo={s['optimo_encontrado']}  "
              f"gen_mejor={s['generacion_del_mejor']}  "
              f"generaciones_corridas={s['generaciones_registradas']}")

    path = os.path.join(raiz, out_path) if not os.path.isabs(out_path) else out_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "m": m,
            "mutation_factor": mutation,
            "runs_configurados": max_runs,
            "seed": seed,
            "resultados": resultados,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {path}")
    return path

"""
Gráficas del algoritmo genético para N-Reinas a partir de `data/metrics.json`.


Uso:
    python analysis/plots.py                      # usa data/metrics.json
    python analysis/plots.py otra/ruta.json figuras/
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import load_metrics, conflict_positions

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

C_BEST, C_MEAN, C_WORST = "#1b4965", "#e07a5f", "#9aa0a6"


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  - {name}.pdf / .png")


def _series(gens, key):
    return np.array([g[key] for g in gens], dtype=float)


def fig_convergencia(data, outdir):
    gens = data["generations"]
    x = _series(gens, "generation")
    best, mean, worst = _series(gens, "best_fitness"), _series(gens, "mean_fitness"), _series(gens, "worst_fitness")
    std = _series(gens, "std_fitness")
    mx = data["config"]["max_fitness"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, best, color=C_BEST, lw=2, label="Mejor")
    ax.plot(x, mean, color=C_MEAN, lw=1.6, label="Promedio")
    ax.fill_between(x, mean - std, mean + std, color=C_MEAN, alpha=0.18,
                    label="Promedio ± 1σ")
    ax.plot(x, worst, color=C_WORST, lw=1, ls="--", label="Peor")
    ax.axhline(mx, color="black", lw=1, ls=":", label=f"Óptimo ({mx})")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Convergencia del fitness (n={data['config']['n']}, "
                 f"m={data['config']['m']}, pm={data['config']['mutation_factor']})")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, outdir, "fig1_convergencia")


def fig_conflictos(data, outdir):
    gens = data["generations"]
    x = _series(gens, "generation")
    bc = _series(gens, "best_conflicts")
    mc = _series(gens, "mean_conflicts")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, mc, color=C_MEAN, lw=1.6, label="Promedio de la población")
    ax.plot(x, bc, color=C_BEST, lw=2, label="Mejor individuo")
    if np.all(bc > 0):
        ax.set_yscale("log")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Conflictos diagonales (menos es mejor)")
    ax.set_title("Reducción de conflictos por generación")
    ax.legend(frameon=False)
    _save(fig, outdir, "fig2_conflictos")


def fig_diversidad(data, outdir):
    gens = data["generations"]
    x = _series(gens, "generation")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, _series(gens, "entropy"), color=C_BEST, lw=2,
            label="Entropía promedio por columna (bits)")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Entropía (bits)")

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(x, _series(gens, "mean_hamming"), color=C_MEAN, lw=1.6,
             label="Distancia de Hamming promedio")
    ax2.plot(x, _series(gens, "unique_ratio"), color=C_WORST, lw=1.4, ls="--",
             label="Proporción de individuos únicos")
    ax2.set_ylabel("Proporción (0–1)")
    ax2.set_ylim(0, 1.05)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", frameon=False)
    ax.set_title("Diversidad de la población")
    _save(fig, outdir, "fig3_diversidad")


def fig_presion_selectiva(data, outdir):
    gens = data["generations"]
    x = _series(gens, "generation")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.plot(x, _series(gens, "selection_pressure"), color=C_BEST, lw=1.8)
    a1.set_ylabel("Mejor / promedio")
    a1.set_title("Presión selectiva y dispersión del fitness")
    a2.plot(x, _series(gens, "std_fitness"), color=C_MEAN, lw=1.8)
    a2.set_ylabel("σ del fitness")
    a2.set_xlabel("Generación")
    _save(fig, outdir, "fig4_presion_selectiva")


def fig_tablero(data, outdir):
    state = data["best_individual"]["state"]
    n = len(state)
    malas = set(conflict_positions(state))
    cols = np.arange(n)
    rows = np.array(state)
    ok = np.array([c not in malas for c in cols])

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    if n <= 40:
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#dddddd", lw=0.6)
    s = max(4, 1200 / n)
    ax.scatter(cols[ok], rows[ok], s=s, color=C_BEST, label="Sin conflicto")
    if (~ok).any():
        ax.scatter(cols[~ok], rows[~ok], s=s, color="#c1121f", marker="X",
                   label="En conflicto diagonal")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    ax.set_title(f"Mejor solución (gen {data['best_individual']['generation']}), "
                 f"{data['best_individual']['conflicts']} conflictos")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    _save(fig, outdir, "fig5_tablero")


def fig_tiempos(data, outdir):
    gens = data["generations"]
    x = _series(gens, "generation")
    t = _series(gens, "approx_time_s")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(x, t, color=C_WORST, lw=1, alpha=0.6, label="Por generación (aprox.)")
    if len(t) > 10:
        k = max(1, len(t) // 40)
        suav = np.convolve(t, np.ones(k) / k, mode="valid")
        ax.plot(x[:len(suav)], suav, color=C_BEST, lw=2, label=f"Media móvil ({k})")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Segundos")
    ax.set_title(f"Tiempo por generación (total: {data['summary']['tiempo_total_s']:.1f} s)")
    ax.legend(frameon=False)
    _save(fig, outdir, "fig6_tiempos")


def tabla_resumen(data, outdir):
    s, c = data["summary"], data["config"]
    filas = [
        ("Tamaño del tablero ($n$)", c["n"]),
        ("Tamaño de población ($m$)", c["m"]),
        ("Generaciones configuradas", c["runs"]),
        ("Generaciones registradas", s["generaciones_registradas"]),
        ("Factor de mutación", c["mutation_factor"]),
        ("Fitness máximo posible", c["max_fitness"]),
        ("Mejor fitness alcanzado", int(s["best_fitness"])),
        ("Conflictos del mejor", s["best_conflicts"]),
        ("Generación del mejor", s["generacion_del_mejor"]),
        ("Fitness promedio inicial", f"{s['fitness_promedio_inicial']:.2f}"),
        ("Fitness promedio final", f"{s['fitness_promedio_final']:.2f}"),
        ("Entropía inicial / final", f"{s['diversidad_inicial_entropy']:.2f} / {s['diversidad_final_entropy']:.2f}"),
        ("Evaluaciones totales", s["evaluaciones_totales"]),
        ("Tiempo total (s)", f"{s['tiempo_total_s']:.1f}"),
    ]
    os.makedirs(outdir, exist_ok=True)
    ruta = os.path.join(outdir, "tabla_resumen.tex")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\small\n")
        f.write("\\caption{Resumen de la corrida del algoritmo genético.}\n")
        f.write("\\label{tab:resumen_ag}\n\\begin{tabular}{lr}\n\\hline\n")
        f.write("\\textbf{Métrica} & \\textbf{Valor} \\\\\n\\hline\n")
        for k, v in filas:
            f.write(f"{k} & {v} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  - tabla_resumen.tex")
    return ruta


def cargar_escalabilidad(path="data/scalability.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_escalabilidad(data, outdir):
    """
    Cómo cambian el tiempo total, las evaluaciones y la velocidad de
    convergencia al crecer n (tamaño del tablero), con m y mutación fijos.

    El tiempo absoluto depende de la máquina donde se corra; lo que sí es
    comparable entre máquinas es la FORMA de la curva (si crece lineal,
    cuadrático, etc.) y cuántas generaciones hacen falta para converger.
    """
    res = sorted(data["resultados"], key=lambda r: r["n"])
    n = np.array([r["n"] for r in res], dtype=float)
    tiempo = np.array([r["tiempo_total_s"] for r in res], dtype=float)
    evals = np.array([r["evaluaciones_totales"] for r in res], dtype=float)
    gen_mejor = np.array([r["generacion_del_mejor"] if r["generacion_del_mejor"] is not None
                          else np.nan for r in res], dtype=float)
    optimo = [r["optimo_encontrado"] for r in res]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3.6))

    a1.plot(n, tiempo, "o-", color=C_BEST)
    a1.set_xlabel("n (tamaño del tablero)")
    a1.set_ylabel("Tiempo total (s)")
    a1.set_title("Tiempo total")

    a2.plot(n, evals, "o-", color=C_MEAN)
    a2.set_xlabel("n (tamaño del tablero)")
    a2.set_ylabel("Evaluaciones totales")
    a2.set_title("Costo computacional")

    colors = [C_BEST if o else C_WORST for o in optimo]
    a3.scatter(n, gen_mejor, c=colors)
    a3.plot(n, gen_mejor, "-", color=C_MEAN, lw=1, alpha=0.5, zorder=0)
    a3.set_xlabel("n (tamaño del tablero)")
    a3.set_ylabel("Generación del mejor")
    a3.set_title("Velocidad de convergencia")

    m = data["m"]
    fig.suptitle(f"Escalabilidad con n (m={m}, mutación={data['mutation_factor']})",
                y=1.05)
    _save(fig, outdir, "fig7_escalabilidad")


def generar_escalabilidad(scalability_path="data/scalability.json", outdir="figures"):
    data = cargar_escalabilidad(scalability_path)
    print(f"Generando figura de escalabilidad en {outdir}/")
    fig_escalabilidad(data, outdir)
    return outdir


def generar_todo(metrics_path="data/metrics.json", outdir="figures"):
    data = load_metrics(metrics_path)
    print(f"Generando figuras en {outdir}/")
    fig_convergencia(data, outdir)
    fig_conflictos(data, outdir)
    fig_diversidad(data, outdir)
    fig_presion_selectiva(data, outdir)
    fig_tablero(data, outdir)
    fig_tiempos(data, outdir)
    tabla_resumen(data, outdir)
    return outdir


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else "data/metrics.json"
    salida = sys.argv[2] if len(sys.argv) > 2 else "figures"
    generar_todo(ruta, salida)

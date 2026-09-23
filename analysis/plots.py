import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from .metrics import load_metrics, conflict_positions

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
    path = os.path.join(outdir, f"{name}.jpg")
    fig.savefig(path, format="jpg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  - {name}.jpg")


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

    fig, ax = plt.subplots(figsize=(6, 6))

    # 1. Fondo de casillas alternadas (estilo ajedrez clásico o madera suave)
    # Matriz donde cada celda alterna entre 0 y 1
    board = (np.indices((n, n)).sum(axis=0) % 2)
    
    # Colores suaves para que no compitan con los puntos de las reinas
    # Casilla clara / Casilla oscura (puedes ajustar a tonos madera o gris)
    cmap_board = ListedColormap(["#f0d9b5", "#b58863"])
    
    # extent alinea las celdas exactamente con las coordenadas de scatter
    ax.imshow(
        board,
        cmap=cmap_board,
        origin="lower",
        extent=(-0.5, n - 0.5, -0.5, n - 0.5),
        zorder=0
    )

    # 2. Dibujar una reina en cada posición del mejor individuo.
    # Reducir el símbolo proporcionalmente para que quepa dentro de una casilla.
    queen_size = max(4, min(18, 320 / n))
    for col, row, is_ok in zip(cols, rows, ok):
        color = C_BEST if is_ok else "#c1121f"
        ax.text(
            col,
            row,
            "♛",
            ha="center",
            va="center",
            fontsize=queen_size,
            color=color,
            fontweight="bold",
            zorder=3,
        )

    # Entradas proxy para conservar una leyenda clara al usar texto como marcador.
    ax.scatter([], [], s=70, color=C_BEST, label="Reina sin conflicto")
    if (~ok).any():
        ax.scatter([], [], s=70, color="#c1121f", label="Reina en conflicto diagonal")

    # 3. Ticks y etiquetas (opcional: notación de ajedrez a-h / 1-8 si n <= 8)
    if n <= 16:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        if n == 8:
            # Letras para columnas y 1-8 para filas
            ax.set_xticklabels([chr(ord('a') + i) for i in range(n)])
            ax.set_yticklabels(range(1, n + 1))
    else:
        # Para n grande, saltos razonables para no saturar
        step = max(1, n // 10)
        ax.set_xticks(np.arange(0, n, step))
        ax.set_yticks(np.arange(0, n, step))

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    ax.set_title(
        f"Mejor solución (n={n}, población={data['config']['m']}, "
        f"gen {data['best_individual']['generation']}), "
        f"{data['best_individual']['conflicts']} conflictos"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.12), frameon=True, fontsize=8)

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

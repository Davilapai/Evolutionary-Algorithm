
import json
import os
import time

import numpy as np


def max_pairs(n: int) -> int:
    """Número máximo de pares de reinas: cota superior del fitness."""
    return n * (n - 1) // 2


def positional_entropy(states: np.ndarray) -> float:
    m, n = states.shape
    if m <= 1:
        return 0.0
    total = 0.0
    for col in range(n):
        _, counts = np.unique(states[:, col], return_counts=True)
        p = counts / m
        total += float(-np.sum(p * np.log2(p)))
    return total / n


def mean_hamming(states: np.ndarray, sample: int = 60, rng=None) -> float:
    m, n = states.shape
    if m <= 1:
        return 0.0
    rng = rng or np.random.default_rng(0)
    k = min(sample, m)
    idx = rng.choice(m, size=k, replace=False)
    sub = states[idx]
    diff = (sub[:, None, :] != sub[None, :, :]).sum(axis=2)
    iu = np.triu_indices(k, k=1)
    return float(diff[iu].mean() / n)


def unique_ratio(states: np.ndarray) -> float:
    m = states.shape[0]
    return len({tuple(row) for row in states.tolist()}) / m


def conflict_positions(state) -> list:
    state = list(state)
    pos, neg = {}, {}
    for col, fil in enumerate(state):
        pos.setdefault(fil + col, []).append(col)
        neg.setdefault(fil - col, []).append(col)
    malas = set()
    for grupo in (*pos.values(), *neg.values()):
        if len(grupo) > 1:
            malas.update(grupo)
    return sorted(malas)


class Recorder:

    def __init__(self, n, m, runs, mutation_factor, seed=None,
                 hamming_sample=60):
        self.config = {
            "n": n,
            "m": m,
            "runs": runs,
            "mutation_factor": mutation_factor,
            "seed": seed,
            "max_fitness": max_pairs(n),
        }
        self.generations = []
        self.best_ever = {"fitness": -1, "generation": None, "state": None}
        self.hamming_sample = hamming_sample
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self._t0 = time.perf_counter()

    def record(self, generation: int, states, evaluation, elapsed: float = 0.0):
        states = np.asarray(states)
        fit = np.asarray(evaluation, dtype=float)
        mp = self.config["max_fitness"]

        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])

        fila = {
            "generation": generation,
            "best_fitness": best_fit,
            "mean_fitness": float(fit.mean()),
            "median_fitness": float(np.median(fit)),
            "worst_fitness": float(fit.min()),
            "std_fitness": float(fit.std()),
            "best_conflicts": int(mp - best_fit),
            "mean_conflicts": float(mp - fit.mean()),
            "selection_pressure": float(best_fit / fit.mean()) if fit.mean() else float("nan"),
            "unique_ratio": unique_ratio(states),
            "entropy": positional_entropy(states),
            "mean_hamming": mean_hamming(states, self.hamming_sample, self._rng),
            "approx_time_s": float(elapsed),
            "elapsed_total_s": time.perf_counter() - self._t0,
        }
        self.generations.append(fila)

        if best_fit > self.best_ever["fitness"]:
            self.best_ever = {
                "fitness": best_fit,
                "conflicts": int(mp - best_fit),
                "generation": generation,
                "state": [int(v) for v in states[best_idx]],
            }
        return fila

    def summary(self):
        gens = self.generations
        mp = self.config["max_fitness"]
        best = self.best_ever
        conv = next((g["generation"] for g in gens
                     if g["best_fitness"] >= best["fitness"]), None)
        estancamiento = (gens[-1]["generation"] - conv) if gens and conv is not None else 0
        return {
            "best_fitness": best["fitness"],
            "best_conflicts": best.get("conflicts"),
            "max_fitness": mp,
            "optimo_encontrado": best.get("conflicts") == 0,
            "generacion_del_mejor": conv,
            "generaciones_registradas": len(gens),
            "generaciones_sin_mejora_final": estancamiento,
            "mejora_total": (gens[-1]["best_fitness"] - gens[0]["best_fitness"]) if gens else 0,
            "fitness_inicial_mejor": gens[0]["best_fitness"] if gens else None,
            "fitness_promedio_inicial": gens[0]["mean_fitness"] if gens else None,
            "fitness_promedio_final": gens[-1]["mean_fitness"] if gens else None,
            "diversidad_inicial_entropy": gens[0]["entropy"] if gens else None,
            "diversidad_final_entropy": gens[-1]["entropy"] if gens else None,
            "tiempo_total_s": time.perf_counter() - self._t0,
            "evaluaciones_totales": self.config["m"] * len(gens),
        }

    def to_dict(self):
        return {
            "config": self.config,
            "summary": self.summary(),
            "best_individual": self.best_ever,
            "generations": self.generations,
        }

    def save(self, path="data/metrics.json"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path


def load_metrics(path="data/metrics.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

import os
import sys
import time
import types

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import Recorder  # noqa: E402

MARCADOR_CLI = 'print("Select the size of the board")'


def _load_simulation(simulation_path, engine_dir, genetic_dir):
    for carpeta in (engine_dir, genetic_dir):
        if carpeta not in sys.path:
            sys.path.insert(0, carpeta)

    with open(simulation_path, encoding="utf-8") as f:
        src = f.read()

    if MARCADOR_CLI in src:
        src = src.split(MARCADOR_CLI, 1)[0]
    else:
        print("Aviso: no encontré el bloque interactivo esperado en "
              "Simulation.py (buscaba: "
              f"{MARCADOR_CLI!r}). Si el archivo cambió de estructura, "
              "puede que este loader necesite ajustarse.")

    modulo = types.ModuleType("_simulacion_taller")
    modulo.__file__ = simulation_path
    exec(compile(src, simulation_path, "exec"), modulo.__dict__)
    return modulo.Simulation


def _attach_recorder(SimulationCls, recorder):
    evaluate_original = SimulationCls.evaluate
    estado = {"gen": -1, "t_prev": time.perf_counter()}

    def evaluate_instrumentado(self):
        fitness = evaluate_original(self)  # comportamiento intacto

        ahora = time.perf_counter()
        estado["gen"] += 1

        estados = np.array([ind.state for ind in self.population])
        fitness_alineado = np.array([ind.fitness() for ind in self.population])

        recorder.record(estado["gen"], estados, fitness_alineado,
                        elapsed=ahora - estado["t_prev"])
        estado["t_prev"] = ahora
        return fitness

    SimulationCls.evaluate = evaluate_instrumentado

    def desenganchar():
        SimulationCls.evaluate = evaluate_original

    return desenganchar


def run_instrumented(raiz, n, m, runs, mutation_factor, seed=None, quiet=False):
    import random

    engine_dir = os.path.join(raiz, "engine")
    genetic_dir = os.path.join(raiz, "genetic_algorithm")
    simulation_path = os.path.join(genetic_dir, "Simulation.py")

    # El código del taller guarda con rutas relativas ("data/data.json"),
    # así que nos paramos en la raíz del repo para que caigan donde deben.
    os.makedirs(os.path.join(raiz, "data"), exist_ok=True)
    cwd_original = os.getcwd()
    os.chdir(raiz)
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))

        Simulation = _load_simulation(simulation_path, engine_dir, genetic_dir)
        recorder = Recorder(n, m, runs, mutation_factor, seed=seed)
        desenganchar = _attach_recorder(Simulation, recorder)

        try:
            if quiet:
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    Simulation.run_simulation(n, m, runs, mutation_factor)
            else:
                Simulation.run_simulation(n, m, runs, mutation_factor)
        finally:
            desenganchar()

        return recorder
    finally:
        os.chdir(cwd_original)

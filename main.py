from pathlib import Path

from genetic_algorithm.Simulation import Simulation
#from analysis.plots import generar_todo

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "data" / "metrics.json"
FIGURES_PATH = ROOT / "figures"

def main():
    n = int(input("Select the size of the board: "))
    m = int(input("Select the size of the population: "))
    runs = int(input("Select the number of runs: "))
    mutation_factor = float(input("Select the mutation factor (0->1): "))
    
    Simulation.run_simulation(n, m, runs, mutation_factor)
    #generar_todo(str(METRICS_PATH), str(FIGURES_PATH))

if __name__ == "__main__":
    main()

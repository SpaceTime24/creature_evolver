"""Using PyGAD to evolve creature-controller weights.

sim.evaluate() takes a batch of flat `genome_size`-length weight vectors and
returns one fitness per genome. So, only the weights of the neural network
evolve, not its structure.

Modify `GA_CONFIG_FILE` to configure the fields of the genetic algorithm.
"""

import json
from pathlib import Path
import time

import pygad
from creature_simulator import SimulationSet

CREATURE_FILE = "spider_multi_leg_joint_mid_limit.json"
ENVIRONMENT_FILE = "sample_environment.json"

GA_CONFIG_FILE = Path("ga_config.json")
TOP_PERFORMERS_TO_SAVE = 10
TOP_PERFORMERS_FILE = Path("top_performers.json")


def save_top_performers(population, fitnesses, generation, count, path, genome_size):
    """Save the best evaluated genomes from the final generation as JSON."""
    if count < 1:
        raise ValueError("count must be at least 1")

    ranked_indices = sorted(
        range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True
    )
    performers = [
        {
            "rank": rank,
            "fitness": float(fitnesses[index]),
            "genome": list(map(float, population[index])),
        }
        for rank, index in enumerate(ranked_indices[:count], start=1)
    ]
    path.write_text(
        json.dumps(
            {
                "generation": generation,
                "creature_config": CREATURE_FILE,
                "environment_config": ENVIRONMENT_FILE,
                "genome_size": genome_size,
                "performers": performers,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {len(performers)} top performers to {path}")


def main():
    sim = SimulationSet.from_config_files(
        CREATURE_FILE,
        ENVIRONMENT_FILE,
        workers=16,
        max_steps=600,
    )

    print("=========================")
    print("Simulation Stats")
    print("-------------------------")
    print(f"genome_size      = {sim.genome_size}")
    print(f"observation_size = {sim.observation_size}")
    print(f"action_size      = {sim.action_size}")
    print("=========================\nRunning...\n")

    def fitness_func(ga_instance, solutions, solutions_indices):
        genomes = [solution.tolist() for solution in solutions]
        fitnesses = sim.evaluate(genomes)
        return fitnesses

    def on_generation(ga_instance):
        fitnesses = ga_instance.last_generation_fitness
        print(
            f"gen {ga_instance.generations_completed}:",
            f"best={fitnesses.max():.3f}",
            f"mean={fitnesses.mean():.3f}",
            f"std={fitnesses.std():.3f}",
        )

    with open(GA_CONFIG_FILE, "r") as file:
        ga_config = json.load(file)

    ga_instance = pygad.GA(
        num_genes=sim.genome_size,
        fitness_func=fitness_func,
        on_generation=on_generation,
        **ga_config,
    )

    start_time = time.perf_counter()
    ga_instance.run()
    seconds_elapsed = time.perf_counter() - start_time

    best_solution, best_fitness, _ = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )

    print("\n=========================\nResults:")
    print("-------------------------")
    print(f"Total run time: {seconds_elapsed:.2f}s ({seconds_elapsed / 60:.2f} min)")
    print("Best genome:\n{!s}".format(best_solution))
    print(f"Best fitness: {best_fitness:.3f}\n")
    save_top_performers(
        population=ga_instance.population,
        fitnesses=ga_instance.last_generation_fitness,
        generation=ga_instance.generations_completed - 1,
        count=TOP_PERFORMERS_TO_SAVE,
        path=TOP_PERFORMERS_FILE,
        genome_size=sim.genome_size,
    )
    print("=========================")


if __name__ == "__main__":
    main()

"""Minimal demonstration of the Python interface.

Builds a simulation set from config files, then runs a tiny random-mutation
"evolution" loop to show the evaluate() contract. Swap the toy loop below for a
real genetic algorithm (DEAP, PyGAD, your own, ...) that operates on flat
`genome_size`-length weight vectors.
"""

import json
import random
from pathlib import Path

from creature_simulator import SimulationSet

# Build the simulations from data-driven config files. `workers` background
# threads run rollouts in parallel; `max_steps` is the rollout length.
sim = SimulationSet.from_config_files(
    "spider_creature_with_limits.json",
    "sample_environment.json",
    workers=16,
    max_steps=400,
)

print(f"genome_size      = {sim.genome_size}")
print(f"observation_size = {sim.observation_size}")
print(f"action_size      = {sim.action_size}")


def random_genome():
    return [random.uniform(-1.0, 1.0) for _ in range(sim.genome_size)]


def mutate(genome, rate=0.1, scale=0.3):
    return [
        g + random.gauss(0.0, scale) if random.random() < rate else g
        for g in genome
    ]


# Uncomment to watch one of the workers in a window while it trains:
# for i in range(1):
#     sim.render(index=i)

POP_SIZE = 100
GENERATIONS = 400
TOP_PERFORMERS_TO_SAVE = 10
TOP_PERFORMERS_FILE = Path("top_performers.json")


def save_top_performers(ranked, generation, count, path):
    """Save the best evaluated genomes from one generation as JSON.

    The stored genomes can be passed directly to ``sim.evaluate`` in a later
    run, provided the same creature/environment configuration is used.
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    performers = [
        {"rank": rank, "fitness": fitness, "genome": genome}
        for rank, (fitness, genome) in enumerate(ranked[:count], start=1)
    ]
    path.write_text(
        json.dumps(
            {
                "generation": generation,
                "creature_config": "spider_creature_with_limits.json",
                "environment_config": "sample_environment.json",
                "genome_size": sim.genome_size,
                "performers": performers,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {len(performers)} top performers to {path}")

population = [random_genome() for _ in range(POP_SIZE)]
last_ranked = []

for generation in range(GENERATIONS):
    # if generation == GENERATIONS - 3:
    #     for i in range(2):
    #         sim.render(index=i)
    # One FFI call evaluates the whole generation in parallel across workers.
    fitnesses = sim.evaluate(population)

    ranked = sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)
    last_ranked = ranked
    best_fitness = ranked[0][0]
    print(f"gen {generation}: best fitness = {best_fitness:.3f}")

    # Elitist reproduction: keep the top few, fill the rest with their mutants.
    survivors = [genome for _, genome in ranked[: POP_SIZE // 4]]
    population = list(survivors)
    while len(population) < POP_SIZE:
        parent = random.choice(survivors)
        population.append(mutate(parent))


save_top_performers(
    last_ranked,
    generation=GENERATIONS - 1,
    count=TOP_PERFORMERS_TO_SAVE,
    path=TOP_PERFORMERS_FILE,
)

input("Enter to stop! ")

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

CREATURE_FILE = "spider_creature_with_limits.json"
ENVIRONMENT_FILE = "sample_environment.json"

# Build the simulations from data-driven config files. `workers` background
# threads run rollouts in parallel; `max_steps` is the rollout length.
sim = SimulationSet.from_config_files(
    CREATURE_FILE,
    ENVIRONMENT_FILE,
    workers=16,
    max_steps=400,
)

print(f"genome_size      = {sim.genome_size}")
print(f"observation_size = {sim.observation_size}")
print(f"action_size      = {sim.action_size}")


def random_genome():
    return [random.uniform(-1.0, 1.0) for _ in range(sim.genome_size)]


def mutate(genome, rate=0.2, scale=0.3):
    return [
        g + random.gauss(0.0, scale) if random.random() < rate else g
        for g in genome
    ]


def crossover(first_parent, second_parent):
    """Create a child by choosing each weight from either parent."""
    return [
        first_gene if random.random() < 0.5 else second_gene
        for first_gene, second_gene in zip(first_parent, second_parent)
    ]


def tournament_select(ranked, tournament_size=3):
    """Choose the fittest genome from a random subset of evaluated genomes."""
    if not ranked:
        raise ValueError("cannot select a parent from an empty population")
    if tournament_size < 1:
        raise ValueError("tournament_size must be at least 1")

    # Sampling with replacement also permits a tournament larger than POP_SIZE.
    tournament = random.choices(ranked, k=tournament_size)
    return max(tournament, key=lambda pair: pair[0])[1]


# Uncomment to watch one of the workers in a window while it trains:
# for i in range(1):
#     sim.render(index=i)

POP_SIZE = 80
GENERATIONS = 150
TOP_PERFORMERS_TO_SAVE = 10
TOP_PERFORMERS_FILE = Path("top_performers.json")
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.8


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
                "creature_config": CREATURE_FILE,
                "environment_config": ENVIRONMENT_FILE,
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
# Keep fitness results for the lifetime of this run.  Survivors are carried into
# later generations unchanged, and crossover can also recreate an existing
# genome, so evaluating only the missing unique genomes avoids duplicate work.
fitness_cache = {}


def evaluate_once(genomes):
    """Return fitnesses while evaluating each distinct genome at most once."""
    missing = []
    seen_missing = set()
    genome_keys = []

    for genome in genomes:
        key = tuple(genome)
        genome_keys.append(key)
        if key not in fitness_cache and key not in seen_missing:
            missing.append(genome)
            seen_missing.add(key)

    if missing:
        missing_fitnesses = sim.evaluate(missing)
        fitness_cache.update(zip((tuple(genome) for genome in missing), missing_fitnesses))

    return [fitness_cache[key] for key in genome_keys]

for generation in range(GENERATIONS):
    # if generation == GENERATIONS - 3:
    #     for i in range(1):
    #         sim.render(index=i)
    # One FFI call evaluates any new genomes in parallel across workers.
    fitnesses = evaluate_once(population)

    ranked = sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)
    last_ranked = ranked
    best_fitness = ranked[0][0]
    print(f"gen {generation}: best fitness = {best_fitness:.3f}")

    # Preserve the strongest genomes, then use tournament selection to favor
    # good parents without always choosing only the same few individuals.
    survivors = [genome for _, genome in ranked[: POP_SIZE // 4]]
    population = list(survivors)
    while len(population) < POP_SIZE:
        first_parent = tournament_select(ranked, TOURNAMENT_SIZE)
        if random.random() < CROSSOVER_RATE:
            second_parent = tournament_select(ranked, TOURNAMENT_SIZE)
            child = crossover(first_parent, second_parent)
        else:
            child = list(first_parent)
        population.append(mutate(child))


save_top_performers(
    last_ranked,
    generation=GENERATIONS - 1,
    count=TOP_PERFORMERS_TO_SAVE,
    path=TOP_PERFORMERS_FILE,
)

input("Enter to stop! ")

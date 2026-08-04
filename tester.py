"""Minimal demonstration of the Python interface.

Builds a simulation set from config files, then runs a tiny random-mutation
"evolution" loop to show the evaluate() contract. Swap the toy loop below for a
real genetic algorithm (DEAP, PyGAD, your own, ...) that operates on flat
`genome_size`-length weight vectors.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CREATURE_FILE = "spider_multi_leg_joint.json"
ENVIRONMENT_FILE = "sample_environment.json"

POP_SIZE = 80
GENERATIONS = 50
TOP_PERFORMERS_TO_SAVE = 10
TOP_PERFORMERS_FILE = Path("top_performers.json")
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.8


@dataclass
class Individual:
    """A genome and the fitness computed for this particular individual."""

    genome: list
    fitness: Optional[float] = None


def mutate(genome, rng, rate=0.2, scale=0.3):
    return [
        gene + rng.gauss(0.0, scale) if rng.random() < rate else gene
        for gene in genome
    ]


def crossover(first_parent, second_parent, rng):
    """Create a child by choosing each weight from either parent."""
    return [
        first_gene if rng.random() < 0.5 else second_gene
        for first_gene, second_gene in zip(first_parent, second_parent)
    ]


def create_offspring(task):
    """Perform one independent crossover/mutation job in a worker process."""
    first_parent, second_parent, should_crossover, seed = task
    rng = random.Random(seed)
    child = (
        crossover(first_parent, second_parent, rng)
        if should_crossover
        else list(first_parent)
    )
    return mutate(child, rng=rng)


def tournament_select(ranked, tournament_size=3):
    """Choose the fittest individual from a random subset."""
    if not ranked:
        raise ValueError("cannot select a parent from an empty population")
    if tournament_size < 1:
        raise ValueError("tournament_size must be at least 1")

    # Sampling with replacement also permits a tournament larger than POP_SIZE.
    tournament = random.choices(ranked, k=tournament_size)
    return max(tournament, key=lambda individual: individual.fitness)


def save_top_performers(ranked, generation, count, path, genome_size):
    """Save the best evaluated genomes from one generation as JSON."""
    if count < 1:
        raise ValueError("count must be at least 1")

    performers = [
        {
            "rank": rank,
            "fitness": individual.fitness,
            "genome": individual.genome,
        }
        for rank, individual in enumerate(ranked[:count], start=1)
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


def load_starting_genomes(path, genome_size):
    """Load genomes from a top-performers archive, without reusing fitness."""
    try:
        archive = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"starting-genomes file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"starting-genomes file is not valid JSON: {path}") from error

    performers = archive.get("performers")
    if not isinstance(performers, list):
        raise ValueError(f"starting-genomes file has no 'performers' list: {path}")

    genomes = []
    for index, performer in enumerate(performers, start=1):
        genome = performer.get("genome") if isinstance(performer, dict) else None
        if not isinstance(genome, list) or len(genome) != genome_size:
            raise ValueError(
                f"performer {index} in {path} has a genome of the wrong size "
                f"(expected {genome_size})"
            )
        # Make a new list and deliberately omit the archive's fitness: it was
        # measured under potentially different world parameters.
        genomes.append(list(genome))

    return genomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--starting-genomes",
        type=Path,
        metavar="TOP_PERFORMERS_JSON",
        help="seed the initial population with genomes from this top-performers archive",
    )
    args = parser.parse_args()

    # Keep the extension import out of worker processes: they only need the
    # pure-Python genetic operators below.
    from creature_simulator import SimulationSet

    # Build the simulations from data-driven config files. `workers` background
    # threads run rollouts in parallel; `max_steps` is the rollout length.
    sim = SimulationSet.from_config_files(
        CREATURE_FILE,
        ENVIRONMENT_FILE,
        workers=16,
        max_steps=700,
    )

    print(f"genome_size      = {sim.genome_size}")
    print(f"observation_size = {sim.observation_size}")
    print(f"action_size      = {sim.action_size}")

    def random_genome():
        return [random.uniform(-1.0, 1.0) for _ in range(sim.genome_size)]

    starting_genomes = (
        load_starting_genomes(args.starting_genomes, sim.genome_size)
        if args.starting_genomes
        else []
    )
    if len(starting_genomes) > POP_SIZE:
        print(
            f"starting-genomes archive has {len(starting_genomes)} genomes; "
            f"using the first {POP_SIZE}"
        )
        starting_genomes = starting_genomes[:POP_SIZE]

    # Loaded genomes get fresh Individuals (and therefore fresh fitness); the
    # remainder of the population is initialized randomly as before.
    population = [Individual(genome) for genome in starting_genomes]
    population.extend(Individual(random_genome()) for _ in range(POP_SIZE - len(population)))
    if args.starting_genomes:
        print(
            f"initialized population with {len(starting_genomes)} loaded genomes and "
            f"{POP_SIZE - len(starting_genomes)} random genomes"
        )
    last_ranked = []

    def evaluate_pending(individuals):
        """Evaluate only individuals whose fitness has not yet been computed."""
        pending = [individual for individual in individuals if individual.fitness is None]
        if pending:
            fitnesses = sim.evaluate([individual.genome for individual in pending])
            for individual, fitness in zip(pending, fitnesses):
                individual.fitness = fitness

    offspring_count = POP_SIZE - POP_SIZE // 4
    genetic_workers = min(os.cpu_count() or 1, offspring_count)
    # Spawn avoids copying the Rust simulation's active worker threads into
    # children. The executor is created once, so process startup is not paid
    # once per generation.
    with ProcessPoolExecutor(
        max_workers=genetic_workers,
        mp_context=mp.get_context("spawn"),
    ) as offspring_pool:
        for generation in range(GENERATIONS):
            # Survivors retain their fitness; only newly created individuals run.
            evaluate_pending(population)
            ranked = sorted(population, key=lambda individual: individual.fitness, reverse=True)
            last_ranked = ranked
            print(f"gen {generation}: best fitness = {ranked[0].fitness:.3f}")

            survivors = ranked[: POP_SIZE // 4]
            tasks = []
            for _ in range(offspring_count):
                first_parent = tournament_select(ranked, TOURNAMENT_SIZE)
                should_crossover = random.random() < CROSSOVER_RATE
                second_parent = (
                    tournament_select(ranked, TOURNAMENT_SIZE)
                    if should_crossover
                    else None
                )
                tasks.append(
                    (
                        first_parent.genome,
                        second_parent.genome if second_parent else None,
                        should_crossover,
                        random.getrandbits(64),
                    )
                )

            # map preserves the task order, so the population is assembled in a
            # stable order even though crossover and mutation run concurrently.
            population = survivors + [
                Individual(genome) for genome in offspring_pool.map(create_offspring, tasks)
            ]

    save_top_performers(
        last_ranked,
        generation=GENERATIONS - 1,
        count=TOP_PERFORMERS_TO_SAVE,
        path=TOP_PERFORMERS_FILE,
        genome_size=sim.genome_size,
    )
    input("Enter to stop! ")


if __name__ == "__main__":
    main()

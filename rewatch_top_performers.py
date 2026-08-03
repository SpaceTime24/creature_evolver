"""Replay saved top-performing genomes with the graphical simulator.

Run this after ``tester.py`` has created ``top_performers.json``:

    python rewatch_top_performers.py

Pass another archive path as the first argument if needed.
"""

import argparse
import json
from pathlib import Path

from creature_simulator import SimulationSet


def load_archive(path):
    with path.open(encoding="utf-8") as archive_file:
        archive = json.load(archive_file)

    required_keys = {"creature_config", "environment_config", "performers"}
    missing_keys = required_keys - archive.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"{path} is missing required field(s): {missing}")
    if not archive["performers"]:
        raise ValueError(f"{path} contains no saved performers")
    return archive


def main():
    parser = argparse.ArgumentParser(description="Replay saved creature genomes")
    parser.add_argument("archive", nargs="?", type=Path, default=Path("top_performers.json"))
    parser.add_argument(
        "--max-steps",
        type=int,
        default=900,
        help="simulation steps per replay (default: 400, matching tester.py)",
    )
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    archive = load_archive(archive_path)
    archive_dir = archive_path.parent
    creature_path = archive_dir / archive["creature_config"]
    environment_path = archive_dir / archive["environment_config"]

    # A single worker guarantees that the window displays the genome being
    # evaluated, rather than a different worker's job.
    sim = SimulationSet.from_config_files(
        str(creature_path),
        str(environment_path),
        workers=1,
        max_steps=args.max_steps,
    )
    saved_genome_size = archive.get("genome_size")
    if saved_genome_size is not None and saved_genome_size != sim.genome_size:
        raise ValueError(
            f"archive expects genomes of length {saved_genome_size}, but the current "
            f"configuration requires {sim.genome_size}"
        )

    sim.render(index=0)
    print(f"replaying {len(archive['performers'])} saved performer(s) from {archive_path}")
    for performer in archive["performers"]:
        rank = performer["rank"]
        saved_fitness = performer["fitness"]
        genome = performer["genome"]
        print(f"\nReplaying rank {rank} (saved fitness {saved_fitness:.3f})...")
        replay_fitness = sim.evaluate([genome])[0]
        print(f"Replay fitness: {replay_fitness:.3f}")
        input("Press Enter to replay the next creature...")


if __name__ == "__main__":
    main()

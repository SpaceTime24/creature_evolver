## Setup

Run these steps once, after cloning the repository:

```bash
# Navigate into the project directory
cd creature_evolver

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install maturin
pip3 install pygad

# Build Backend
cd creature_simulator
bash build.sh
cd ..
```
## Configuration

### Genetic Algorithm

You can configure PyGAD parameters in `ga_config.json`. See the [PyGAD documentation](https://pygad.readthedocs.io/en/latest/) for a full list of available fields, and check `evolve.py` for any additional custom fields used in the Python interface.

### Virtual Environment

Activate your virtual environment before running any code or installing dependencies:

```bash
source .venv/bin/activate
```

### Building the Simulator Backend

Run this any time the backend (Rust) code changes:

```bash
cd creature_simulator
bash build.sh
```

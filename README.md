# MSc Thesis

## Setup

Requires **Python 3.10**. Install the dependencies with:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository structure

* **algorithms**: The implementations for the different BayesOpt algorithms
* **data**: The data for the different runs of the different experiments
* **diffcbed**: Supporting code for the differentiable causal experimental design components
* **graphs**: The graph datastructure written in an object orientated way. It contains functions for fitting data to the graphs, as well as other functions which make use of the dependency structure of it.
* **posterior_model**: The posterior models over graph structures used by the algorithms
* **results**: The results for the different algorithms with different graph data structures
* **scripts**: The scripts that will run the experiments, with per-benchmark variants in **scripts_erdos**, **scripts_dream**, **scripts_gwps**, **scripts_geometry** and **scripts_random**
* **tests**: Tests for the algorithms and graph utilities
* **utils**: The remaining functions for the graphs and the algorithms. Also contains a lot of data manipulation code.

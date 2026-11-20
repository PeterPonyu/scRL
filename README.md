# scRL: Single-Cell Reinforcement Learning

[![PyPI](https://img.shields.io/pypi/v/scrl-fatedecision.svg?color=brightgreen&style=flat)](https://pypi.org/project/scrl-fatedecision/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24271462.svg)](https://doi.org/10.5281/zenodo.24271462)
[![Python Versions](https://img.shields.io/pypi/pyversions/scrl-fatedecision)](https://pypi.org/project/scrl-fatedecision/)
[![Documentation Status](https://readthedocs.org/projects/scrl/badge/?version=latest)](https://scrl.readthedocs.io/en/latest/?badge=latest)
[![Publish to PyPI](https://github.com/PeterPonyu/scRL/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/PeterPonyu/scRL/actions/workflows/publish-pypi.yml)
[![GitHub Release](https://img.shields.io/github/v/release/PeterPonyu/scRL?sort=semver)](https://github.com/PeterPonyu/scRL/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**scRL** (single-cell Reinforcement Learning) is a computational framework that applies reinforcement learning to single-cell RNA sequencing data to study cellular fate decisions. By treating cell differentiation as a sequential decision-making process, scRL is designed to help analyze when and where fate commitments emerge during development and disease.

### Key Features

- **Actor-Critic Architecture**: Uses deep reinforcement learning to model cellular differentiation trajectories
- **Grid-Based Embedding**: Transforms high-dimensional single-cell data into an interpretable 2-D or 3-D grid
- **Fate Decision Detection**: Identifies early decision points that precede overt lineage commitment
- **Multiple Analysis Modes**:
  - **Gene Module**: Analyzes gene expression patterns and regulatory roles
  - **Lineage Module**: Evaluates lineage commitments and differentiation potentials
  - **Trajectory Module**: Simulates and visualizes differentiation pathways
- **Flexible Reward Systems**: Supports both contribution-based and decision-based reward patterns

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 1.13.1
- CUDA support (optional, for GPU acceleration)

Core dependencies (automatically installed):
- anndata >= 0.8.0
- numpy >= 1.23.5
- pandas >= 1.5.2
- scikit-learn >= 1.2.1
- networkx >= 2.8.8
- matplotlib >= 3.6.3
- seaborn >= 0.11.2
- joblib >= 1.2.0
- tqdm >= 4.64.1
- pygam >= 0.8.0

### Install from PyPI

```bash
pip install scrl-fatedecision
```

**Note**: For the examples in this README, you'll also need Scanpy with its Leiden clustering extra:
```bash
pip install "scanpy[leiden]"
```

### Install from Source

```bash
git clone https://github.com/PeterPonyu/scRL.git
cd scRL
pip install -e .
```

## Quick Start

This example follows the [Basics of scRL](https://scrl.readthedocs.io/en/latest/notebook/Basics_on_scRL.html) tutorial. `grids_from_embedding` returns the grid results object `gres`; `project_cluster`, `align_pseudotime`, `d_rewards`, `get_state_value` and `project_back` store their results in `gres` and return `None`.

```python
import scRL
import scanpy as sc

# Load and preprocess your single-cell data
adata = sc.datasets.paul15()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata, n_comps=50)
sc.tl.tsne(adata, perplexity=50)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

# 2-D embedding for the grids, cluster labels, and the latent space for the agent
X = adata.obsm['X_tsne']
clusters = adata.obs['leiden']
X_pca = adata.obsm['X_pca']

# Create grid embedding
gres = scRL.grids_from_embedding(X)

# Project cluster information onto grids
scRL.project_cluster(gres, clusters)

# Align pseudotime from the starting cluster
scRL.align_pseudotime(gres, early_cluster='4')

# Lineage rewards from the starting cluster towards the target clusters
scRL.d_rewards(gres, starts=['4'], ends=['0', '3', '5'], mode='Decision')

# Train the actor-critic agent on these rewards
trainer = scRL.trainer(
    algo='ActorCritic',
    gres=gres,
    reward_type='d',
    reward_mode='Decision',
    starts_prob=False,
    X_latent=X_pca,
    num_episodes=5000
)
returns, values = trainer.train()

# Fate decision values: critic state values on the grids, projected back to cells
scRL.get_state_value(gres, trainer, key='decision_value')
scRL.project_back(gres, 'decision_value')
adata.obs['decision_value'] = gres.embedding['decision_value']
```

The cluster labels come from the tutorial's Leiden clustering, where cluster `'4'` holds the starting cells and `'0'`, `'3'` and `'5'` are the myeloid target clusters. Leiden labels depend on the clustering run, so set `early_cluster`, `starts` and `ends` from the clusters in your own data.

## 3-D grids

The same pipeline runs on a 3-D embedding, for example a three-component UMAP. `grids_from_embedding` keeps the grid points where cells lie and adds one layer of grid points around them, which forms the boundary of the grid. The grid is built the same way for any 3-D embedding.

```python
sc.tl.umap(adata, n_components=3)
X3 = adata.obsm['X_umap']

gres = scRL.grids_from_embedding(X3)                 # 3-D regular grid over the embedding
scRL.project_cluster(gres, clusters)
scRL.align_pseudotime(gres, early_cluster='4')
scRL.d_rewards(gres, starts=['4'], ends=['0', '3', '5'], mode='Decision')
trainer = scRL.trainer('ActorCritic', gres, reward_type='d', reward_mode='Decision',
                       starts_prob=False, X_latent=X_pca, num_episodes=5000)
returns, values = trainer.train()
scRL.get_state_value(gres, trainer, key='decision_value')
scRL.project_back(gres, 'decision_value')
scRL.plot_grids(gres, key='decision_value')          # grid points coloured by decision value
```

In 3-D the agent moves one grid step along one axis at a time, so it has six moves instead of the eight of the 2-D grid. `n` is the number of grid points across the widest axis of the data (30 by default in 3-D), and `radius` sets how far from the cells the grid extends, in grid steps. `scRL.plot_grids` draws the grid, or the cells, coloured by any stored value; `interactive=True` gives a figure you can rotate (needs plotly). As in 2-D, training outcomes vary between random seeds, so look at the return curve and repeat the training with a few seeds. 4-D and 5-D embeddings are also accepted, but this is experimental.

A complete script on the paul15 data is in `examples/grid3d_paul15.py`.

`examples/grid3d_walkthrough.py` runs every mode on a 3-D grid of the same data and writes its figures to `grid3d_walkthrough_output/`. `python examples/grid3d_walkthrough.py --check` reruns its deterministic part on shipped 3-D coordinates and compares the results with `examples/expected/` (test data and expected results: `source/testdata.rst`).

## Exact solution and lineage gate

The grid environment is small and deterministic, so the best return from every grid point can also be computed exactly, without training and, once pseudotime is fixed, without a random element (`align_pseudotime` samples its start points; `seed=0` makes them reproducible). `scRL.exact_values(gres)` stores this value and the best move for the current reward. It solves a relaxed problem in which returning to a grid point does not end the walk, so with rewards that are not negative (lineage rewards, and gene rewards without `punish_keys`) its values are an upper bound on what a policy can collect under the deep environment's rules and a reference for the value landscape; a greedy policy read from it and run under the deep environment's rules reaches the lineage and then ends by revisiting a point, so its return is lower than the value. `scRL.fate_gate` combines one value map per lineage into a ranking: the heaviest lineage of each point and how undecided it is (its weights are not shares of cells), and `scRL.plot_fates` draws them.

```python
for end in ['0', '3']:
    scRL.d_rewards(gres, starts=['4'], ends=[end], mode='Decision')
    scRL.exact_values(gres)
scRL.fate_gate(gres, ['exact_0', 'exact_3'], names=['lineage 0', 'lineage 3'])
scRL.plot_fates(gres)
```

`scRL.fate_probabilities(gres, {'lineage 0': '0', 'lineage 3': '3'})` gives instead, for every grid point, the share of walks on the grid that enter each lineage first within a fixed number of moves (walks that arrive at no listed lineage are counted as 'none'), drawn with `scRL.plot_fates(gres, key_add='fate')`. By default the walk has a mild bias toward later pseudotime (`forward=0.25`) and 5,000 moves to arrive. At `forward=1` and above each move is strongly drawn to later pseudotime, so a walk seldom moves back toward a neighbouring branch; a smaller value lets it, which can keep fate signals of neighbouring lineages at early grid points, and needs more moves.

## Comparing conditions

Two conditions are compared on one grid built from all cells. `scRL.project_conditions` counts each condition's cells at every grid point, `scRL.compare_abundance` tests where their numbers differ and reports, per cluster, how many grid points held by one condition only are expected from sampling alone, `scRL.compare_values(gres, test, reference, ends=[...])` labels each condition's map from its own cells, solves the lineage reward exactly on both maps and compares the values against a label-shuffling null, and `scRL.plot_conditions` draws the result.

```python
scRL.project_conditions(gres, adata.obs['condition'])
scRL.compare_abundance(gres, test='knockout', reference='control')
scRL.plot_conditions(gres)
```

With several libraries per condition pass `samples=` (one library label per cell) to `compare_abundance` and `compare_fates`: the share of each library's cells per lineage (and, when clusters are projected, per cluster) is then compared across libraries, which is where a difference between conditions can be shown with a handful of libraries; `scRL.shares_by_sample` gives the per-library table. A difference in lineage shares is first of all a difference in which cell states are present: read it with `standardise=True` (on the clusters present in every library: the difference after reweighting to one cluster composition, and the difference split into a composition part and a within-cluster part), look for shifts within cell states with `within=` and `stat='mean'`, and give `pairs=` when libraries form blocks.

## Documentation

Comprehensive documentation is available at [https://scrl.readthedocs.io](https://scrl.readthedocs.io/en/latest/)

### Tutorials

- [Basics of scRL](https://scrl.readthedocs.io/en/latest/notebook/Basics_on_scRL.html) - Introduction to core functionality
- [Application 1](https://scrl.readthedocs.io/en/latest/notebook/Application1.html) - Real-world example
- [Application 2](https://scrl.readthedocs.io/en/latest/notebook/Application2.html) - Advanced use cases

### API Reference

- [Grid Embedding](https://scrl.readthedocs.io/en/latest/GridCore.html) - Grid generation and projection functions
- [Environment](https://scrl.readthedocs.io/en/latest/environment.html) - Reward system and RL environment
- [Trainer](https://scrl.readthedocs.io/en/latest/trainer.html) - Training algorithms and configurations
- [Simulator](https://scrl.readthedocs.io/en/latest/simulator.html) - Trajectory simulation tools
- [Trajectory Analysis](https://scrl.readthedocs.io/en/latest/trajectory.html) - Path analysis functions

## How It Works

scRL integrates three key components:

1. **Grid Embedding Generation**: Transforms single-cell data into a structured 2D grid space using boundary scanning algorithms, preserving topological relationships while enabling efficient exploration.

2. **Reinforcement Learning Training**: An Actor-Critic agent learns to navigate the grid environment, guided by rewards based on gene expression patterns or lineage commitments. The critic evaluates the strength of fate decisions at each state, while the actor identifies optimal differentiation routes.

3. **Functional Analysis Modules**:
   - **Gene Module**: Explores how individual genes contribute to or determine cell fate
   - **Lineage Module**: Analyzes commitment to specific cell lineages
   - **Trajectory Module**: Simulates future differentiation paths using an autoencoder trained alongside the RL agent

## Scientific Background

Understanding how cells develop into different types during growth and disease is a central problem in single-cell analysis. scRL frames cell development as a sequential decision-making problem to estimate when cells begin committing to future fates.

### Reported Findings

- Identifies early decision points before cells show obvious lineage commitment
- Evaluated across hematopoiesis, cancer, organogenesis, and perturbation datasets
- Includes analyses of regulatory factors associated with fate decisions

## Citation

If you use scRL in your research, please cite:

```bibtex
@article{fu2025scrl,
  title={scRL: Utilizing Reinforcement Learning to Evaluate Fate Decisions in Single-Cell Data},
  author={Fu, Zeyu and Chen, Chunlin and Wang, Song and Wang, Junping and Chen, Shilei},
  journal={Biology},
  volume={14},
  number={6},
  pages={679},
  year={2025},
  doi={10.3390/biology14060679}
}
```
        
        

## Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/PeterPonyu/scRL/issues)
- **Documentation**: [https://scrl.readthedocs.io](https://scrl.readthedocs.io/en/latest/)
- **Contact**: fuzeyu99@126.com

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/PeterPonyu/scRL/blob/main/LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, questions, or new features.



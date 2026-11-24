# scRL: Single-Cell Reinforcement Learning

[![PyPI](https://img.shields.io/pypi/v/scrl-fatedecision.svg?color=brightgreen&style=flat)](https://pypi.org/project/scrl-fatedecision/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24271462.svg)](https://doi.org/10.5281/zenodo.24271462)
[![Python Versions](https://img.shields.io/pypi/pyversions/scrl-fatedecision)](https://pypi.org/project/scrl-fatedecision/)
[![Documentation Status](https://readthedocs.org/projects/scrl/badge/?version=latest)](https://scrl.readthedocs.io/en/latest/?badge=latest)
[![Publish to PyPI](https://github.com/PeterPonyu/scRL/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/PeterPonyu/scRL/actions/workflows/publish-pypi.yml)
[![GitHub Release](https://img.shields.io/github/v/release/PeterPonyu/scRL?sort=semver)](https://github.com/PeterPonyu/scRL/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**scRL** (single-cell Reinforcement Learning) applies reinforcement learning to single-cell RNA sequencing data to
study cell fate decisions. It treats differentiation as a sequence of decisions on a grid built over the cells, to
study when and where commitment to a fate emerges.

### Key Features

- **Grids over an embedding**: 2-D or 3-D grids (4-D and 5-D experimental) built over the cells
- **Decision and contribution values**: learned by an actor-critic agent, or computed exactly without training
- **Analysis modes**: gene, lineage and trajectory modules
- **Rewards**: contribution-based or decision-based
- **Comparing conditions**: grid-based comparisons, and a test of per-cell readouts with the sample as the unit

## Installation

Requires Python 3.9 or later and PyTorch 1.13.1 or later; a GPU is optional. The other dependencies install with the
package.

```bash
pip install scrl-fatedecision
```

The examples below also use Scanpy with Leiden clustering:

```bash
pip install "scanpy[leiden]"
```

To install from source:

```bash
git clone https://github.com/PeterPonyu/scRL.git
cd scRL
pip install -e .
```

## Quick Start

This example follows the [Basics of scRL](https://scrl.readthedocs.io/en/latest/notebook/Basics_on_scRL.html)
tutorial. `grids_from_embedding` returns a grid object, `gres`; the later steps store their results in it.

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

In this tutorial, Leiden cluster `'4'` holds the starting cells and `'0'`, `'3'` and `'5'` are the myeloid targets.
Cluster labels change between clustering runs, so choose `early_cluster`, `starts` and `ends` from your own data.

## 3-D grids

The same steps run on a 3-D embedding, such as a three-component UMAP. The grid covers the cells and one layer of
boundary points around them.

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

- **Moves and grid size.** In 3-D the agent moves one step along one axis, which gives six moves instead of eight.
  `n` sets the number of grid points across the widest axis (30 by default in 3-D), and `radius` sets how far the grid
  extends beyond the cells.
- **Training.** Results vary between random seeds, so check the return curve and repeat with a few seeds.
- **Plotting.** `scRL.plot_grids` colours the grid or the cells by any stored value; `interactive=True` gives a figure
  you can rotate (it needs plotly).
- **Examples.** `examples/grid3d_paul15.py` is a complete script. `examples/grid3d_walkthrough.py` runs every mode on a
  3-D grid.

## Exact solution and lineage gate

The grid is small and deterministic, so the best return from every grid point can also be computed exactly, without
training.

- **`scRL.exact_values(gres)`** stores these values and the best move for the current reward. It allows a walk to
  revisit grid points, so for non-negative rewards its values are an upper bound on what a trained agent collects.
- **`scRL.exact_trajectories(gres)`** follows the exact values from the start cluster. It gives trajectories for any
  lineage, including lineages far from the start that trained agents often miss.
- **`scRL.fate_gate`** combines one value map per lineage into a ranking of the lineages at each grid point;
  `scRL.plot_fates` draws it.
- **Other views.** `scRL.plot_trajectories`, `scRL.plot_mesh` and `scRL.plot_voxels` draw trajectories and the grid in
  2-D or 3-D.
- **Reproducibility.** `align_pseudotime(..., seed=0)` makes the start points reproducible.

```python
for end in ['0', '3']:
    scRL.d_rewards(gres, starts=['4'], ends=[end], mode='Decision')
    scRL.exact_values(gres)
scRL.fate_gate(gres, ['exact_0', 'exact_3'], names=['lineage 0', 'lineage 3'])
scRL.plot_fates(gres)
```

`scRL.fate_probabilities(gres, {'lineage 0': '0', 'lineage 3': '3'})` gives, for every grid point, the share of random
walks on the grid that reach each lineage first; walks that reach none are counted as 'none'. Draw the result with
`scRL.plot_fates(gres, key_add='fate')`. The walks lean slightly toward later pseudotime (`forward=0.25` by default),
and larger values make them turn back less often.

A grid can also read cells that were not used to build it. `scRL.place_cells(reference, embedding, query)` places new
cells in the reference embedding, and `scRL.project_cells(gres, positions, key='fate')` gives them the grid's values
and flags cells that fall outside the grid. The new cells should come from the same kind of population.

## Comparing conditions

On one grid built from all cells:
- `scRL.project_conditions` counts each condition's cells at every grid point;
- `scRL.compare_abundance` tests where the counts differ;
- `scRL.compare_values` compares the conditions' exact lineage values against a label-shuffling null;
- `scRL.plot_conditions` draws the result.

```python
scRL.project_conditions(gres, adata.obs['condition'])
scRL.compare_abundance(gres, test='knockout', reference='control')
scRL.plot_conditions(gres)
```

With several libraries per condition, pass `samples=` (one library label per cell) to `compare_abundance` and
`compare_fates`; they then compare each library's share of cells per cluster or lineage across libraries, and
`scRL.shares_by_sample` gives the per-library table. In `compare_fates`, `standardise=True`, `within=` and `pairs=`
separate a change in which cell states are present from a shift within states.

### A per-cell readout between conditions

`scRL.compare_readout(values, samples, groups)` compares a per-cell readout, such as a pseudotime, a fate probability
or a signature score, between two conditions with the sample (animal or donor) as the unit. By default it runs a t-test
on the samples' mean readout and reports the difference with its 95% interval:
- the pooled t-test when the conditions have the same number of samples and similar numbers of cells per sample;
- Welch's t-test otherwise.

`scRL.design_effect` shows how strongly cells cluster within samples, and the false-positive rate this implies for a
test that counts cells.

`examples/protocol_example.py` is a worked example on simulated animals. The documentation page
[Protocol: comparing a per-cell readout between conditions](https://scrl.readthedocs.io/en/latest/readout_protocol.html)
walks through it.

## Documentation

Full documentation: [https://scrl.readthedocs.io](https://scrl.readthedocs.io/en/latest/)

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

1. **Grid embedding**: builds a regular grid over a 2-D or 3-D embedding of the cells (4-D and 5-D are experimental).
2. **Reinforcement learning**: an actor-critic agent learns to move on the grid, guided by rewards from gene
   expression or lineage commitment. The critic's state values give the Decision or Contribution
   value of each grid point, and the actor's moves give routes on the grid.
3. **Analysis modules**:
   - **Gene module**: how individual genes contribute to cell fate.
   - **Lineage module**: commitment to specific lineages.
   - **Trajectory module**: simulates differentiation paths with an autoencoder trained alongside the agent.

## Scientific Background

Understanding how cells develop into different types during growth and disease is a central problem in single-cell
analysis. scRL frames development as a sequence of decisions to estimate when cells begin committing to future fates.
It has been applied to haematopoiesis, cancer, organogenesis and perturbation datasets; the original study (see
Citation) describes the framework and these applications.

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

- **Issues**: [GitHub Issues](https://github.com/PeterPonyu/scRL/issues)
- **Documentation**: [https://scrl.readthedocs.io](https://scrl.readthedocs.io/en/latest/)
- **Contact**: fuzeyu99@126.com

## License

MIT License; see the [LICENSE](https://github.com/PeterPonyu/scRL/blob/main/LICENSE) file.

## Contributing

Contributions are welcome: open an issue or a pull request.

# scRL: Single-Cell Reinforcement Learning

[![PyPI](https://img.shields.io/pypi/v/scrl-fatedecision.svg?color=brightgreen&style=flat)](https://pypi.org/project/scrl-fatedecision/)
[![Documentation Status](https://readthedocs.org/projects/scrl/badge/?version=latest)](https://scrl.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**scRL** (single-cell Reinforcement Learning) is an advanced computational framework that leverages reinforcement learning to decode cellular fate decisions from single-cell RNA sequencing data. By treating cell differentiation as a sequential decision-making process, scRL identifies the precise timing and location of fate commitments during development and disease.

### Key Features

- **Actor-Critic Architecture**: Employs deep reinforcement learning to model cellular differentiation trajectories
- **Grid-Based Embedding**: Transforms high-dimensional single-cell data into an interpretable 2D manifold
- **Fate Decision Detection**: Identifies early decision points that precede overt lineage commitment
- **Multiple Analysis Modes**:
  - **Gene Module**: Analyzes gene expression patterns and regulatory roles
  - **Lineage Module**: Evaluates lineage commitments and differentiation potentials
  - **Trajectory Module**: Simulates and visualizes differentiation pathways
- **Flexible Reward Systems**: Supports both contribution-based and decision-based reward patterns
- **State-of-the-Art Performance**: Outperforms 15+ existing trajectory inference methods

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

**Note**: For the examples in this README, you'll also need Scanpy:
```bash
pip install scanpy
```

### Install from Source

```bash
git clone https://github.com/PeterPonyu/scRL.git
cd scRL
pip install -e .
```

## Quick Start

```python
import scRL
import scanpy as sc

# Load and preprocess your single-cell data
adata = sc.datasets.paul15()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=50)
sc.tl.tsne(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

# Extract embedding and cluster information
X = adata.obsm['X_tsne']
clusters = adata.obs['leiden']

# Create grid embedding
gres = scRL.grids_from_embedding(X)

# Project cluster information onto grids
gres = scRL.project_cluster(gres, clusters)

# Align pseudotime
gres = scRL.align_pseudotime(gres, start_cluster='0')

# Train reinforcement learning model
trainer = scRL.trainer(
    algo='ActorCritic',
    gres=gres,
    reward_type='c',
    reward_mode='Decision'
)

# Get state values for fate decisions
scRL.get_state_value(gres, trainer, key='state_value')
```

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

Understanding how cells develop into different types during growth and disease is crucial for advancing medicine, but current methods cannot pinpoint exactly when and where cells make these critical decisions. scRL treats cell development like a strategic decision-making game—just as a chess player learns optimal moves, the system identifies precise moments when cells decide their future fate.

### Key Findings

- Successfully identifies early decision points before cells show obvious lineage commitment
- Outperforms 15+ state-of-the-art trajectory inference methods across five evaluation dimensions
- Reveals previously unknown regulatory factors controlling fate decisions
- Validated on diverse biological systems: hematopoiesis, cancer, organogenesis, and perturbation datasets

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, questions, or new features.



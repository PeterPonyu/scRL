"""Fate decision values on a 3-D grid for the paul15 haematopoiesis data.

Needs scanpy. Run from the repository root:

    python examples/grid3d_paul15.py

The figure grid3d_decision.png (grid structure, grid pseudotime, grid decision value) is written next to
this script.
"""
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # use the scRL package of this repository
import scRL

data_dir = HERE.parent / 'source' / 'notebook' / 'data'
if data_dir.is_dir():
    sc.settings.datasetdir = data_dir          # reuse the copy of paul15 shipped with the tutorials

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

adata = sc.datasets.paul15()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata, n_comps=50, random_state=0)
sc.pp.neighbors(adata, random_state=0)
sc.tl.umap(adata, n_components=3, random_state=0)

X3 = adata.obsm['X_umap']
clusters = adata.obs['paul15_clusters']
X_pca = adata.obsm['X_pca']

gres = scRL.grids_from_embedding(X3)                 # 3-D regular grid over the embedding
scRL.project_cluster(gres, clusters)
scRL.align_pseudotime(gres, early_cluster='7MEP')
scRL.d_rewards(gres, starts=['7MEP'], ends=['1Ery', '2Ery'], mode='Decision')
trainer = scRL.trainer('ActorCritic', gres, reward_type='d', reward_mode='Decision',
                       starts_prob=False, X_latent=X_pca, num_episodes=5000)
returns, values = trainer.train()
print(f'Mean return over the last 500 episodes: {np.mean(returns[-500:]):.2f}')
scRL.get_state_value(gres, trainer, key='decision_value')
scRL.project_back(gres, 'decision_value')

fig = plt.figure(figsize=(17, 5))
axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]
scRL.plot_grids(gres, ax=axes[0], title='Grid structure')
scRL.plot_grids(gres, key='pseudotime', ax=axes[1], title='Grid pseudotime')
scRL.plot_grids(gres, key='decision_value', ax=axes[2], title='Grid decision value')
out = HERE / 'grid3d_decision.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')

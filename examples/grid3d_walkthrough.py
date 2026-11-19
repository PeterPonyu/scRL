"""Every mode of scRL on a 3-D grid, on the paul15 haematopoiesis data.

Needs scanpy (plotly for the interactive figure). Run from the repository root:

    python examples/grid3d_walkthrough.py

It takes a few minutes and writes its figures to ./grid3d_walkthrough_output/. The short example
examples/grid3d_paul15.py is the quick start; this script goes through lineage rewards in both modes, gene
rewards, the exact solution next to a soft actor-critic, a fate map over three lineages, trajectories and the
simulator, two conditions compared on one grid, and a rotating and an interactive view.
"""
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # use the scRL package of this repository
import scRL

data_dir = HERE.parent / 'source' / 'notebook' / 'data'
if data_dir.is_dir():
    sc.settings.datasetdir = data_dir          # reuse the copy of paul15 shipped with the tutorials

out = Path('grid3d_walkthrough_output')
out.mkdir(exist_ok=True)


def seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def save(fig, name):
    fig.savefig(out / f'{name}.png')
    plt.close(fig)


ERY = ['7MEP', '6Ery', '5Ery', '4Ery', '3Ery', '2Ery', '1Ery']
MARKERS = ['Klf1', 'Gata1', 'Irf8', 'Csf1r', 'Elane', 'Mpo']
seed(0)

with scRL.plot_style():
    # 1. Data and grid: a 3-D UMAP of the cells and a regular grid laid over it
    adata = sc.datasets.paul15()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    markers = pd.DataFrame(adata[:, MARKERS].X, columns=MARKERS)    # log-normalised, not scaled
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50, random_state=0)
    sc.pp.neighbors(adata, random_state=0)
    sc.tl.umap(adata, n_components=3, random_state=0)
    X3, X_pca = adata.obsm['X_umap'], adata.obsm['X_pca']
    clusters = adata.obs['paul15_clusters'].reset_index(drop=True)
    gres = scRL.grids_from_embedding(X3)
    scRL.project_cluster(gres, clusters)
    scRL.align_pseudotime(gres, early_cluster='7MEP')
    scRL.project_back(gres, 'pseudotime')
    scRL.project(gres, markers)
    save(scRL.plot_grids(gres, key='clusters').figure, '01_grid_clusters')
    save(scRL.plot_slices(gres, key='clusters'), '01_slices_clusters')
    print(f'1. grid with {len(gres.grids["mapped_grids"])} grid points over {len(X3)} cells')

    # 2. Lineage mode, Decision: exact solution and a soft actor-critic for the erythroid lineage
    scRL.d_rewards(gres, starts=['7MEP'], ends=['1Ery', '2Ery'])
    reach = scRL.reachability(gres, 'd', 'Decision')
    scRL.exact_values(gres, key_add='ery_exact')
    seed(0)
    trainer = scRL.trainer('SAC', gres, reward_type='d', reward_mode='Decision', starts_prob=False,
                           X_latent=X_pca, num_episodes=2000, capacity=2000)
    returns, values = trainer.train()
    scRL.get_state_value(gres, trainer, 'ery_decision')
    rho = spearmanr(gres.grids['ery_decision'], gres.grids['ery_exact']).correlation
    save(scRL.plot_training(returns, values), '02_training')
    save(scRL.plot_report(gres, 'ery_exact', clusters=ERY, title='Erythroid lineage, exact value'),
         '02_report_exact')
    save(scRL.plot_slices(gres, key='ery_exact', policy=gres.grids['ery_exact_best']), '02_slices_policy')
    print(f'2. {reach["reached"]} of {reach["rewarded"]} rewarded grid points reachable; '
          f'learned against exact values: Spearman {rho:.2f}')

    # 3. Lineage mode, Contribution: the same lineage valued toward its far end
    scRL.d_rewards(gres, starts=['7MEP'], ends=['1Ery', '2Ery'], mode='Contribution')
    scRL.exact_values(gres, reward_mode='Contribution', key_add='ery_contribution')
    save(scRL.plot_trend(gres, ['ery_exact', 'ery_contribution'], labels=['Decision', 'Contribution'],
                         clusters=ERY, cells=False).figure, '03_trend_modes')
    print('3. erythroid values in Decision and Contribution mode along pseudotime')

    # 4. Gene mode: the reward of the erythroid genes Klf1 and Gata1
    scRL.c_rewards(gres, reward_keys=['Klf1', 'Gata1'], starts=['7MEP'])
    scRL.exact_values(gres, reward_type='c', key_add='ery_genes')
    rho = spearmanr(gres.grids['ery_genes'], gres.grids['ery_exact']).correlation
    print(f'4. gene reward against lineage reward: Spearman {rho:.2f}')

    # 5. Fate map over three lineages: a gate over exact values, and lineage shares from a walk
    lineages = {'erythroid': ['1Ery', '2Ery'], 'monocyte': ['14Mo', '15Mo'], 'neutrophil': ['16Neu', '17Neu']}
    for name, ends in lineages.items():
        scRL.d_rewards(gres, starts=['7MEP'], ends=ends)
        scRL.exact_values(gres, key_add=f'{name}_exact')
    scRL.fate_gate(gres, [f'{n}_exact' for n in lineages], names=list(lineages))
    scRL.fate_probabilities(gres, lineages)
    save(scRL.plot_fates(gres), '05_fates_gate')
    save(scRL.plot_fates(gres, key_add='fate'), '05_fates_shares')
    share = pd.Series(gres.grids['fate_top']).value_counts(normalize=True).round(2).to_dict()
    mep = np.asarray(gres.grids['mapped_grids_clusters']) == '7MEP'
    from_mep = {n: round(float(gres.grids[f'fate_{n}'][mep].mean()), 2) for n in lineages}
    print(f'5. heaviest lineage of the grid points from the walk: {share}; from 7MEP: {from_mep}, '
          f'arrived {gres.grids["fate_reached"][mep].mean():.2f}')

    # 6. Trajectories and simulator with the soft actor-critic of section 2
    scRL.d_rewards(gres, starts=['7MEP'], ends=['1Ery', '2Ery'])
    scRL.get_traj_df(gres, trainer, n=100, key='ery')
    scRL.traj_results(gres, ERY, 'ery')
    save(plt.gcf(), '06_trajectory')
    gres.grids['gene_exp'] = gres.grids['proj']
    trainer.train_simulator(500, 16)
    scRL.get_sim_df(gres, trainer, lineages=['1Ery', '2Ery', '14Mo', '15Mo'], steps=30, prefix='ery')
    here = Path.cwd()
    import os
    os.chdir(out)
    try:
        scRL.sim_results(gres, n_sample=1000, prefix='ery')
    finally:
        os.chdir(here)
    plt.close('all')
    print(f'6. trajectory of {len(gres.trajectory["ery_traj"])} averaged steps; simulator figures in '
          f'{out / "sim_output"}')

    # 7. Two conditions on one grid, made from the public data: 'treated' keeps only 15% of the neutrophil cells
    rng = np.random.default_rng(0)
    conditions = np.where(rng.random(len(X3)) < .5, 'treated', 'control')
    neutrophil = clusters.isin(['16Neu', '17Neu']).to_numpy()
    conditions[neutrophil] = np.where(rng.random(neutrophil.sum()) < .15, 'treated', 'control')
    scRL.project_conditions(gres, conditions)
    scRL.compare_abundance(gres, test='treated', reference='control', n_shuffles=100)
    print('7. per cluster:')
    print(gres.grids['abundance_clusters'].round(2).to_string())
    save(scRL.plot_conditions(gres), '07_conditions')
    save(scRL.plot_projections(gres, key='abundance_call'), '07_calls')
    values = scRL.compare_values(gres, 'treated', 'control', ends=['16Neu', '17Neu'], starts=['7MEP'],
                                 mode='Contribution', n_shuffles=50)
    print(values.round(3).to_string())
    print(scRL.compare_fates(gres, 'treated', 'control', key='fate_top').round(3).to_string())

    # 8. Sharing: a rotating view and an interactive page
    scRL.save_rotation(gres, out / 'rotation.gif', key='ery_exact', frames=36)
    try:
        page = scRL.plot_grids(gres, key=['ery_exact', 'fate_top', 'clusters'], interactive=True)
        page.write_html(out / 'interactive.html')
        print(f'8. rotation and interactive view written to {out}')
    except ImportError:
        print(f'8. rotation written to {out}; install plotly for the interactive view')

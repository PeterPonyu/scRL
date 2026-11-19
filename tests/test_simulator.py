import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import scRL
from conftest import make_tree, seed_all


def _ready3d():
    X, labels, Z, genes = make_tree(d=3)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    scRL.project(gres, genes)
    return gres, Z


@pytest.mark.parametrize('d', [2, 3])
def test_simulation_plots(d, scan2d, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if d == 2:
        gres, Z = scan2d, make_tree(d=2)[2]
    else:
        gres, Z = _ready3d()
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('ActorCritic', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=200)
    t.train()
    scRL.project_back(gres, 'pseudotime')
    t.train_simulator(100, 8)
    scRL.get_sim_df(gres, t, lineages=['branchA', 'branchB'], steps=10, prefix='s')
    scRL.sim_results(gres, n_sample=200, prefix='s')
    scRL.sim_results2(gres, ['root', 'trunk'], ['branchA'], prefix='s')
    scRL.sim_results3(gres, top=1, prefix='s')
    scRL.get_traj_df(gres, t, n=30, key='tr')
    scRL.sim_results(gres, n_sample=200, traj='tr', sim_traj=True, prefix='s')
    assert [f'c{k + 1}' for k in range(d)] == list(gres.simulating['s_df_traj'].columns[:d])
    plt.close('all')
    for name in ('lineages', 'pseudotime', 'pc_steps_corr', 'sim_genes_corr_bar'):
        assert (tmp_path / 'sim_output' / f'{name}.png').exists()
    assert list(gres.simulating['s_df_pc_corr'].columns) == ['lin1', 'lin2', 'steps', 'bins', 'time']
    assert list(gres.simulating['s_df_gene_corr'].columns) == ['time_corr']
    assert len(gres.simulating['s_df_gene_corr']) == 3

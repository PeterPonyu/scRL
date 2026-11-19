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


def test_3d_simulation_figures(tmp_path, monkeypatch):
    from test_plot import _assert_layout
    from test_trajectory import _legend_inside
    monkeypatch.chdir(tmp_path)
    gres, Z = _ready3d()
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('ActorCritic', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=200)
    t.train()
    scRL.project_back(gres, 'pseudotime')
    t.train_simulator(100, 8)
    lineages = ['branchA', 'branchB']
    scRL.get_sim_df(gres, t, lineages=lineages, steps=10, prefix='s')
    plt.close('all')
    scRL.sim_results(gres, n_sample=200, prefix='s')
    steps = gres.simulating['s_steps']
    names = ['lineages', 'pseudotime'] + [f'{k}_{s}' for s in [steps*.2, steps*.4, steps*.6, steps*.8, steps]
                                          for k in ('types', 'time', 'pie')]
    for name in names:
        assert (tmp_path / 'sim_output' / f'{name}.png').exists(), name
    titled = [plt.figure(n) for n in plt.get_fignums()
              if any(ax.get_title() == f'Simulated cells up to step {steps}' for ax in plt.figure(n).axes)]
    types_fig = [f for f in titled if f.axes[0].get_legend() is not None][0]
    labels = [t.get_text() for t in types_fig.axes[0].get_legend().get_texts()]
    assert labels and set(labels) <= set(lineages)
    _assert_layout(types_fig)
    _legend_inside(types_fig)
    plt.close('all')
    scRL.sim_results(gres, n_sample=200, sim_traj=True, prefix='s')
    assert list(gres.simulating['s_df_traj'].columns[:3]) == ['c1', 'c2', 'c3']
    plt.close('all')


def test_sim_results2_takes_single_names(scan2d, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gres, Z = scan2d, make_tree(d=2)[2]
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('ActorCritic', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=100)
    t.train()
    scRL.project_back(gres, 'pseudotime')
    t.train_simulator(50, 8)
    scRL.get_sim_df(gres, t, lineages=['branchA', 'branchB'], steps=10, prefix='s')
    scRL.sim_results(gres, n_sample=100, prefix='s')
    keys = ['s_lin1', 's_lin2', 's_df_pc_corr', 's_df_exp_corr']
    scRL.sim_results2(gres, ['root'], ['branchA'], prefix='s')
    expected = {k: gres.simulating[k] for k in keys} | {k: gres.embedding[k] for k in ('ori_pc_df', 'ori_exp_df')}
    scRL.sim_results2(gres, 'root', 'branchA', prefix='s')
    got = {k: gres.simulating[k] for k in keys} | {k: gres.embedding[k] for k in ('ori_pc_df', 'ori_exp_df')}
    for k in expected:
        if isinstance(expected[k], pd.DataFrame):
            pd.testing.assert_frame_equal(got[k], expected[k])
        else:
            assert got[k] == expected[k]
    plt.close('all')

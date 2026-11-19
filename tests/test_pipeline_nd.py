import numpy as np
import pandas as pd
import pytest

import scRL
from conftest import make_tree, seed_all


def _ready(d, **grid_kw):
    X, labels, Z, genes = make_tree(d=d)
    gres = scRL.grids_from_embedding(X, **grid_kw)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    scRL.project(gres, genes)
    return gres, Z


@pytest.mark.parametrize('algo', ['ActorCritic', 'DDQN', 'TabularQ'])
def test_three_agents_train_in_3d(algo):
    gres, Z = _ready(3)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer(algo, gres, reward_type='d', reward_mode='Decision', starts_prob=False, X_latent=Z, num_episodes=300)
    returns, _ = t.train()
    assert len(returns) == 300 and np.isfinite(returns).all()
    t.eval_fate()
    assert np.isfinite(np.asarray(gres.grids['fate_branchA'], dtype=float)).all()


@pytest.mark.parametrize('d', [2, 3])
def test_ddqn_fate_is_the_mean_q_value(d, scan2d):
    import torch
    if d == 2:
        gres, Z = scan2d, make_tree(d=2)[2]
    else:
        gres, Z = _ready(3)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('DDQN', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=100)
    t.train()
    t.eval_fate()
    fate = np.asarray(gres.grids['fate_branchA'], dtype=float)
    assert fate.shape == (len(gres.grids['mapped_grids']),) and np.isfinite(fate).all()
    net = t.agent.policy_net
    states = torch.tensor(t.env.state_space.mean(axis=1), device=next(net.parameters()).device)
    expected = net(states).mean(dim=1).detach().cpu().numpy()
    np.testing.assert_allclose(fate, expected, rtol=0, atol=1e-6)


def test_decision_value_readout_in_3d():
    gres, Z = _ready(3)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('ActorCritic', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=300)
    t.train()
    scRL.get_state_value(gres, t, 'decision')
    scRL.project_back(gres, 'decision')
    assert gres.grids['decision'].shape == (len(gres.grids['mapped_grids']),)
    v = gres.embedding['decision']
    assert v.shape == (600,) and np.isfinite(v).all()
    assert (v >= 0).all() and (v <= 1).all() and np.ptp(v) > 0


def test_knn_larger_than_the_number_of_cells():
    gres, Z = _ready(3)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    t = scRL.trainer('ActorCritic', gres, reward_type='d', starts_prob=False, X_latent=Z, KNN=1000, num_episodes=10)
    assert t.env.state_space.shape[1] == 600


def test_gene_rewards_with_start_probabilities_in_3d():
    gres, Z = _ready(3)
    scRL.c_rewards(gres, reward_keys=['geneA'], starts=['root'], starts_keys=['geneRoot'], punish_keys=['geneB'], mode='Contribution')
    seed_all(0)
    t = scRL.trainer('ActorCritic', gres, reward_type='c', reward_mode='Contribution', starts_prob=True, X_latent=Z, num_episodes=200)
    assert len(t.train()[0]) == 200


@pytest.mark.parametrize('d, kw', [(2, dict(method='occupancy')), (4, {}), (5, {}), (3, dict(moves='full'))])
def test_other_grids_run_end_to_end(d, kw):
    gres, Z = _ready(d, **kw)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('ActorCritic', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=100)
    assert len(t.train()[0]) == 100


def test_2d_scan_pipeline_still_runs(scan2d):
    scRL.d_rewards(scan2d, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('ActorCritic', scan2d, reward_type='d', starts_prob=False, X_latent=make_tree(d=2)[2], num_episodes=100)
    assert len(t.train()[0]) == 100
    scRL.get_state_value(scan2d, t, 'v'); scRL.project_back(scan2d, 'v')
    assert scan2d.embedding['v'].shape == (600,)


def test_trajectories_in_3d():
    gres, Z = _ready(3)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('ActorCritic', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=300)
    t.train()
    scRL.get_traj_df(gres, t, n=30, key='tr')
    assert gres.trajectory['tr_traj'].shape[1] == 3
    scRL.traj_results(gres, ['branchA'], 'tr')
    scRL.traj_results(gres, ['branchA'], 'tr', points=True)


@pytest.mark.parametrize('d', [2, 3])
def test_simulator_runs(d, scan2d):
    if d == 2:
        gres, Z = scan2d, make_tree(d=2)[2]
    else:
        gres, Z = _ready(3)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('ActorCritic', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=50)
    t.train()
    t.train_simulator(50, 8)
    scRL.get_sim_df(gres, t, lineages=['branchA', 'branchB'], steps=5, prefix='s')
    assert gres.simulating['s_df_coord'].shape[1] == d
    assert gres.simulating['s_df_exp'].shape[1] == 3

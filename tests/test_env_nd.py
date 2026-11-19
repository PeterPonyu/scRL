import random
import warnings

import numpy as np
import pandas as pd
import pytest

import scRL
from scRL.EnvironmentCore import deepEnv
from scRL.EnvironmentNDCore import deepEnvND, tabularEnvND, reachability
from scRL.GridNDCore import lattice_from_scan
from scRL.TabularQCore import tabularEnv
from conftest import make_tree, seed_all

Z2 = make_tree(d=2)[2]


def _pair(scan, **kw):
    nd = lattice_from_scan(scan)
    scRL.d_rewards(scan, starts=['root'], ends=['branchA'], **kw)
    scRL.d_rewards(nd, starts=['root'], ends=['branchA'], **kw)
    return scan, nd


GENE_REWARDS = dict(reward_keys=['geneA'], starts=['root'], starts_keys=['geneRoot'], punish_keys=['geneB'],
                    mode='Contribution')


def _pair_genes(scan):
    nd = lattice_from_scan(scan)
    scRL.c_rewards(scan, **GENE_REWARDS)
    scRL.c_rewards(nd, **GENE_REWARDS)
    return scan, nd


def _drive(old, new, key, n_episodes=300):
    rng = np.random.default_rng(0)
    ends = set()
    for ep in range(n_episodes):
        random.seed(ep); np.random.seed(ep); a0 = old.reset()
        random.seed(ep); np.random.seed(ep); b0 = new.reset()
        np.testing.assert_array_equal(a0, b0)
        while True:
            act = int(rng.integers(8))
            a, b = old.step(act), new.step(act)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1] == pytest.approx(b[1], abs=1e-12) and a[2:] == b[2:]
            assert getattr(old, key) == getattr(new, key)
            if a[2] or a[3]:
                ends.add((round(float(a[1]), 3) in (-2.0, -1.0), a[2], a[3]))
                break
    assert len(ends) >= 2                       # the walk met more than one kind of ending


def test_deep_env_matches_2d_code_step_for_step(scan2d):
    scan, nd = _pair(scan2d)
    kw = dict(max_step=50, KNN=10, reward_type='d', reward_mode='Decision', starts_probs=False)
    _drive(deepEnv(scan, Z2, **kw), deepEnvND(nd, Z2, **kw), 'state_idx')


def test_tabular_env_matches_2d_code_step_for_step(scan2d_n50):
    scan, nd = _pair(scan2d_n50)
    kw = dict(max_step=50, reward_type='d', reward_mode='Decision', starts_probs=False)
    _drive(tabularEnv(scan, **kw), tabularEnvND(nd, **kw), 'state')


def test_deep_env_with_start_probabilities_matches_2d_code(scan2d):
    scan, nd = _pair_genes(scan2d)
    kw = dict(max_step=50, KNN=10, reward_type='c', reward_mode='Contribution', starts_probs=True)
    _drive(deepEnv(scan, Z2, **kw), deepEnvND(nd, Z2, **kw), 'state_idx')


def test_tabular_env_with_start_probabilities_matches_2d_code(scan2d_n50):
    scan, nd = _pair_genes(scan2d_n50)
    kw = dict(max_step=50, reward_type='c', reward_mode='Contribution', starts_probs=True)
    _drive(tabularEnv(scan, **kw), tabularEnvND(nd, **kw), 'state')


def _grid3d(tree3d):
    gres = scRL.grids_from_embedding(tree3d[0])
    scRL.project_cluster(gres, pd.Series(pd.Categorical(tree3d[1])))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    return gres


def test_stop_rules_in_3d(tree3d):
    gres = _grid3d(tree3d)
    with pytest.warns(UserWarning, match='reach'):            # branchA lies beyond 6 moves of the root
        env = deepEnvND(gres, tree3d[2], max_step=5, KNN=10, reward_type='d', reward_mode='Decision',
                        starts_probs=False)
    nb, R = gres.grids['neighbors'], gres.qlearning['d_Decision_rewards'].to_numpy()
    pt = gres.grids['pseudotime']
    pos = {g: p for p, g in enumerate(gres.grids['mapped_grids'])}
    starts, boundary = set(gres.grids['starts_cluster_grids']), set(gres.grids['mapped_boundary'])
    seen = set()
    seed_all(0)
    for _ in range(400):
        env.reset()
        while True:
            p, here = pos[env.state_idx], env.state_idx
            a = random.randrange(6)
            _, r, done, trunc = env.step(a)
            if nb[p, a] == -1:
                assert (r, done) == (-2, True); seen.add('off')
            elif done and env.state_idx in boundary - starts:
                assert r == pytest.approx(R[p, a] + pt[env.state_idx] - pt[here], abs=1e-12)
                seen.add('boundary')
            elif done:
                assert r == -1; seen.add('revisit')
            elif trunc:
                assert env.time_step == 6; seen.add('cap')
            if done or trunc:
                break
    assert {'boundary', 'revisit', 'cap'} <= seen


def test_reachability_and_warning(tree3d):
    gres = _grid3d(tree3d)
    info = reachability(gres, 'd', 'Decision', max_step=50)
    assert info['rewarded'] > 0 and info['reached'] > 0 and info['min_steps'] >= 1
    with pytest.warns(UserWarning, match='reach'):
        deepEnvND(gres, tree3d[2], max_step=0, KNN=10, reward_type='d', reward_mode='Decision', starts_probs=False)
    with pytest.raises(ValueError, match='starts_keys'):
        deepEnvND(gres, tree3d[2], reward_type='d', reward_mode='Decision', starts_probs=True)

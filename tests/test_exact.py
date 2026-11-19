import itertools

import numpy as np
import pandas as pd
import pytest

import scRL
from scRL.EnvironmentNDCore import _stop_mask
from scRL.GridNDCore import lattice_from_scan
from conftest import make_tree, seed_all


def _small(n=12):
    X, labels, Z, genes = make_tree(d=3)
    gres = scRL.grids_from_embedding(X, n=n)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    scRL.project(gres, genes)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    return gres


def _sequence_value(nb, R, stop, t, p, moves, gamma):
    """Discounted return of one fixed sequence of moves from p, following the stop rules."""
    total, discount = 0.0, 1.0
    for a in moves:
        q = nb[p, a]
        if q < 0:
            return total + discount * -2.0
        if stop[q]:
            return total + discount * (R[p, a] + t[q] - t[p])
        total += discount * R[p, a]
        discount *= gamma
        p = q
    return total


def test_exact_values_equal_brute_force():
    gres = _small()
    V = scRL.exact_values(gres, max_step=3)
    nb, R = gres.grids['neighbors'], gres.qlearning['d_Decision_rewards'].to_numpy(dtype=float)
    stop = _stop_mask(gres, gres.grids['starts_cluster_grids'])
    t = gres.grids['pseudotime'].loc[gres.grids['mapped_grids']].to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    points = list(rng.choice(len(nb), 8, replace=False)) + list(np.flatnonzero(stop)[:2])
    for p in points:
        best = max(_sequence_value(nb, R, stop, t, p, seq, .9)
                   for seq in itertools.product(range(nb.shape[1]), repeat=4))
        assert V[p] == pytest.approx(best, abs=1e-10)


def test_exact_values_fixed_point():
    gres = _small()
    V = scRL.exact_values(gres, max_step=300)
    nb, R = gres.grids['neighbors'], gres.qlearning['d_Decision_rewards'].to_numpy(dtype=float)
    stop = _stop_mask(gres, gres.grids['starts_cluster_grids'])
    t = gres.grids['pseudotime'].loc[gres.grids['mapped_grids']].to_numpy(dtype=float)
    Q = np.full(nb.shape, -2.0)
    for p, a in zip(*np.nonzero(nb >= 0)):
        q = nb[p, a]
        Q[p, a] = R[p, a] + t[q] - t[p] if stop[q] else R[p, a] + .9 * V[q]
    np.testing.assert_allclose(V, Q.max(1), atol=1e-8)
    np.testing.assert_array_equal(gres.grids['exact_branchA_move'], Q.argmax(1))


def test_exact_values_other_rewards():
    gres = _small()
    scRL.c_rewards(gres, reward_keys=['geneA'], starts=['root'], mode='Contribution')
    V = scRL.exact_values(gres, reward_type='c', reward_mode='Contribution')
    assert np.isfinite(V).all() and 'exact_geneA' in gres.grids


def test_exact_values_on_a_scan_grid(scan2d):
    scRL.d_rewards(scan2d, starts=['root'], ends=['branchA'])
    nd = lattice_from_scan(scan2d)
    V = scRL.exact_values(scan2d)
    np.testing.assert_array_equal(V, scRL.exact_values(nd))
    np.testing.assert_array_equal(scan2d.grids['exact_branchA'], V)
    assert 'neighbors' not in scan2d.grids


def test_exact_values_keys_and_errors():
    gres = _small()
    V = scRL.exact_values(gres, key_add='mine')
    np.testing.assert_array_equal(gres.grids['mine'], V)
    assert gres.grids['mine_move'].shape == V.shape
    with pytest.raises(ValueError, match='d_rewards'):
        scRL.exact_values(gres, reward_mode='Contribution')


def _one_step(nb, R, stop, t, p, a):
    q = nb[p, a]
    if q < 0:
        return -2.0, True
    if stop[q]:
        return R[p, a] + t[q] - t[p], True
    return R[p, a], False


def _check_environment(gres, env, reward_table):
    mapped = np.asarray(gres.grids['mapped_grids'])
    lattice = gres if 'neighbors' in gres.grids else lattice_from_scan(gres)
    nb = lattice.grids['neighbors']
    R = gres.qlearning[reward_table].to_numpy(dtype=float)
    stop = _stop_mask(lattice, gres.grids.get('starts_cluster_grids'))
    t = gres.grids['pseudotime'].loc[mapped].to_numpy(dtype=float)
    env.reset()
    for p in range(len(mapped)):
        for a in range(nb.shape[1]):
            env.reset()
            env.state_idx = mapped[p]
            env.trajectory = [mapped[p]]
            env.time_step = 0
            if hasattr(env, '_p'):
                env._p, env._visited = p, {mapped[p]}
            _, reward, done, _ = env.step(a)
            expected, expected_done = _one_step(nb, R, stop, t, p, a)
            assert float(reward) == pytest.approx(expected, abs=1e-12) and done == expected_done


@pytest.mark.parametrize('rewards', ['lineage', 'gene'])
def test_one_step_model_equals_the_2d_environment(scan2d, rewards):
    from scRL.EnvironmentCore import deepEnv
    Z = make_tree(d=2)[2]
    if rewards == 'lineage':
        scRL.d_rewards(scan2d, starts=['root'], ends=['branchA'])
        table, kw = 'd_Decision_rewards', dict(reward_type='d')
    else:
        scRL.c_rewards(scan2d, reward_keys=['geneA'], starts=['root'])
        table, kw = 'c_Decision_rewards', dict(reward_type='c')
    env = deepEnv(scan2d, Z, KNN=10, reward_mode='Decision', starts_probs=False, **kw)
    _check_environment(scan2d, env, table)


def test_one_step_model_equals_the_nd_environment():
    from scRL.EnvironmentNDCore import deepEnvND
    gres = _small()
    env = deepEnvND(gres, make_tree(d=3)[2], KNN=10, reward_type='d', reward_mode='Decision', starts_probs=False)
    _check_environment(gres, env, 'd_Decision_rewards')

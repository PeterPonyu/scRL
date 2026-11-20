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


# ---------------------------------------------------------------- trajectories from the exact solution
def _tree_walks(gres, **kw):
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    scRL.exact_values(gres)
    return scRL.exact_trajectories(gres, **kw)


def _environment(gres, reward_type='d', reward_mode='Decision'):
    from scRL.GridNDCore import is_nd
    lattice = gres if is_nd(gres) else lattice_from_scan(gres)
    nb = np.asarray(lattice.grids['neighbors'])
    R = gres.qlearning[f'{reward_type}_{reward_mode}_rewards'].to_numpy(dtype=float)
    stop = _stop_mask(lattice, lattice.grids.get('starts_cluster_grids'))
    t = gres.grids['pseudotime'].loc[gres.grids['mapped_grids']].to_numpy(dtype=float)
    return nb, R, stop, t


def _loop_walks(nb, R, stop, t, starts, gamma=.9, max_step=50):
    """Independent plain-loop version: backward induction, then the greedy walk without returns."""
    M, A = len(nb), len(nb[0])
    V = [0.0] * M
    for _ in range(max_step + 1):
        Q = [[(-2.0 if nb[p][a] < 0 else R[p][a] + t[nb[p][a]] - t[p] if stop[nb[p][a]]
               else R[p][a] + gamma * V[nb[p][a]]) for a in range(A)] for p in range(M)]
        V = [max(row) for row in Q]
    out = []
    for s in starts:
        path, p, G, disc = [s], s, 0.0, 1.0
        for _ in range(max_step + 1):
            options = [a for a in range(A) if nb[p][a] >= 0 and nb[p][a] not in path]
            if not options:
                if all(nb[p][a] < 0 for a in range(A)):
                    G -= 2.0 * disc
                break
            best = max(Q[p][a] for a in options)
            a = next(a for a in options if Q[p][a] >= best - 1e-9)
            q = nb[p][a]
            path.append(q)
            if stop[q]:
                G += disc * (R[p][a] + t[q] - t[p])
                break
            G += disc * R[p][a]
            disc *= gamma
            p = q
        out.append((path, G))
    return out


def test_trajectories_never_return_and_stay_under_the_bound(scan2d):
    for gres in (_small(), scan2d):
        walks = _tree_walks(gres)
        assert len(walks) == len(gres.grids['starts_cluster_grids'])
        assert all(len(p) == len(set(p)) for p in walks['path'])                    # no grid point twice
        assert (walks['return'] <= walks['bound'] + 1e-9).all()                     # the relaxed bound
        assert (walks['moves'] <= 51).all() and set(walks['end']) <= {'stop point', 'step cap', 'trapped'}
        nb = _environment(gres)[0]
        mapped = np.asarray(gres.grids['mapped_grids'])
        pos = pd.Index(mapped)
        for p in walks['path']:                                                    # every move is along the table
            i = pos.get_indexer(p)
            assert all(j in nb[a] for a, j in zip(i[:-1], i[1:]))


def test_trajectories_reach_the_branch_on_the_tree(scan2d):
    # Every walk from the root enters branchA (share 1.0): on both tree grids the branch can be entered from the
    # root within the step cap, and the relaxed values rise along the trunk toward it, so the greedy walk
    # never runs out of unvisited points on the way.
    for gres in (_small(), scan2d):
        walks = _tree_walks(gres)
        s = gres.trajectory['exact_branchA_walks_summary']
        assert walks['entered'].all() and s['entered'] == 1.0 and s['bound_holds']
        ends = set(gres.grids['ends_cluster_grids'])
        for p, k in zip(walks['path'], walks['entry_move']):
            assert p[k] in ends and not ends.intersection(p[1:k])                  # first entry at entry_move
        assert 0 < s['bound_share_median'] <= 1


def test_trajectories_match_a_plain_loop(scan2d):
    for gres in (_small(), scan2d):
        walks = _tree_walks(gres)
        nb, R, stop, t = _environment(gres)
        mapped = np.asarray(gres.grids['mapped_grids'])
        starts = pd.Index(mapped).get_indexer(walks['start'])
        loop = _loop_walks(nb.tolist(), R.tolist(), stop.tolist(), t.tolist(), starts.tolist())
        for (path, G), p, g in zip(loop, walks['path'], walks['return']):
            assert mapped[path].tolist() == list(p)
            assert g == pytest.approx(G, abs=1e-12)


def test_trapped_walk_and_point_without_neighbours():
    from scRL.ExactCore import _solve, _walk, WALK_ENDS
    nb = np.array([[1, -1], [0, 2], [1, -1], [-1, -1]])                           # 0 - 1 - 2, and 3 alone
    R = np.zeros(nb.shape)
    R[1, 1] = 1.0                                                                  # entering 2 pays 1
    stop, t = np.zeros(4, bool), np.zeros(4)
    Q, V = _solve(nb, R, stop, t, .9, 5)
    target = np.array([False, False, True, False])
    path, moves, end, entry, G = _walk(nb, R, stop, t, Q, target, np.array([0, 3]), .9, 5, False)
    assert path[0, :3].tolist() == [0, 1, 2] and moves.tolist() == [2, 1] and entry.tolist() == [2, -1]
    assert [WALK_ENDS[e] for e in end] == ['trapped', 'left the grid']           # no return to 1; 3 must leave
    assert G == pytest.approx([.9, -2.0]) and (G <= V[[0, 3]] + 1e-12).all()


def test_trajectories_starts_and_seed():
    gres = _small()
    a = _tree_walks(gres, starts=7, seed=3, key_add='a')
    b = scRL.exact_trajectories(gres, starts=7, seed=3, key_add='b')
    assert a['start'].tolist() == b['start'].tolist() and a['start'].is_unique
    assert set(a['start']) <= set(gres.grids['starts_cluster_grids'])
    for col in ('return', 'moves'):
        np.testing.assert_array_equal(a[col], b[col])
    point = int(gres.grids['mapped_grids'][len(gres.grids['mapped_grids']) // 2])
    one = scRL.exact_trajectories(gres, starts=[point], key_add='one')
    assert one['start'].tolist() == [point] and one['path'][0][0] == point
    with pytest.raises(ValueError, match='not kept grid points'):
        scRL.exact_trajectories(gres, starts=[-7])
    with pytest.raises(ValueError, match='at least 1'):
        scRL.exact_trajectories(gres, starts=0)
    with pytest.raises(ValueError, match='only'):
        scRL.exact_trajectories(gres, starts=10 ** 6)
    for bad in (2.5, True):
        with pytest.raises(TypeError, match='whole number'):
            scRL.exact_trajectories(gres, starts=bad)
    none = scRL.exact_trajectories(gres, starts=[], key_add='none')
    assert len(none) == 0 and gres.trajectory['none_summary']['walks'] == 0
    assert gres.trajectory['none_traj'].shape == (0, 3)


def test_trajectories_errors():
    gres = _small()
    with pytest.raises(KeyError, match='call exact_values first'):
        scRL.exact_trajectories(gres, key='never_solved')
    scRL.exact_values(gres)
    scRL.d_rewards(gres, starts=['root'], ends=['branchB'])                        # another lineage
    with pytest.raises(ValueError, match='solved for the reward'):
        scRL.exact_trajectories(gres, key='exact_branchA')
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'], beta=3)               # same lineage, other table
    with pytest.raises(ValueError, match='no longer match'):
        scRL.exact_trajectories(gres, key='exact_branchA')
    scRL.exact_values(gres)
    del gres.grids['exact_branchA_settings']
    with pytest.raises(ValueError, match='No settings'):
        scRL.exact_trajectories(gres)


def test_trajectories_settings_follow_exact_values():
    gres = _small()
    scRL.exact_values(gres, gamma=.8, max_step=20, key_add='short')
    walks = scRL.exact_trajectories(gres, key='short')
    assert gres.grids['short_settings'] == dict(reward_type='d', reward_mode='Decision', gamma=.8, max_step=20,
                                                reward_key='branchA')
    assert (walks['moves'] <= 21).all() and (walks['return'] <= walks['bound'] + 1e-9).all()
    nb, R, stop, t = _environment(gres)
    starts = pd.Index(gres.grids['mapped_grids']).get_indexer(walks['start'])
    loop = _loop_walks(nb.tolist(), R.tolist(), stop.tolist(), t.tolist(), starts.tolist(), gamma=.8, max_step=20)
    np.testing.assert_allclose(walks['return'], [G for _, G in loop], atol=1e-12)


def test_trajectories_with_gene_rewards(scan2d):
    import warnings
    scRL.c_rewards(scan2d, reward_keys=['geneA'], starts=['root'])
    scRL.exact_values(scan2d, reward_type='c')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        walks = scRL.exact_trajectories(scan2d)
    s = scan2d.trajectory['exact_geneA_walks_summary']
    assert s['bound_holds'] and (walks['return'] <= walks['bound'] + 1e-9).all() and walks['entered'].mean() > .5
    scRL.c_rewards(scan2d, reward_keys=['geneA'], starts=['root'], punish_keys=['geneB'])
    scRL.exact_values(scan2d, reward_type='c')
    with pytest.warns(UserWarning, match='not an upper bound'):
        scRL.exact_trajectories(scan2d)
    assert not scan2d.trajectory['exact_geneA_walks_summary']['bound_holds']


def test_trajectories_until_entry_and_trajectory_views(scan2d):
    import matplotlib.pyplot as plt
    for gres, clusters in ((_small(), ['root', 'trunk', 'branchA']), (scan2d, ['root', 'trunk', 'branchA'])):
        full = _tree_walks(gres)
        cut = scRL.exact_trajectories(gres, until_entry=True, key_add='to_branch')
        ends = set(gres.grids['ends_cluster_grids'])
        done = cut['end'] == 'entered'
        assert done.any() and (cut.loc[done, 'entry_move'] == cut.loc[done, 'moves']).all()
        assert all(p[-1] in ends for p in cut.loc[done, 'path'])
        for p, q in zip(cut['path'], full['path']):                                # the same walk, cut short
            assert list(q[:len(p)]) == list(p)
        assert gres.grids['to_branch_visits'].sum() == sum(len(p) for p in cut['path'])
        assert gres.trajectory['to_branch_traj'].shape[1] == gres.grids['grids'].shape[1]
        for points in (False, True):
            plt.close('all')
            scRL.traj_results(gres, clusters, 'to_branch', points=points)
            assert plt.gcf().axes
        plt.close('all')


def _random_graph(rng):
    """A small undirected graph in the shape of a move table, with non-negative rewards, random stop points and a
    pseudotime that can fall along a move."""
    M, A = int(rng.integers(2, 16)), 6
    nb = -np.ones((M, A), int)
    for _ in range(int(rng.integers(0, 2 * M))):
        a, b = rng.integers(0, M, 2)
        if a != b and (nb[a] < 0).any() and (nb[b] < 0).any() and b not in nb[a]:
            nb[a, np.flatnonzero(nb[a] < 0)[0]] = b
            nb[b, np.flatnonzero(nb[b] < 0)[0]] = a
    R = rng.random((M, A)) * rng.integers(0, 2, (M, A))
    stop = (nb < 0).any(1) & (rng.random(M) < .7) | (rng.random(M) < .1)
    t = rng.normal(size=M) * rng.choice([0, 1, 3])
    return nb, R, stop, t


def _replay(nb, R, stop, t, Q, target, s, gamma, max_step, until_entry):
    """The walk replayed move by move from the environment's rules, choosing by the move values Q."""
    path, G, disc, entry, p = [s], 0.0, 1.0, -1, s
    for k in range(max_step + 1):
        kept = [a for a in range(nb.shape[1]) if nb[p, a] >= 0]
        free = [a for a in kept if nb[p, a] not in path]
        if not kept:
            return path, G - 2.0 * disc, 'left the grid', entry
        if not free:
            return path, G, 'trapped', entry
        a = next(a for a in free if Q[p, a] >= max(Q[p, b] for b in free) - 1e-9)
        q = nb[p, a]
        path.append(q)
        if target[q] and entry < 0:
            entry = k + 1
        if stop[q]:
            return path, G + disc * (R[p, a] + t[q] - t[p]), 'stop point', entry
        G += disc * R[p, a]
        disc *= gamma
        p = q
        if until_entry and entry == k + 1:
            return path, G, 'entered', entry
    return path, G, 'step cap', entry


def test_walk_matches_a_replay_and_the_bound_on_random_graphs():
    from scRL.ExactCore import _solve, _walk, WALK_ENDS
    rng = np.random.default_rng(2)
    walks = excess = 0
    for _ in range(300):
        nb, R, stop, t = _random_graph(rng)
        gamma, max_step = float(rng.choice([.5, .9, .99])), int(rng.choice([0, 1, 2, 5, 20]))
        target = rng.random(len(nb)) < .3
        until_entry = bool(rng.random() < .5)
        Q, V = _solve(nb, R, stop, t, gamma, max_step)
        S = np.arange(len(nb))
        path, moves, end, entry, G = _walk(nb, R, stop, t, Q, target, S, gamma, max_step, until_entry)
        for s in S:
            p, g, e, k = _replay(nb, R, stop, t, Q, target, s, gamma, max_step, until_entry)
            assert path[s][path[s] >= 0].tolist() == p and WALK_ENDS[end[s]] == e and entry[s] == k
            assert G[s] == pytest.approx(g, abs=1e-12)
            if not stop[s]:                                  # the bound holds for walks not started on a stop point
                excess = max(excess, G[s] - V[s])
            walks += 1
    assert walks > 2000 and excess <= 1e-9


def test_bound_flag_for_starts_on_stop_points():
    gres = _small()
    walks = _tree_walks(gres)
    assert gres.trajectory['exact_branchA_walks_summary']['bound_holds']
    from scRL.EnvironmentNDCore import _stop_mask
    stop = _stop_mask(gres, gres.grids['starts_cluster_grids'])
    on_stop = np.asarray(gres.grids['mapped_grids'])[stop][:3]
    with pytest.warns(UserWarning, match='stop points'):
        scRL.exact_trajectories(gres, starts=on_stop, key_add='edge')
    assert not gres.trajectory['edge_summary']['bound_holds']


def test_trajectories_from_start_probabilities(scan2d):
    scRL.c_rewards(scan2d, reward_keys=['geneA'], starts_keys=['geneRoot'])    # no start cluster
    scRL.exact_values(scan2d, reward_type='c')
    with pytest.raises(ValueError, match='no start cluster'):
        scRL.exact_trajectories(scan2d)
    a = scRL.exact_trajectories(scan2d, starts=6, seed=4, key_add='a')
    b = scRL.exact_trajectories(scan2d, starts=6, seed=4, key_add='b')
    assert a['start'].tolist() == b['start'].tolist() and a['start'].is_unique
    probs = np.asarray(scan2d.grids['starts_probs'])
    position = pd.Index(scan2d.grids['mapped_grids']).get_indexer(a['start'])
    assert (probs[position] > 0).all()

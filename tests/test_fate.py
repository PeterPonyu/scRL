import numpy as np
import pandas as pd
import pytest

import scRL
from conftest import make_tree, seed_all


def _gres():
    X, labels, Z, genes = make_tree(d=3)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    return gres


def test_gate_basic_properties():
    gres = _gres()
    M = len(gres.grids['mapped_grids'])
    rng = np.random.default_rng(0)
    for name in ('a', 'b', 'c'):
        gres.grids[name] = rng.normal(size=M)
    gate = scRL.fate_gate(gres, ['a', 'b', 'c'])
    np.testing.assert_allclose(gate.sum(1), 1)
    assert list(gate.columns) == ['a', 'b', 'c'] and list(gate.index) == list(gres.grids['mapped_grids'])
    assert set(gres.grids['gate_top']) <= {'a', 'b', 'c'}
    gres.grids['same'] = gres.grids['a'].copy()
    gate = scRL.fate_gate(gres, ['a', 'same'], key_add='g2')
    np.testing.assert_allclose(gate.to_numpy(), .5)
    np.testing.assert_allclose(gres.grids['g2_entropy'], 1)


def test_gate_top_rank_and_sharpness():
    gres = _gres()
    M = len(gres.grids['mapped_grids'])
    x = np.arange(M, dtype=float)
    gres.grids['up'], gres.grids['down'], gres.grids['down2'] = x, -x, -x
    gate = scRL.fate_gate(gres, ['up', 'down', 'down2'], tau=.1)
    assert gate['up'].iloc[-1] > .99
    sharp = scRL.fate_gate(gres, ['up', 'down'], tau=1e-6, key_add='sharp')
    middle = np.abs(np.arange(M) - (M - 1) / 2) > 1
    assert set(np.round(sharp.to_numpy()[middle], 6).ravel()) <= {0.0, 1.0}


def test_gate_cells_and_lineages():
    gres = _gres()
    for end in ('branchA', 'branchB'):
        scRL.d_rewards(gres, starts=['root'], ends=[end])
        scRL.exact_values(gres)
    gate = scRL.fate_gate(gres, ['exact_branchA', 'exact_branchB'], names=['A', 'B'])
    W = np.column_stack([gres.embedding['gate_A'], gres.embedding['gate_B']])
    np.testing.assert_allclose(W.sum(1), 1)
    X = gres.embedding['embedding']
    clusters = np.asarray(gres.embedding['clusters'])
    tipA = np.flatnonzero(clusters == 'branchA')[np.argmax(X[clusters == 'branchA'][:, 2])]
    tipB = np.flatnonzero(clusters == 'branchB')[np.argmax(X[clusters == 'branchB'][:, 1])]
    assert gres.embedding['gate_top'][tipA] == 'A' and gres.embedding['gate_top'][tipB] == 'B'
    assert set(gres.embedding['gate_top_colors']) == {'A', 'B', 'none'}
    assert 0 <= gres.embedding['gate_entropy'].min() and gres.embedding['gate_entropy'].max() <= 1


def test_gate_errors():
    gres = _gres()
    gres.grids['a'] = np.zeros(len(gres.grids['mapped_grids']))
    with pytest.raises(ValueError):
        scRL.fate_gate(gres, ['a'])
    with pytest.raises(KeyError):
        scRL.fate_gate(gres, ['a', 'nope'])
    with pytest.raises(ValueError):
        scRL.fate_gate(gres, ['a', 'a'], names=['x'])


def test_policy_views_accept_move_arrays():
    import torch
    from test_plot import _arrows, _policy_trainer
    gres = _gres()
    t = _policy_trainer(gres, 'ActorCritic', 3)
    t.agent.actor = lambda x: torch.nn.functional.one_hot(torch.full((len(x),), 0, device=x.device), 6).float()
    zeros = np.zeros(len(gres.grids['mapped_grids']), dtype=int)
    np.testing.assert_array_equal(_arrows(scRL.plot_policy(gres, zeros, every=2), True),
                                  _arrows(scRL.plot_policy(gres, t, every=2), True))
    from matplotlib.quiver import Quiver
    a = scRL.plot_slices(gres, key='pseudotime', policy=zeros, policy_every=2)
    b = scRL.plot_slices(gres, key='pseudotime', policy=t, policy_every=2)
    qa = [c for ax in a.axes for c in ax.collections if isinstance(c, Quiver)]
    qb = [c for ax in b.axes for c in ax.collections if isinstance(c, Quiver)]
    assert len(qa) == len(qb) > 0
    for x, y in zip(qa, qb):
        np.testing.assert_array_equal(np.column_stack([x.U, x.V]), np.column_stack([y.U, y.V]))


def _corridor(n=12):
    """A straight 1-wide corridor of kept points along the first axis of a 3-D lattice."""
    from scRL.GridCore import Grids_Results
    from scRL.GridNDCore import attach_lattice
    shape = (n, 3, 3)
    g = Grids_Results()
    g.grids['grids'] = np.indices(shape).reshape(3, -1).T.astype(float)
    g.grids['n'] = n
    g.grids['mapped_grids'] = np.ravel_multi_index((np.arange(n), np.ones(n, int), np.ones(n, int)), shape)
    attach_lattice(g, shape, 'axis')
    labels = np.array(['mid'] * n, dtype=object)
    labels[0], labels[-1] = 'left', 'right'
    g.grids['mapped_grids_clusters'] = labels
    g.embedding['embedding'] = g.grids['grids'][g.grids['mapped_grids']] + .01
    g.embedding['clusters'] = pd.Series(pd.Categorical(labels))
    return g


def test_gate_with_missing_values():
    gres = _gres()
    M = len(gres.grids['mapped_grids'])
    rng = np.random.default_rng(0)
    gres.grids['a'], gres.grids['b'] = rng.normal(size=M), rng.normal(size=M)
    gres.grids['a'][::5] = np.nan
    gate = scRL.fate_gate(gres, ['a', 'b'])
    assert gate.iloc[::5].isna().all().all() and (gres.grids['gate_top'][::5] == 'none').all()
    rest = np.ones(M, bool); rest[::5] = False
    np.testing.assert_allclose(gate[rest].sum(1), 1)
    assert set(gres.grids['gate_top'][rest]) <= {'a', 'b'} and len(set(gres.grids['gate_top'][rest])) == 2
    W = np.column_stack([gres.embedding['gate_a'], gres.embedding['gate_b']])
    np.testing.assert_allclose(W.sum(1), 1)


def test_tied_moves_draw_the_mean():
    from scRL.GridCore import Grids_Results
    from scRL.GridNDCore import attach_lattice
    from test_plot import _arrows
    shape = (6, 6, 6)
    g = Grids_Results()
    g.grids['grids'] = np.indices(shape).reshape(3, -1).T.astype(float)
    g.grids['n'] = 6
    g.grids['mapped_grids'] = np.arange(216)
    attach_lattice(g, shape, 'axis')
    g.embedding['embedding'] = g.grids['grids'] + .01
    best = np.zeros(g.grids['neighbors'].shape, bool)
    best[:, g.grids['actions'].index('x+')] = best[:, g.grids['actions'].index('y+')] = True
    arrows = _arrows(scRL.plot_policy(g, best, every=1), True)
    np.testing.assert_allclose(arrows, np.tile([.4, .4, 0], (len(arrows), 1)), atol=1e-9)


def test_move_array_indexed_by_stored_position():
    gres = _gres()
    M = len(gres.grids['mapped_grids'])
    moves = np.arange(M) % 6
    from scRL.PlotCore import _policy_moves, _lattice_spacing
    travel = np.asarray(gres.grids['mapped_travel'])
    got = _policy_moves(gres, moves, travel)
    pos = {g: p for p, g in enumerate(gres.grids['mapped_grids'])}
    expected = gres.grids['offsets'][[pos[g] % 6 for g in travel]] * _lattice_spacing(gres)[0]
    np.testing.assert_allclose(got, expected)
    assert len(np.unique(np.round(got, 9), axis=0)) == 6


def test_exact_best_mask():
    gres = _gres()
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    scRL.exact_values(gres)
    best = gres.grids['exact_branchA_best']
    assert best.shape == gres.grids['neighbors'].shape and best[np.arange(len(best)), gres.grids['exact_branchA_move']].all()
    assert (best.sum(1) > 1).any()


def test_reserved_lineage_names_and_foreign_trainer(scan2d):
    gres = _gres()
    M = len(gres.grids['mapped_grids'])
    gres.grids['a'], gres.grids['b'] = np.arange(M, dtype=float), -np.arange(M, dtype=float)
    for name in ('top', 'entropy', 'none'):
        with pytest.raises(ValueError, match='reserved'):
            scRL.fate_gate(gres, ['a', 'b'], names=[name, 'x'])
        with pytest.raises(ValueError, match='reserved'):
            scRL.fate_probabilities(gres, {name: 'branchA', 'x': 'branchB'})
    from test_plot import _policy_trainer
    foreign = _policy_trainer(scan2d, 'ActorCritic', 2)
    with pytest.raises(ValueError, match='grid'):
        scRL.fate_probabilities(gres, {'A': 'branchA', 'B': 'branchB'}, policy=foreign)


ENDS = {'A': 'branchA', 'B': 'branchB'}


def test_shares_gamblers_ruin_and_short_walks():
    g = _corridor()
    shares = scRL.fate_probabilities(g, {'L': 'left', 'R': 'right'}, forward=0, steps=20000)
    np.testing.assert_allclose(shares['R'].to_numpy(), np.arange(12) / 11, atol=1e-6)
    np.testing.assert_allclose(shares.sum(1), 1, atol=1e-6)
    short = scRL.fate_probabilities(g, {'L': 'left', 'R': 'right'}, forward=0, steps=3, key_add='short')
    reached = short.sum(1).to_numpy()
    distance = np.minimum(np.arange(12), 11 - np.arange(12))
    assert (reached[distance > 3] == 0).all() and (reached[distance <= 3] > 0).all()


def test_shares_grow_with_steps():
    gres = _gres()
    few = scRL.fate_probabilities(gres, ENDS, steps=200, key_add='few')
    many = scRL.fate_probabilities(gres, ENDS, steps=1000, key_add='many')
    assert (few.to_numpy() <= many.to_numpy() + 1e-12).all()


@pytest.mark.parametrize('forward', [0, 1, 5, 20, 80, 200])
def test_shares_stay_valid_at_any_bias(forward):
    gres = _gres()
    shares = scRL.fate_probabilities(gres, ENDS, forward=forward)
    H = shares.to_numpy()
    assert (H >= 0).all() and (H.sum(1) <= 1 + 1e-12).all()
    clusters = np.asarray(gres.grids['mapped_grids_clusters'])
    assert (shares['A'].to_numpy()[clusters == 'branchA'] == 1).all()
    np.testing.assert_allclose(gres.grids['fate_reached'], H.sum(1))


def test_shares_on_the_tree():
    X, labels, Z, genes = make_tree(d=3)
    labels = labels.astype(object)
    far = np.linalg.norm(X - np.array([6, 0, 0]), axis=1) > 3.5          # the far half of each branch
    labels[(labels == 'branchA') & far] = 'tipA'
    labels[(labels == 'branchB') & far] = 'tipB'
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels.astype(str))))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    shares = scRL.fate_probabilities(gres, {'A': 'tipA', 'B': ['tipB']}, forward=5)
    clusters = np.asarray(gres.grids['mapped_grids_clusters'])
    lower_a = clusters == 'branchA'                                        # downstream of the branch point
    assert lower_a.sum() > 20 and np.median(shares['A'].to_numpy()[lower_a]) > .9
    W = np.column_stack([gres.embedding['fate_A'], gres.embedding['fate_B']])
    assert (W >= 0).all() and (W.sum(1) <= 1 + 1e-9).all()
    assert scRL.plot_fates(gres, key_add='fate') is not None


def test_shares_island_and_unlisted_tip():
    X, labels, Z, genes = make_tree(d=3)
    island = X[:40] + np.array([0, 30, 0], dtype=np.float32)
    gres = scRL.grids_from_embedding(np.vstack([X, island]))
    scRL.project_cluster(gres, pd.Series(pd.Categorical(list(labels) + ['island'] * 40)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    clusters = np.asarray(gres.grids['mapped_grids_clusters'])
    shares = scRL.fate_probabilities(gres, {'A': 'branchA', 'R': 'root'}, forward=3)
    on_island = clusters == 'island'
    assert (gres.grids['fate_reached'][on_island] == 0).all() and (gres.grids['fate_top'][on_island] == 'none').all()
    assert np.isnan(gres.grids['fate_entropy'][on_island]).all()
    tip = clusters == 'branchB'                                            # not listed: walks there do not arrive
    assert (gres.grids['fate_top'][tip] == 'none').mean() > .8
    trunk = clusters == 'trunk'
    assert np.isfinite(shares.to_numpy()[trunk]).all() and (gres.grids['fate_reached'][trunk] > 0).all()
    assert gres.grids['fate_top_colors']['none'] == '#d9d9d9'


def test_shares_with_policies(scan2d):
    gres = _gres()
    walk = scRL.fate_probabilities(gres, ENDS, forward=1, key_add='walk')
    nb = gres.grids['neighbors']
    t = gres.grids['pseudotime'].loc[gres.grids['mapped_grids']].to_numpy()
    rows, cols = np.nonzero(nb >= 0)
    diff = t[nb[rows, cols]] - t[rows]
    weights = np.zeros(nb.shape)
    weights[rows, cols] = np.exp(diff / np.median(np.abs(diff[diff != 0])))
    same = scRL.fate_probabilities(gres, ENDS, policy=weights, key_add='same')
    np.testing.assert_allclose(same.to_numpy(), walk.to_numpy(), atol=1e-10)
    stopped = weights.copy()
    stopped[0] = 0
    scRL.fate_probabilities(gres, ENDS, policy=stopped, key_add='stopped')
    clusters = np.asarray(gres.grids['mapped_grids_clusters'])
    if clusters[0] not in ('branchA', 'branchB'):
        assert gres.grids['stopped_reached'][0] == 0
    from test_plot import _policy_trainer
    actor = _policy_trainer(gres, 'ActorCritic', 3)
    shares = scRL.fate_probabilities(gres, ENDS, policy=actor, key_add='actor')
    assert (shares.to_numpy() >= 0).all() and (shares.sum(1) <= 1 + 1e-12).all()
    with pytest.raises(ValueError, match='actor'):
        scRL.fate_probabilities(gres, ENDS, policy=_policy_trainer(gres, 'TabularQ', 3))
    with pytest.raises(ValueError, match='grid'):
        scRL.fate_probabilities(gres, ENDS, policy=_policy_trainer(scan2d, 'ActorCritic', 2))


def test_shares_one_hot_path():
    g = _corridor()
    weights = np.zeros(g.grids['neighbors'].shape)
    weights[:, g.grids['actions'].index('x+')] = 1
    shares = scRL.fate_probabilities(g, {'L': 'left', 'R': 'right'}, policy=weights)
    np.testing.assert_allclose(shares['R'].to_numpy()[1:], 1)


def test_shares_on_a_scan_grid_and_in_comparisons(scan2d):
    shares = scRL.fate_probabilities(scan2d, ENDS)
    assert len(shares) == len(scan2d.grids['mapped_grids']) and (shares.to_numpy() >= 0).all()
    scRL.project_conditions(scan2d, np.where(np.arange(600) % 2, 'x', 'y'))
    table = scRL.compare_fates(scan2d, 'x', 'y', key='fate_top', n_permutations=50, n_boot=20)
    np.testing.assert_allclose(table[['reference', 'test']].sum(0), 1)


def test_shares_errors():
    gres = _gres()
    with pytest.raises(ValueError):
        scRL.fate_probabilities(gres, {'A': 'branchA'})
    with pytest.raises(ValueError):
        scRL.fate_probabilities(gres, {'A': ['branchA', 'trunk'], 'B': ['trunk']})
    with pytest.raises(ValueError):
        scRL.fate_probabilities(gres, {'A': 'branchA', 'B': 'nowhere'})
    with pytest.raises(ValueError, match='steps'):
        scRL.fate_probabilities(gres, ENDS, steps=0)
    with pytest.raises(ValueError, match='reserved'):
        scRL.fate_probabilities(gres, {'reached': 'branchA', 'B': 'branchB'})


def test_shares_from_a_trained_soft_actor_critic():
    gres = _gres()
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('SAC', gres, reward_type='d', starts_prob=False, X_latent=make_tree(d=3)[2], num_episodes=300,
                     capacity=2000)
    t.train()
    shares = scRL.fate_probabilities(gres, ENDS, policy=t, key_add='sac')
    H = shares.to_numpy()
    assert H.dtype == np.float64 and (H >= 0).all() and (H.sum(1) <= 1 + 1e-12).all()


def test_policy_arrays_are_validated():
    gres = _gres()
    base = np.ones(gres.grids['neighbors'].shape)
    for bad in (np.nan, np.inf, -1.0):
        weights = base.copy()
        weights[3, 1] = bad
        with pytest.raises(ValueError, match='finite and not negative'):
            scRL.fate_probabilities(gres, ENDS, policy=weights)

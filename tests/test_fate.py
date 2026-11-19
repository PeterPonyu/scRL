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


def test_fate_probabilities_gamblers_ruin():
    g = _corridor()
    shares = scRL.fate_probabilities(g, {'L': 'left', 'R': 'right'}, forward=0)
    np.testing.assert_allclose(shares['R'].to_numpy(), np.arange(12) / 11, atol=1e-8)
    np.testing.assert_allclose(shares.sum(1), 1)


def test_fate_probabilities_on_the_tree():
    X, labels, Z, genes = make_tree(d=3)
    labels = labels.astype(object)
    far = np.linalg.norm(X - np.array([6, 0, 0]), axis=1) > 3.5          # the far half of each branch
    labels[(labels == 'branchA') & far] = 'tipA'
    labels[(labels == 'branchB') & far] = 'tipB'
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels.astype(str))))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    shares = scRL.fate_probabilities(gres, {'A': 'tipA', 'B': ['tipB']}, forward=5)
    finite = ~shares.isna().any(axis=1)
    np.testing.assert_allclose(shares[finite].sum(1), 1)
    clusters = np.asarray(gres.grids['mapped_grids_clusters'])
    assert (shares['A'].to_numpy()[clusters == 'tipA'] == 1).all()
    lower_a = clusters == 'branchA'                                        # downstream of the branch point
    assert lower_a.sum() > 20 and np.median(shares['A'].to_numpy()[lower_a]) > .9
    W = np.column_stack([gres.embedding['fate_A'], gres.embedding['fate_B']])
    np.testing.assert_allclose(W[~np.isnan(W).any(1)].sum(1), 1)
    assert scRL.plot_fates(gres, key_add='fate') is not None


def test_fate_probabilities_island():
    X, labels, Z, genes = make_tree(d=3)
    island = X[:40] + np.array([0, 30, 0], dtype=np.float32)
    gres = scRL.grids_from_embedding(np.vstack([X, island]))
    scRL.project_cluster(gres, pd.Series(pd.Categorical(list(labels) + ['island'] * 40)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    shares = scRL.fate_probabilities(gres, {'A': 'branchA', 'B': 'branchB'})
    on_island = np.asarray(gres.grids['mapped_grids_clusters']) == 'island'
    assert shares[on_island].isna().all().all() and (gres.grids['fate_top'][on_island] == 'none').all()
    assert np.isnan(gres.grids['fate_entropy'][on_island]).all()
    assert shares[~on_island].notna().any(axis=1).all()
    assert gres.grids['fate_top_colors']['none'] == '#d9d9d9'


def test_fate_probabilities_policies():
    gres = _gres()
    walk = scRL.fate_probabilities(gres, {'A': 'branchA', 'B': 'branchB'}, forward=1, key_add='walk')
    nb = gres.grids['neighbors']
    t = gres.grids['pseudotime'].loc[gres.grids['mapped_grids']].to_numpy()
    rows, cols = np.nonzero(nb >= 0)
    diff = t[nb[rows, cols]] - t[rows]
    weights = np.zeros(nb.shape)
    weights[rows, cols] = np.exp(diff / np.median(np.abs(diff[diff != 0])))
    same = scRL.fate_probabilities(gres, {'A': 'branchA', 'B': 'branchB'}, policy=weights, key_add='same')
    np.testing.assert_allclose(same.to_numpy(), walk.to_numpy(), atol=1e-10)
    t_tab = __import__('test_plot')._policy_trainer(gres, 'TabularQ', 3)
    with pytest.raises(ValueError, match='actor'):
        scRL.fate_probabilities(gres, {'A': 'branchA', 'B': 'branchB'}, policy=t_tab)


def test_fate_probabilities_one_hot_path():
    g = _corridor()
    right = g.grids['actions'].index('x+')
    weights = np.zeros(g.grids['neighbors'].shape)
    weights[:, right] = 1
    shares = scRL.fate_probabilities(g, {'L': 'left', 'R': 'right'}, policy=weights)
    np.testing.assert_allclose(shares['R'].to_numpy()[1:], 1)


def test_fate_probabilities_scan_grid_and_conditions(scan2d):
    shares = scRL.fate_probabilities(scan2d, {'A': 'branchA', 'B': 'branchB'})
    assert len(shares) == len(scan2d.grids['mapped_grids'])
    scRL.project_conditions(scan2d, np.where(np.arange(600) % 2, 'x', 'y'))
    table = scRL.compare_fates(scan2d, 'x', 'y', key='fate_top', n_permutations=50, n_boot=20)
    np.testing.assert_allclose(table[['reference', 'test']].sum(0), 1)


def test_fate_probabilities_errors():
    gres = _gres()
    with pytest.raises(ValueError):
        scRL.fate_probabilities(gres, {'A': 'branchA'})
    with pytest.raises(ValueError):
        scRL.fate_probabilities(gres, {'A': ['branchA', 'trunk'], 'B': ['trunk']})
    with pytest.raises(ValueError):
        scRL.fate_probabilities(gres, {'A': 'branchA', 'B': 'nowhere'})


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


@pytest.mark.parametrize('forward', [1, 5, 10, 20, 40, 80, 200])
def test_fate_probabilities_stable_or_refused(forward):
    gres = _gres()
    try:
        shares = scRL.fate_probabilities(gres, {'A': 'branchA', 'B': 'branchB'}, forward=forward)
    except RuntimeError as err:
        assert 'forward' in str(err)
        return
    finite = shares.dropna()
    np.testing.assert_allclose(finite.sum(1), 1, atol=1e-6)
    assert finite.to_numpy().min() >= 0 and finite.to_numpy().max() <= 1


def test_fate_probabilities_dead_ends_at_strong_bias():
    gres = _gres()
    shares = scRL.fate_probabilities(gres, {'A': 'branchA', 'B': 'branchB'}, forward=40)
    missing = shares.isna().any(axis=1).to_numpy()
    assert missing.any() and (gres.grids['fate_top'][missing] == 'none').all()
    nb = gres.grids['neighbors']
    t = gres.grids['pseudotime'].loc[gres.grids['mapped_grids']].to_numpy()
    clusters = np.asarray(gres.grids['mapped_grids_clusters'])
    peak = [p for p in range(len(nb)) if clusters[p] not in ('branchA', 'branchB')
            and all(t[q] <= t[p] for q in nb[p] if q >= 0)]
    assert peak and missing[peak].any()


def test_tiny_policy_weights_count_as_absent():
    g = _corridor()
    right, left = g.grids['actions'].index('x+'), g.grids['actions'].index('x-')
    weights = np.zeros(g.grids['neighbors'].shape)
    weights[:, right] = 1
    weights[:, left] = 1e-20
    shares = scRL.fate_probabilities(g, {'L': 'left', 'R': 'right'}, policy=weights)
    np.testing.assert_allclose(shares['R'].to_numpy()[1:], 1)
    weights[:, right] = 0
    weights[5, right] = 1e-20
    weights[:, left] = 1
    shares = scRL.fate_probabilities(g, {'L': 'left', 'R': 'right'}, policy=weights, key_add='back')
    assert shares['R'].to_numpy()[5] == 0


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

import numpy as np
import pandas as pd
import pytest

import scRL
from scRL.EnvironmentCore import get_state
from scRL.GridNDCore import lattice_from_scan, get_state_nd
from conftest import make_tree, seed_all

LABELS = pd.Series(pd.Categorical(make_tree(d=2)[1]))


def test_cluster_projection_equals_2d_code(scan2d):
    nd = lattice_from_scan(scan2d)
    del nd.grids['mapped_grids_clusters']
    scRL.project_cluster(nd, LABELS)
    assert np.array_equal(nd.grids['mapped_grids_clusters'], scan2d.grids['mapped_grids_clusters'])


@pytest.mark.parametrize('boundary', [False, True])
def test_pseudotime_equals_2d_code(scan2d, boundary):
    nd = lattice_from_scan(scan2d)
    np.random.seed(3); scRL.align_pseudotime(scan2d, 'root', boundary=boundary)
    np.random.seed(3); scRL.align_pseudotime(nd, 'root', boundary=boundary)
    a, b = scan2d.grids['pseudotime'], nd.grids['pseudotime']
    assert list(a.index) == list(b.index)
    np.testing.assert_allclose(b.values, a.values, rtol=0, atol=1e-12)


def test_gene_projection_equals_2d_code(scan2d):
    nd = lattice_from_scan(scan2d)
    genes = make_tree(d=2)[3]
    scRL.project(nd, genes)
    assert list(nd.grids['proj'].columns) == list(scan2d.grids['proj'].columns)
    np.testing.assert_allclose(nd.grids['proj'].values.astype(float),
                               scan2d.grids['proj'].values.astype(float), atol=1e-5)


def test_project_back_equals_2d_code(scan2d):
    nd = lattice_from_scan(scan2d)
    scRL.project_back(scan2d, 'pseudotime'); scRL.project_back(nd, 'pseudotime')
    np.testing.assert_allclose(nd.embedding['pseudotime'], scan2d.embedding['pseudotime'], atol=1e-5)


def test_state_vectors_equal_2d_code(scan2d):
    Z = make_tree(d=2)[2]
    np.testing.assert_allclose(get_state_nd(lattice_from_scan(scan2d), Z, 10).mean(1),
                               get_state(scan2d, Z, 10).mean(1), atol=1e-6)


def test_unknown_start_cluster_raises(tree3d):
    gres = scRL.grids_from_embedding(tree3d[0])
    scRL.project_cluster(gres, pd.Series(pd.Categorical(tree3d[1])))
    with pytest.raises(ValueError, match='root'):
        scRL.align_pseudotime(gres, 'no_such_cluster')


def test_100k_cells_project_quickly():
    import time
    X, labels, _, _ = make_tree(n_cells=1000, d=3)
    X = np.repeat(X, 100, axis=0) + np.random.default_rng(0).normal(0, .05, (100000, 3)).astype(np.float32)
    gres = scRL.grids_from_embedding(X)
    t0 = time.time()
    scRL.project_cluster(gres, pd.Series(pd.Categorical(np.repeat(labels, 100))))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    scRL.project_back(gres, 'pseudotime')
    assert time.time() - t0 < 30 and gres.embedding['pseudotime'].shape == (100000,)


def test_islands_get_finite_pseudotime():
    X, labels, _, _ = make_tree(d=3)
    island = X[:40] + np.array([0, 30, 0], dtype=np.float32)          # a far, disconnected clump
    gres = scRL.grids_from_embedding(np.vstack([X, island]))
    scRL.project_cluster(gres, pd.Series(pd.Categorical(list(labels) + ['island'] * 40)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    t = gres.grids['pseudotime']
    assert np.isfinite(t.values).all() and t.min() == 0 and t.max() == 1


@pytest.mark.parametrize('mode', ['Decision', 'Contribution'])
def test_lineage_rewards_equal_2d_code(scan2d, mode):
    nd = lattice_from_scan(scan2d)
    scRL.d_rewards(scan2d, starts=['root'], ends=['branchA'], mode=mode)
    scRL.d_rewards(nd, starts=['root'], ends=['branchA'], mode=mode)
    a, b = scan2d.qlearning[f'd_{mode}_rewards'].astype(float), nd.qlearning[f'd_{mode}_rewards']
    assert list(a.index) == list(b.index) and list(a.columns) == list(b.columns)
    np.testing.assert_allclose(b.values, a.values, rtol=0, atol=1e-12)
    assert np.array_equal(nd.qlearning['matrix'], scan2d.qlearning['matrix'])
    assert list(nd.grids['starts_cluster_grids']) == list(scan2d.grids['starts_cluster_grids'])
    assert list(nd.grids['ends_cluster_grids']) == list(scan2d.grids['ends_cluster_grids'])


@pytest.mark.parametrize('kw', [
    dict(reward_keys=['geneA'], starts=['root'], mode='Decision'),
    dict(reward_keys=['geneA'], starts=['root'], starts_keys=['geneRoot'], punish_keys=['geneB'], mode='Contribution'),
    dict(reward_keys=['geneA', 'geneB'], starts_keys=['geneRoot'], mode='Decision'),
])
def test_gene_rewards_equal_2d_code(scan2d, kw):
    nd = lattice_from_scan(scan2d)
    scRL.c_rewards(scan2d, **kw); scRL.c_rewards(nd, **kw)
    key = f"c_{kw['mode']}_rewards"
    np.testing.assert_allclose(nd.qlearning[key].values, scan2d.qlearning[key].astype(float).values, rtol=0, atol=1e-10)
    if 'starts_keys' in kw:
        np.testing.assert_allclose(nd.grids['starts_probs'], scan2d.grids['starts_probs'], atol=1e-12)
    assert (nd.grids['starts_cluster_grids'] is None) == (scan2d.grids['starts_cluster_grids'] is None)


def _grid3d(tree3d):
    gres = scRL.grids_from_embedding(tree3d[0])
    scRL.project_cluster(gres, pd.Series(pd.Categorical(tree3d[1])))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    return gres


def test_lineage_rewards_in_3d(tree3d):
    gres = _grid3d(tree3d)
    scRL.d_rewards(gres, starts='root', ends=['branchA'])           # a bare string is one cluster name
    R, nb = gres.qlearning['d_Decision_rewards'].values, gres.grids['neighbors']
    assert R.shape == nb.shape == (len(gres.grids['mapped_grids']), 6)
    assert ((R == -1) == (nb == -1)).all()
    is_end = gres.grids['mapped_grids_clusters'] == 'branchA'
    into_end = (nb >= 0) & is_end[np.clip(nb, 0, None)]
    assert (R[into_end] > 0).all() and (R[into_end] <= 1).all() and (R[(nb >= 0) & ~into_end] == 0).all()
    with pytest.raises(ValueError, match='branchA'):
        scRL.d_rewards(gres, starts=['root'], ends=['nowhere'])


def _split_pair(scan, cut, single=False, order=None):
    """
    The 2-D scan grid with lattice columns cut and cut+1 removed, as a 2-D code grid and as an n-D grid.

    single: also keep only one grid point of column cut+2, in a row also kept in column cut-1, so that the
    nearest pair between the two parts is unique. order: a permutation of the stored order of the kept points.
    """
    from scRL.GridCore import Grids_Results
    from scRL.GridNDCore import attach_lattice
    n, mapped = scan.grids['n'], scan.grids['mapped_grids']
    col, row = mapped // n, mapped % n
    keep = (col != cut) & (col != cut + 1)
    if single:
        shared = np.intersect1d(row[col == cut - 1], row[col == cut + 2])
        keep &= (col != cut + 2) | (row == shared[len(shared) // 2])
    kept = mapped[keep]
    clusters = scan.grids['mapped_grids_clusters'][keep]
    if order is not None:
        kept, clusters = kept[order], clusters[order]
    nd = Grids_Results()
    nd.embedding.update(scan.embedding)
    nd.grids.update(n=n, grids=scan.grids['grids'], mapped_grids=kept, mapped_grids_clusters=clusters)
    attach_lattice(nd, (n, n), 'full')
    old = Grids_Results()
    old.embedding.update(scan.embedding)
    adj = pd.DataFrame(nd.grids['mapped_adj'].toarray(), index=list(kept), columns=list(kept))
    old.grids.update(n=n, grids=scan.grids['grids'], mapped_grids=kept, mapped_grids_clusters=clusters,
                     mapped_boundary=nd.grids['mapped_boundary'], mapped_adj=adj)
    return old, nd


def _parts(nd):
    from scipy.sparse.csgraph import connected_components
    n_parts, part = connected_components(nd.grids['mapped_adj'], directed=False)
    assert n_parts == 2
    return part


def _nearest_pairs(nd, in_a, in_b):
    """All pairs (grid index in a, grid index in b) at the minimum distance, on the stored coordinates."""
    G, mapped = nd.grids['grids'], nd.grids['mapped_grids']
    D = np.vstack([np.linalg.norm(G[mapped[p]] - G[mapped[in_b]], axis=1) for p in in_a])
    rows, cols = np.nonzero(D == D.min())
    return sorted(zip(mapped[in_a[rows]].tolist(), mapped[in_b[cols]].tolist()))


@pytest.mark.parametrize('cut', [4, 7, 9])
@pytest.mark.parametrize('start', ['root', 'branchA'])
@pytest.mark.parametrize('boundary', [False, True])
def test_split_grid_pseudotime_equals_2d_code_when_the_nearest_pair_is_unique(scan2d, cut, start, boundary):
    old, nd = _split_pair(scan2d, cut, single=True)
    part = _parts(nd)
    assert len(_nearest_pairs(nd, np.flatnonzero(part == 0), np.flatnonzero(part == 1))) == 1
    np.random.seed(4); scRL.align_pseudotime(old, start, boundary=boundary, n_jobs=1)
    np.random.seed(4); scRL.align_pseudotime(nd, start, boundary=boundary)
    a, b = old.grids['pseudotime'], nd.grids['pseudotime']
    assert list(a.index) == list(b.index)
    np.testing.assert_allclose(b.values, a.values, rtol=0, atol=1e-12)


@pytest.mark.parametrize('cut, start', [(2, 'root'), (7, 'root'), (7, 'branchA')])
def test_split_grid_tie_rule(scan2d, cut, start):
    from scipy.sparse.csgraph import dijkstra
    _, nd = _split_pair(scan2d, cut)
    part, mapped, adj = _parts(nd), nd.grids['mapped_grids'], nd.grids['mapped_adj']
    np.random.seed(4)
    starts = np.random.choice(mapped[np.where(nd.grids['mapped_grids_clusters'] == start)[0]], 10)
    pos = {g: p for p, g in enumerate(mapped)}
    starts = np.array([pos[g] for g in starts])
    main = np.bincount(part[starts], minlength=2).argmax()
    in_main, in_other = np.flatnonzero(part == main), np.flatnonzero(part != main)
    pairs = _nearest_pairs(nd, in_main, in_other)
    assert len(pairs) > 1                                     # the tie rule is exercised
    m, o = pos[pairs[0][0]], pos[pairs[0][1]]                 # smallest main grid index, then partner
    t = dijkstra(adj, directed=False, indices=np.unique(starts[part[starts] == main])).mean(0)
    t[in_other] = dijkstra(adj, directed=False, indices=o)[in_other] + t[m]
    t = (t - t.min()) / (t.max() - t.min())
    np.random.seed(4); scRL.align_pseudotime(nd, start, boundary=False)
    np.testing.assert_allclose(nd.grids['pseudotime'].values, t, rtol=0, atol=1e-12)


@pytest.mark.parametrize('cut, start', [(2, 'root'), (7, 'root'), (7, 'branchA')])
def test_split_grid_pseudotime_does_not_depend_on_stored_order(scan2d, cut, start, monkeypatch):
    _, nd = _split_pair(scan2d, cut)
    _, shuffled = _split_pair(scan2d, cut, order=np.random.default_rng(1).permutation(len(nd.grids['mapped_grids'])))
    candidates = np.sort(nd.grids['mapped_grids'][nd.grids['mapped_grids_clusters'] == start])
    fixed = np.random.default_rng(4).choice(candidates, 10)
    monkeypatch.setattr(np.random, 'choice', lambda a, size: fixed)   # identical starts whatever the order
    scRL.align_pseudotime(nd, start, boundary=False)
    scRL.align_pseudotime(shuffled, start, boundary=False)
    a, b = nd.grids['pseudotime'], shuffled.grids['pseudotime']
    assert not np.array_equal(a.index, b.index)
    np.testing.assert_allclose(b.loc[a.index].values, a.values, rtol=0, atol=1e-12)


def test_2d_reward_functions_accept_a_single_cluster_name(scan2d):
    import copy
    listed, named = scan2d, copy.deepcopy(scan2d)
    scRL.d_rewards(listed, starts=['root'], ends=['branchA'])
    scRL.d_rewards(named, starts='root', ends='branchA')
    pd.testing.assert_frame_equal(named.qlearning['d_Decision_rewards'], listed.qlearning['d_Decision_rewards'])
    for key in ('starts_cluster_grids', 'ends_cluster_grids'):
        assert list(named.grids[key]) == list(listed.grids[key]) and len(listed.grids[key]) > 0
    assert named.qlearning['reward_key'] == listed.qlearning['reward_key'] == 'branchA'
    scRL.c_rewards(listed, reward_keys=['geneA'], starts=['root'])
    scRL.c_rewards(named, reward_keys=['geneA'], starts='root')
    pd.testing.assert_frame_equal(named.qlearning['c_Decision_rewards'], listed.qlearning['c_Decision_rewards'])
    assert list(named.grids['starts_cluster_grids']) == list(listed.grids['starts_cluster_grids'])


def test_empty_start_name_means_no_start(scan2d):
    nd = lattice_from_scan(scan2d)
    for gres in (scan2d, nd):
        scRL.c_rewards(gres, reward_keys=['geneA'], starts='')
        assert gres.grids['starts_cluster_grids'] is None


INT_CODES = {'root': 0, 'trunk': 1, 'branchA': 2, 'branchB': 3}


@pytest.mark.parametrize('d', [2, 3])
def test_integer_cluster_names(d, scan2d, tree3d):
    import copy
    if d == 2:
        gres, labels = scan2d, make_tree(d=2)[1]
    else:
        gres, labels = scRL.grids_from_embedding(tree3d[0]), tree3d[1]
    scRL.project_cluster(gres, pd.Series(pd.Categorical([INT_CODES[c] for c in labels])))
    seed_all(0); scRL.align_pseudotime(gres, 0, boundary=False)
    listed, named = gres, copy.deepcopy(gres)
    scRL.d_rewards(listed, starts=[0], ends=[2])
    scRL.d_rewards(named, starts=0, ends=[2])
    pd.testing.assert_frame_equal(named.qlearning['d_Decision_rewards'], listed.qlearning['d_Decision_rewards'])
    for key in ('starts_cluster_grids', 'ends_cluster_grids'):
        assert list(named.grids[key]) == list(listed.grids[key]) and len(listed.grids[key]) > 0
    assert named.qlearning['reward_key'] == listed.qlearning['reward_key'] == '2'


def test_project_back_with_equidistant_grid_points():
    from scRL.GridCore import Grids_Results
    from scRL.GridNDCore import project_back_nd
    g = Grids_Results()
    g.grids['grids'] = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [5., 5.]])
    g.grids['mapped_grids'] = np.arange(5)
    g.grids['neighbors'] = np.zeros((5, 4), int)                     # marks an n-D grid
    g.grids['v'] = np.array([0., 1., 2., 3., 100.])
    g.embedding['embedding'] = np.array([[.5, .5], [0., 0.]])
    project_back_nd(g, 'v', neighbors=4, negative=True)
    assert np.isfinite(g.embedding['v']).all() and g.embedding['v'][0] == pytest.approx(1.5)


def test_project_back_unchanged_on_a_3d_grid(tree3d):
    from scipy.spatial import cKDTree
    gres = scRL.grids_from_embedding(tree3d[0])
    scRL.project_cluster(gres, pd.Series(pd.Categorical(tree3d[1])))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    scRL.project_back(gres, 'pseudotime')
    X = gres.embedding['embedding']                                    # the kernel as committed before the fix
    points = gres.grids['grids'][gres.grids['mapped_grids']]
    dist, nn = cKDTree(points).query(X, k=15)
    sigma = dist.std(1, keepdims=True)
    weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
    weight = weight / weight.sum(1, keepdims=True)
    val = (weight * np.asarray(gres.grids['pseudotime'])[nn]).sum(1)
    np.testing.assert_array_equal(gres.embedding['pseudotime'], (val - val.min()) / (val.max() - val.min()))

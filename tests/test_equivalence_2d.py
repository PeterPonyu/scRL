import numpy as np
import pandas as pd
import pytest

import scRL
from scRL.EnvironmentCore import get_state
from scRL.GridNDCore import lattice_from_scan, get_state_nd
from conftest import make_tree, seed_all

LABELS = pd.Series(pd.Categorical(make_tree(d=2)[1]))


def test_cluster_projection_equals_v1(scan2d):
    nd = lattice_from_scan(scan2d)
    del nd.grids['mapped_grids_clusters']
    scRL.project_cluster(nd, LABELS)
    assert np.array_equal(nd.grids['mapped_grids_clusters'], scan2d.grids['mapped_grids_clusters'])


@pytest.mark.parametrize('boundary', [False, True])
def test_pseudotime_equals_v1(scan2d, boundary):
    nd = lattice_from_scan(scan2d)
    np.random.seed(3); scRL.align_pseudotime(scan2d, 'root', boundary=boundary)
    np.random.seed(3); scRL.align_pseudotime(nd, 'root', boundary=boundary)
    a, b = scan2d.grids['pseudotime'], nd.grids['pseudotime']
    assert list(a.index) == list(b.index)
    np.testing.assert_allclose(b.values, a.values, rtol=0, atol=1e-12)


def test_gene_projection_equals_v1(scan2d):
    nd = lattice_from_scan(scan2d)
    genes = make_tree(d=2)[3]
    scRL.project(nd, genes)
    assert list(nd.grids['proj'].columns) == list(scan2d.grids['proj'].columns)
    np.testing.assert_allclose(nd.grids['proj'].values.astype(float),
                               scan2d.grids['proj'].values.astype(float), atol=1e-5)


def test_project_back_equals_v1(scan2d):
    nd = lattice_from_scan(scan2d)
    scRL.project_back(scan2d, 'pseudotime'); scRL.project_back(nd, 'pseudotime')
    np.testing.assert_allclose(nd.embedding['pseudotime'], scan2d.embedding['pseudotime'], atol=1e-5)


def test_state_vectors_equal_v1(scan2d):
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
def test_lineage_rewards_equal_v1(scan2d, mode):
    nd = lattice_from_scan(scan2d)
    scRL.d_rewards(scan2d, starts=['root'], ends=['branchA'], mode=mode)
    scRL.d_rewards(nd, starts=['root'], ends=['branchA'], mode=mode)
    a, b = scan2d.qlearning[f'd_{mode}_rewards'].astype(float), nd.qlearning[f'd_{mode}_rewards']
    assert list(a.index) == list(b.index) and list(a.columns) == list(b.columns)
    np.testing.assert_allclose(b.values, a.values, rtol=0, atol=1e-12)
    assert list(nd.grids['starts_cluster_grids']) == list(scan2d.grids['starts_cluster_grids'])
    assert list(nd.grids['ends_cluster_grids']) == list(scan2d.grids['ends_cluster_grids'])


@pytest.mark.parametrize('kw', [
    dict(reward_keys=['geneA'], starts=['root'], mode='Decision'),
    dict(reward_keys=['geneA'], starts=['root'], starts_keys=['geneRoot'], punish_keys=['geneB'], mode='Contribution'),
    dict(reward_keys=['geneA', 'geneB'], starts_keys=['geneRoot'], mode='Decision'),
])
def test_gene_rewards_equal_v1(scan2d, kw):
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

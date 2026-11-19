import numpy as np
import pytest
from scipy.spatial import cKDTree

import scRL
from scRL.GridNDCore import is_nd
from conftest import make_tree


def test_3d_defaults(tree3d):
    X = tree3d[0]
    g = scRL.grids_from_embedding(X).grids
    assert g['ndim'] == 3 and g['method'] == 'occupancy' and g['n'] == 30
    assert g['actions'] == ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']
    assert np.allclose(g['spacing'], g['spacing'][0])
    assert len(g['mapped_grids']) + len(g['masked_grids']) == int(np.prod(g['shape']))
    assert set(g['mapped_boundary']) | set(g['mapped_travel']) == set(g['mapped_grids'])
    assert not set(g['mapped_boundary']) & set(g['mapped_travel'])
    pts = g['grids'][g['mapped_grids']]
    dist, nn = cKDTree(pts).query(X, k=1)
    assert dist.max() <= g['spacing'][0] * np.sqrt(3) / 2 + 1e-6      # every cell sits next to a kept point
    assert set(g['mapped_grids'][nn]) <= set(g['mapped_travel'])       # ... and that point is a travel point
    assert g['core'].sum() <= len(g['mapped_travel'])


def test_2d_default_is_the_2d_scan(tree2d):
    gres = scRL.grids_from_embedding(tree2d[0], n=20, n_jobs=1)
    assert not is_nd(gres) and gres.grids['n'] == 20 and 'method' not in gres.grids


def test_2d_occupancy_uses_the_8_scan_moves(tree2d):
    g = scRL.grids_from_embedding(tree2d[0], method='occupancy').grids
    assert g['ndim'] == 2 and g['n'] == 50 and g['actions'] == ['R', 'RT', 'T', 'LT', 'L', 'LB', 'B', 'RB']


@pytest.mark.parametrize('d, n_moves', [(4, 8), (5, 10)])
def test_4d_and_5d_build(d, n_moves):
    g = scRL.grids_from_embedding(make_tree(d=d)[0]).grids
    assert g['ndim'] == d and len(g['actions']) == n_moves and len(g['shape']) == d


def test_full_moves_in_3d(tree3d):
    assert len(scRL.grids_from_embedding(tree3d[0], moves='full').grids['actions']) == 26


def test_full_moves_in_4d_warn():
    with pytest.warns(UserWarning, match='80 moves'):
        g = scRL.grids_from_embedding(make_tree(d=4)[0], moves='full').grids
    assert len(g['actions']) == 80


def test_bad_inputs_raise(tree3d, tree2d):
    X = tree3d[0]
    with pytest.raises(ValueError):
        scRL.grids_from_embedding(X[:, :1])
    with pytest.raises(ValueError):
        scRL.grids_from_embedding(np.hstack([X, X]))                    # 6 columns
    with pytest.raises(ValueError):
        scRL.grids_from_embedding(np.where(np.arange(len(X))[:, None] == 3, np.nan, X))
    with pytest.raises(ValueError):
        scRL.grids_from_embedding(np.ones((50, 3)))                     # all cells identical
    with pytest.raises(ValueError):
        scRL.grids_from_embedding(X[:1])                                # a single cell
    with pytest.raises(ValueError):
        scRL.grids_from_embedding(X, method='scan')                     # the scan is 2-D only
    with pytest.raises(ValueError, match='n'):
        scRL.grids_from_embedding(X, n=5000)                            # lattice too large


def test_100k_cells_build_quickly():
    import time
    X = np.repeat(make_tree(n_cells=1000, d=3)[0], 100, axis=0)
    X = X + np.random.default_rng(0).normal(0, .05, X.shape).astype(np.float32)
    t0 = time.time()
    gres = scRL.grids_from_embedding(X)
    assert time.time() - t0 < 20

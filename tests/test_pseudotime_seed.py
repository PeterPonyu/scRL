"""Start points of align_pseudotime: a local generator with ``seed``, the global one without it."""
import numpy as np
import pandas as pd
import pytest

import scRL
from conftest import make_tree

# Pseudotime with seed=None after np.random.seed(7), computed with the code before ``seed`` was added:
# number of grid points, sum, and the values at five evenly spaced positions.
PREVIOUS = {
    (2, False): (179, 97.37333333333333, [0.08666666666666667, 0.21333333333333335, 0.6, 0.8, 1.0]),
    (2, True): (179, 100.27672955974842, [0.03773584905660376, 0.24528301886792447, 0.6226415094339622,
                                          0.8113207547169811, 1.0]),
    (3, False): (2570, 1276.3618090452264, [0.05527638190954775, 0.2663316582914573, 0.5175879396984925,
                                            0.6432160804020102, 0.9798994974874372]),
    (3, True): (2570, 1204.2121212121215, [0.06565656565656568, 0.25757575757575757, 0.4797979797979799,
                                           0.6060606060606062, 0.9343434343434345]),
}


@pytest.fixture(scope='module')
def grid3d():
    X, labels, Z, genes = make_tree(d=3)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    return gres


def _grid(d, scan2d, grid3d):
    return scan2d if d == 2 else grid3d


def _pseudotime(gres, **kw):
    scRL.align_pseudotime(gres, 'root', key_add='pt', **kw)
    return gres.grids['pt'].to_numpy()


@pytest.mark.parametrize('d', [2, 3])
@pytest.mark.parametrize('boundary', [False, True])
def test_seed_none_keeps_the_previous_draw(scan2d, grid3d, d, boundary):
    np.random.seed(7)
    t = _pseudotime(_grid(d, scan2d, grid3d), boundary=boundary)
    n, total, values = PREVIOUS[(d, boundary)]
    assert len(t) == n and t.sum() == pytest.approx(total, abs=1e-9)
    np.testing.assert_allclose(t[np.linspace(0, n - 1, 5).astype(int)], values, rtol=0, atol=1e-12)


@pytest.mark.parametrize('d', [2, 3])
def test_seed_gives_the_same_pseudotime_and_leaves_the_global_generator(scan2d, grid3d, d):
    gres = _grid(d, scan2d, grid3d)
    np.random.seed(11)
    a = _pseudotime(gres, seed=5)
    np.random.seed(99)
    state = np.random.get_state()
    b = _pseudotime(gres, seed=5)
    after = np.random.get_state()
    np.testing.assert_array_equal(a, b)
    assert after[0] == state[0] and np.array_equal(after[1], state[1]) and after[2:] == state[2:]
    differ = [not np.array_equal(a, _pseudotime(gres, seed=s)) for s in range(1, 6)]
    assert any(differ)

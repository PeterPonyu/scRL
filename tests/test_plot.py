import numpy as np
import pandas as pd
import pytest

import scRL
from conftest import make_tree, seed_all


def _gres(d):
    X, labels, Z, _ = make_tree(d=d)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    scRL.project_back(gres, 'pseudotime')
    return gres


def test_3d_views():
    gres = _gres(3)
    ax = scRL.plot_grids(gres)
    assert ax.name == '3d'
    n_travel = len(gres.grids['mapped_travel'])
    ax = scRL.plot_grids(gres, key='pseudotime', title='Pseudotime')
    assert ax.name == '3d' and ax.get_title() == 'Pseudotime'
    assert sum(len(c.get_offsets()) for c in ax.collections) == n_travel
    ax = scRL.plot_grids(gres, key='pseudotime', cells=True, view=(10, 30))
    assert sum(len(c.get_offsets()) for c in ax.collections) == 600
    with pytest.raises(KeyError):
        scRL.plot_grids(gres, key='nope')


def test_2d_scan_result_and_4d(scan2d):
    assert scRL.plot_grids(scan2d, key='pseudotime').name == 'rectilinear'
    assert scRL.plot_grids(_gres(4), key='pseudotime', dims=(0, 1, 3)).name == '3d'


def test_2d_axes_for_a_3d_grid_raise():
    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot()
    with pytest.raises(ValueError, match='3-D axes'):
        scRL.plot_grids(_gres(3), key='pseudotime', ax=ax)


def test_array_in_stored_order_matches_series(scan2d):
    scan2d.grids['pt_array'] = np.asarray(scan2d.grids['pseudotime'])
    a = scRL.plot_grids(scan2d, key='pseudotime').collections[0]
    b = scRL.plot_grids(scan2d, key='pt_array').collections[0]
    a.figure.canvas.draw(); b.figure.canvas.draw()
    assert len(a.get_offsets()) == len(scan2d.grids['mapped_travel'])
    np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
    np.testing.assert_array_equal(a.get_array(), b.get_array())
    np.testing.assert_array_equal(a.get_facecolors(), b.get_facecolors())


def test_interactive():
    pytest.importorskip('plotly')
    fig = scRL.plot_grids(_gres(3), key='pseudotime', interactive=True)
    assert type(fig).__name__ == 'Figure' and fig.data[0].type == 'scatter3d'

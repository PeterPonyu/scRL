"""Fate decision state: the one-step variance of the fate potential, its identity with the uncertainty, its map."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scRL
from scRL.CorridorCore import _lineages, _reaching, _walk
from test_corridors import ENDS, _tree

ARGS = dict(ends=ENDS, start='root', weights='uniform')


def _line(extra_sink=False):
    """Points 0..4 on a line, 0 and 4 absorbing (lineages 0 and 1), symmetric walk."""
    P = np.zeros((5, 5))
    P[0, 1] = P[4, 3] = 1.0
    for p in (1, 2, 3):
        P[p, p - 1] = P[p, p + 1] = .5
    return sparse.csr_matrix(P), np.array([0, -1, -1, -1, 1])


def _identity_error(gres, key_add, forward=.25):
    """Largest |h(1-h) - G d| over the transient points that can arrive, for every lineage."""
    names, owner = _lineages(gres, ENDS)
    P = _walk(gres, gres.grids['neighbors'], forward, None)
    ok = _reaching(P, owner >= 0) & (owner < 0)
    idx = np.flatnonzero(ok)
    G = np.linalg.inv(np.eye(len(idx)) - P[idx][:, idx].toarray())
    H = gres.grids['corridor_fate'].to_numpy()[idx]
    d = gres.grids[f'{key_add}_intensity'].to_numpy()[idx]
    return np.abs(H * (1 - H) - G @ d).max()


def test_line_of_five_by_hand():
    from scRL.CorridorCore import _decision_terms
    P, owner = _line()
    mu = np.array([0, 0, 1., 0, 0])
    H, d, g, ok = _decision_terms(P, owner, mu)
    assert ok.tolist() == [False, True, True, True, False]
    np.testing.assert_allclose(H[:, 1], [0, .25, .5, .75, 1])
    np.testing.assert_allclose(d[:, 1], [0, 1 / 16, 1 / 16, 1 / 16, 0])      # 2 * 1/2 * (1/4)^2
    np.testing.assert_allclose(d[:, 0], d[:, 1])
    np.testing.assert_allclose(g, [0, 1, 2, 1, 0])                           # row of (I-Q)^-1 for the middle point
    np.testing.assert_allclose(g * d[:, 1], [0, 1 / 16, 1 / 8, 1 / 16, 0])
    assert np.isclose((g * d[:, 1]).sum(), .5 * .5)                          # h(1-h) at the start
    G = np.array([[1.5, 1, .5], [1, 2, 1], [.5, 1, 1.5]])
    np.testing.assert_allclose(G @ d[1:4, 1], H[1:4, 1] * (1 - H[1:4, 1]))   # 3/16, 1/4, 3/16


def test_identity_with_mass_that_does_not_arrive():
    """A walk that can enter a part of the grid without end points: the lost mass counts as a third outcome."""
    from scRL.CorridorCore import _decision_terms
    P = sparse.csr_matrix(np.array([[0, 1, 0, 0], [1 / 3, 0, 1 / 3, 1 / 3], [0, 1, 0, 0], [0, 0, 0, 1.]]))
    owner = np.array([0, -1, 1, -1])
    H, d, g, ok = _decision_terms(P, owner, np.array([0, 1., 0, 0]))
    assert ok.tolist() == [False, True, False, False]
    np.testing.assert_allclose(H[1], [1 / 3, 1 / 3])
    np.testing.assert_allclose(d[1], [2 / 9, 2 / 9])           # 1/3 * ((2/3)^2 + (1/3)^2 + (1/3)^2): the lost point counts
    np.testing.assert_allclose(g, [0, 1, 0, 0])
    assert np.allclose(d[1], H[1] * (1 - H[1]))                # G(1,1) = 1
    assert (d[[0, 2, 3]] == 0).all()                           # end points and the trapped point decide nothing


def test_identity_and_zeros_on_the_tree():
    gres = _tree()
    scRL.fate_corridors(gres, ENDS, start='root', weights='uniform')
    scRL.fate_decisions(gres, **ARGS)
    assert _identity_error(gres, 'decision') < 1e-8
    names, owner = _lineages(gres, ENDS)
    d = gres.grids['decision_intensity']
    H = gres.grids['corridor_fate']
    assert (d.to_numpy() >= 0).all() and (d.to_numpy()[owner >= 0] == 0).all()
    for n in names:
        fixed = (H[n].to_numpy() < 1e-9) | (H[n].to_numpy() > 1 - 1e-9)
        assert fixed.any() and d[n].to_numpy()[fixed].max() < 1e-12
    assert (d.to_numpy()[owner < 0] > 0).any()


def test_map_sums_to_outcome_variance_and_two_lineages_agree():
    gres = _tree()
    out = scRL.fate_decisions(gres, **ARGS)
    scRL.fate_corridors(gres, ENDS, start='root', weights='uniform')
    H = gres.grids['corridor_fate'].to_numpy()
    mu = np.asarray(gres.grids['corridor_start'])
    D = gres.grids['decision']
    var = mu @ (H * (1 - H))
    assert np.allclose(H.sum(1)[mu > 0], 1)
    np.testing.assert_allclose(D.sum().to_numpy(), var, atol=1e-8)
    np.testing.assert_allclose(out['variance'].to_numpy(), var, atol=1e-8)
    assert np.allclose(gres.grids['decision_total'], D.sum(1))
    d = gres.grids['decision_intensity']
    np.testing.assert_allclose(d['A'], d['B'], atol=1e-10)
    np.testing.assert_allclose(D['A'], D['B'], atol=1e-10)


def test_stored_keys_and_same_numbers_with_or_without_corridors():
    a = _tree()
    out = scRL.fate_decisions(a, **ARGS)
    assert scRL.fate_decisions is scRL.CorridorCore.fate_decisions
    mapped = np.asarray(a.grids['mapped_grids'])
    M = len(mapped)
    for key in ('decision', 'decision_intensity'):
        t = a.grids[key]
        assert isinstance(t, pd.DataFrame) and t.shape == (M, 2) and list(t.columns) == ['A', 'B']
        assert (t.index == mapped).all()
    for key in ('decision_A', 'decision_B', 'decision_intensity_A', 'decision_total', 'decision_start'):
        assert np.asarray(a.grids[key]).shape == (M,)
    n_cells = len(a.embedding['embedding'])
    for key in ('decision_A', 'decision_intensity_B', 'decision_total'):
        assert np.asarray(a.embedding[key]).shape == (n_cells,)
    assert a.grids['decision_settings']['names'] == ['A', 'B']
    assert list(out.index) == ['A', 'B'] and out.index.name == 'lineage'
    assert list(out.columns) == ['probability', 'variance', 'start_points']
    b = _tree()
    scRL.fate_corridors(b, ENDS, start='root', weights='uniform')
    out_b = scRL.fate_decisions(b, **ARGS)
    pd.testing.assert_frame_equal(a.grids['decision'], b.grids['decision'])
    pd.testing.assert_frame_equal(a.grids['decision_intensity'], b.grids['decision_intensity'])
    pd.testing.assert_frame_equal(out, out_b)
    corr = scRL.fate_corridors(b, ENDS, start='root', weights='uniform', key_add='c2')
    np.testing.assert_allclose(out.probability.to_numpy(), corr.probability.to_numpy())


def test_errors():
    gres = _tree()
    with pytest.raises(ValueError, match='at least two lineages'):
        scRL.fate_decisions(gres, {'A': 'tipA'}, start='root')
    with pytest.raises(ValueError, match='nowhere'):
        scRL.fate_decisions(gres, {'A': 'tipA', 'B': 'nowhere'}, start='root')
    with pytest.raises(ValueError, match='two lineages'):
        scRL.fate_decisions(gres, {'A': 'tipA', 'B': 'tipA'}, start='root')
    with pytest.raises(ValueError, match='reserved'):
        scRL.fate_decisions(gres, {'top': 'tipA', 'B': 'tipB'}, start='root')
    with pytest.raises(ValueError, match='reserved'):
        scRL.fate_decisions(gres, {'total': 'tipA', 'B': 'tipB'}, start='root')
    with pytest.raises(ValueError, match='label no kept grid point'):
        scRL.fate_decisions(gres, ENDS, start='nowhere')
    with pytest.raises(ValueError, match='weights'):
        scRL.fate_decisions(gres, ENDS, start='root', weights='volume')
    with pytest.warns(UserWarning, match='end cluster'):
        scRL.fate_decisions(gres, ENDS, start=['root', 'tipA'], key_add='x')
    from conftest import make_tree
    X, labels, Z, genes = make_tree(d=3)
    bare = scRL.grids_from_embedding(X)
    with pytest.raises((KeyError, ValueError), match='project_cluster'):
        scRL.fate_decisions(bare, ENDS, start='root')
    scRL.project_cluster(bare, pd.Series(pd.Categorical(labels)))
    with pytest.raises(KeyError, match='align_pseudotime'):
        scRL.fate_decisions(bare, {'A': 'branchA', 'B': 'branchB'}, start='root')


def test_two_dimensional_scan_grid(scan2d):
    ends = {'A': 'branchA', 'B': 'branchB'}
    scRL.fate_corridors(scan2d, ends, start='root')
    out = scRL.fate_decisions(scan2d, ends, start='root')
    H = scan2d.grids['corridor_fate'].to_numpy()
    mu = np.asarray(scan2d.grids['corridor_start'])
    np.testing.assert_allclose(scan2d.grids['decision'].sum().to_numpy(), mu @ (H * (1 - H)), atol=1e-8)
    assert (scan2d.grids['decision_intensity'].to_numpy() >= 0).all() and out.variance.gt(0).all()


@pytest.mark.parametrize('key', ['decision_A', 'decision_intensity_B', 'decision_total'])
def test_the_grid_view_colours_by_the_decision_map(key):
    gres = _tree()
    scRL.fate_decisions(gres, **ARGS)
    ax = scRL.plot_grids(gres, key=key)
    assert ax.name == '3d' and len(ax.figure.axes) == 2                      # panel and colour bar
    plt.close('all')
    ax = scRL.plot_grids(gres, key=key, cells=True)
    assert len(ax.figure.axes) == 2
    plt.close('all')

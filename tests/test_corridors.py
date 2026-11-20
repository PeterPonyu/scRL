"""Fate corridors from a start region: exactness against sampled walks, shares, profiles and views."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import scRL
from scRL.CorridorCore import _lineages, _walk
from conftest import make_tree, seed_all

ENDS = {'A': 'tipA', 'B': 'tipB'}


def _tree(d=3, n=None):
    X, labels, Z, genes = make_tree(d=d)
    labels = labels.astype(object)
    far = np.linalg.norm(X - np.array([6, 0, 0][:d]), axis=1) > 3.5          # the far half of each branch
    labels[(labels == 'branchA') & far] = 'tipA'
    labels[(labels == 'branchB') & far] = 'tipB'
    gres = scRL.grids_from_embedding(X) if n is None else scRL.grids_from_embedding(X, n=n)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels.astype(str))))
    seed_all(0)
    scRL.align_pseudotime(gres, 'root', boundary=False)
    return gres


def test_corridors_match_sampled_walks_and_shares():
    gres = _tree()
    out = scRL.fate_corridors(gres, ENDS, start='root', weights='uniform')
    assert np.isclose(out.probability.sum(), 1, atol=1e-6)
    names, owner = _lineages(gres, ENDS)
    from scRL.GridNDCore import is_nd
    P = _walk(gres, gres.grids['neighbors'], .25, None)
    mu = np.asarray(gres.grids['corridor_start'])
    rng = np.random.default_rng(0)
    M = len(mu)
    visits = np.zeros((M, 2))
    ends = np.zeros(2)
    indptr, indices, data = P.indptr, P.indices, P.data
    for _ in range(20000):
        p = rng.choice(M, p=mu)
        path = [p]
        while owner[p] < 0:
            lo, hi = indptr[p], indptr[p + 1]
            p = indices[lo + np.searchsorted(np.cumsum(data[lo:hi]), rng.random() * data[lo:hi].sum())]
            path.append(p)
        lin = owner[p]
        ends[lin] += 1
        np.add.at(visits[:, lin], path, 1)
    for i, n in enumerate(names):
        sampled = visits[:, i] / visits[:, i].sum()
        exact = np.asarray(gres.grids[f'corridor_{n}'])
        exact = exact / exact.sum()
        assert np.abs(sampled - exact).max() < .02
        assert abs(ends[i] / ends.sum() - out.probability.iloc[i]) < .02
    fate = gres.grids['corridor_fate']
    shares = scRL.fate_probabilities(gres, ENDS, steps=20000)
    assert np.abs(fate.to_numpy() - shares.to_numpy()).max() < 1e-6
    for i, n in enumerate(names):                     # entering the lineage: probabilities that sum to 1
        assert np.isclose(np.asarray(gres.grids[f'corridor_{n}'])[owner == i].sum(), 1)


def test_potency_and_profile_on_the_tree():
    gres = _tree()
    scRL.fate_corridors(gres, ENDS, start='root')
    pot = np.asarray(gres.grids['corridor_potency'], dtype=float)
    labels = np.asarray(gres.grids['mapped_grids_clusters']).astype(str)
    assert np.nanmedian(pot[labels == 'root']) > 1.8 and np.nanmax(pot[np.isin(labels, ['tipA', 'tipB'])]) == 1
    prof = scRL.corridor_profile(gres)
    ov = prof['overlap'].loc['A-B'].to_numpy()
    defined = ov[np.isfinite(ov)]
    assert defined[0] > .8 and defined[-1] < .2           # one route at the start, two at the end
    pair = prof['pairs'].iloc[0]
    assert 0 < pair.overlap < 1 and np.isfinite(pair.divergence)
    st = prof['stages']
    assert (st.width.dropna() >= 0).all() and set(st.lineage) == {'A', 'B'}
    assert st.groupby('lineage').mass.sum().round(6).eq(1).all()
    assert prof['tree']['newick'] in ('(A,B);', '(B,A);')
    dec = prof['decisions'].set_index('lineage')
    t = gres.grids['pseudotime'].loc[np.asarray(gres.grids['mapped_grids'])].to_numpy(float)
    assert dec.pseudotime.between(np.nanmin(t), np.nanmax(t)).all() and (dec.share > 0).all()


def test_start_region_inputs_and_errors():
    gres = _tree()
    a = scRL.fate_corridors(gres, ENDS, start='root', weights='cells')
    b = scRL.fate_corridors(gres, ENDS, start='root', weights='uniform', key_add='uni')
    assert not np.allclose(gres.grids['corridor_start'], gres.grids['uni_start'])
    mask = np.asarray(gres.grids['corridor_potency'], dtype=float) >= 1.9
    c = scRL.fate_corridors(gres, ENDS, start=mask, key_add='multi')
    assert (np.asarray(gres.grids['multi_start'])[~mask] == 0).all()
    with pytest.raises(ValueError, match='label no kept grid point'):
        scRL.fate_corridors(gres, ENDS, start='nowhere')
    with pytest.warns(UserWarning, match='end cluster'):
        scRL.fate_corridors(gres, ENDS, start=['root', 'tipA'], key_add='x')
    with pytest.raises(ValueError, match="weights"):
        scRL.fate_corridors(gres, ENDS, start='root', weights='volume')
    with pytest.raises(KeyError, match='fate_corridors'):
        scRL.corridor_profile(gres, key_add='missing')


@pytest.mark.parametrize('d', [2, 3])
def test_corridor_views(d):
    gres = _tree(d=d)
    scRL.fate_corridors(gres, ENDS, start='root')
    ax = scRL.plot_corridors(gres)
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legend, panel = ax.get_legend().get_window_extent(r), ax.get_window_extent(r)
    assert legend.x0 >= panel.x1 - 1 and legend.x1 <= fig.bbox.x1 + 1
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[:2] == ['A', 'B'] and 'Start region' in labels
    plt.close('all')
    with pytest.raises(ValueError, match='not in the corridors'):
        scRL.plot_corridors(gres, lineages=['C'])
    plt.close('all')

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


def _panels(fig):
    return [ax for ax in fig.axes if ax.get_label() != '<colorbar>' and ax.get_visible()]


def _points(axes):
    return sum(len(c.get_offsets()) for ax in axes for c in ax.collections)


def test_plot_slices():
    gres = _gres(3)
    fig = scRL.plot_slices(gres, key='pseudotime', n_slices=5)
    panels = _panels(fig)
    assert len(panels) == 5
    assert _points(panels) == len(gres.grids['mapped_travel'])
    assert len({ax.get_xlim() for ax in panels}) == 1 and len({ax.get_ylim() for ax in panels}) == 1
    fig = scRL.plot_slices(gres, key='pseudotime', cells=True, n_slices=4, axis=0)
    assert len(_panels(fig)) == 4 and _points(_panels(fig)) == 600
    fig = scRL.plot_slices(gres, n_slices=3)
    assert len(_panels(fig)) == 3 and _points(_panels(fig)) == len(gres.grids['mapped_grids'])


def test_plot_slices_rejects_2d(scan2d):
    with pytest.raises(ValueError):
        scRL.plot_slices(scan2d, key='pseudotime')


def test_plot_views():
    views = ((10, 20), (30, -40), (90, -90))
    fig = scRL.plot_views(_gres(3), key='pseudotime', views=views)
    panels = [ax for ax in fig.axes if ax.name == '3d']
    assert [(ax.elev, ax.azim) for ax in panels] == list(views)


def _bin_medians(x, y, bins):
    edges = np.linspace(x.min(), x.max(), bins + 1)
    k = np.minimum(((x - x.min()) / (x.max() - x.min()) * bins).astype(int), bins - 1)
    return np.array([np.median(y[k == b]) for b in range(bins) if (k == b).any()])


def test_plot_trend():
    gres = _gres(3)
    gres.embedding['same'] = np.asarray(gres.embedding['pseudotime']).copy()
    ax = scRL.plot_trend(gres, 'same', bins=10)
    y = ax.lines[0].get_ydata()
    expected = _bin_medians(gres.embedding['pseudotime'], gres.embedding['same'], 10)
    np.testing.assert_allclose(y, expected)
    assert (np.diff(y) > 0).all()
    gres.embedding['other'] = 1 - gres.embedding['same']
    ax = scRL.plot_trend(gres, ['same', 'other'], labels=['up', 'down'])
    assert len(ax.lines) == 2 and [t.get_text() for t in ax.get_legend().get_texts()] == ['up', 'down']


def test_plot_clusters():
    gres = _gres(3)
    ax = scRL.plot_clusters(gres, 'pseudotime')
    v = pd.Series(gres.embedding['pseudotime']).groupby(np.asarray(gres.embedding['clusters'])).median()
    assert [t.get_text() for t in ax.get_xticklabels()] == list(v.sort_values(ascending=False).index)


def _policy_trainer(gres, algo, d):
    Z = make_tree(d=d)[2]
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    return scRL.trainer(algo, gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=10)


def _spacing(gres):
    G, shape = gres.grids['grids'], gres.grids.get('shape', (gres.grids['n'],) * 2)
    strides = [int(np.prod(shape[k + 1:])) for k in range(len(shape))]
    return np.array([G[s, k] - G[0, k] for k, s in enumerate(strides)])


def _offsets(gres):
    from scRL.GridNDCore import OFFSETS_2D
    return np.asarray(gres.grids['offsets']) if 'offsets' in gres.grids else np.array(OFFSETS_2D)


def _arrows(ax, three):
    if three:
        n = len(ax.collections[-1]._segments3d) // 3
        return np.array([seg[0] - seg[1] for seg in ax.collections[-1]._segments3d[:n]])   # shaft: tip, tail
    q = ax.collections[-1]
    return np.column_stack([q.U, q.V])


@pytest.mark.parametrize('d', [2, 3])
def test_plot_policy(d, scan2d):
    import torch
    gres = scan2d if d == 2 else _gres(3)
    move = 2
    expected = _offsets(gres)[move] * _spacing(gres)
    n_travel = len(gres.grids['mapped_travel'])

    t = _policy_trainer(gres, 'ActorCritic', d)
    A = len(_offsets(gres))
    t.agent.actor = lambda x: torch.nn.functional.one_hot(torch.full((len(x),), move, device=x.device), A).float()
    arrows = _arrows(scRL.plot_policy(gres, t), d == 3)
    assert len(arrows) == n_travel
    np.testing.assert_allclose(arrows, np.tile(expected, (n_travel, 1)), atol=1e-6)

    t = _policy_trainer(gres, 'TabularQ', d)
    t.agent.Q.loc[:, :] = 0.0
    t.agent.Q.loc[:, move] = 1.0
    arrows = _arrows(scRL.plot_policy(gres, t, every=2), d == 3)
    assert len(arrows) == len(range(0, n_travel, 2))
    np.testing.assert_allclose(arrows, np.tile(expected, (len(arrows), 1)), atol=1e-6)


def test_plot_training():
    from scRL.utils import moving_average
    r = np.random.default_rng(0).normal(size=300)
    ax = scRL.plot_training(r, window=11)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), moving_average(r, 11))
    assert len(ax.figure.axes) == 1
    ax = scRL.plot_training(r, values=np.arange(1000.0), window=10)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.convolve(r, np.ones(10) / 10, mode='valid'))
    assert len(ax.figure.axes) == 2

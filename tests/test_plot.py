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
    if d == 3:
        expected = .8 * expected                  # 3-D arrows show the direction, 0.8 grid steps long
    arrows = _arrows(scRL.plot_policy(gres, t, every=1), d == 3)
    assert len(arrows) == n_travel
    np.testing.assert_allclose(arrows, np.tile(expected, (n_travel, 1)), atol=1e-6)

    t = _policy_trainer(gres, 'TabularQ', d)
    t.agent.Q.loc[:, :] = 0.0
    t.agent.Q.loc[:, move] = 1.0
    arrows = _arrows(scRL.plot_policy(gres, t, every=2), d == 3)
    assert len(arrows) == len(_thinned(gres, range(d), 2))
    np.testing.assert_allclose(arrows, np.tile(expected * (2 if d == 3 else 1), (len(arrows), 1)), atol=1e-6)


def _lattice(gres, flat):
    shape = gres.grids.get('shape', (gres.grids['n'],) * 2)
    return np.array(np.unravel_index(np.asarray(flat), shape)).T


def _thinned(gres, dims, k):
    """Travel points whose lattice coordinates along dims are multiples of k."""
    travel = np.asarray(gres.grids['mapped_travel'])
    return travel[(_lattice(gres, travel)[:, list(dims)] % k == 0).all(1)]


def test_plot_training():
    from scRL.utils import moving_average
    r = np.random.default_rng(0).normal(size=300)
    ax = scRL.plot_training(r, window=11)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), moving_average(r, 11))
    assert len(ax.figure.axes) == 1
    ax = scRL.plot_training(r, values=np.arange(300.0), window=10)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.convolve(r, np.ones(10) / 10, mode='valid'))
    assert len(ax.figure.axes) == 2
    fig = scRL.plot_training(r, values=np.arange(1000.0), window=11)
    returns_ax, values_ax = fig.axes
    assert returns_ax.get_xlabel() == 'Episode' and values_ax.get_xlabel() == 'Step'
    np.testing.assert_allclose(returns_ax.lines[0].get_ydata(), moving_average(r, 11))
    np.testing.assert_allclose(values_ax.lines[0].get_ydata(), np.arange(1000.0))
    with pytest.raises(ValueError):
        scRL.plot_training(r, values=np.arange(1000.0), ax=returns_ax)
    _assert_layout(fig)


def _text_boxes(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for ax in fig.axes:
        candidates = [ax.title, ax.xaxis.label, ax.yaxis.label] + ([ax.zaxis.label] if ax.name == '3d' else [])
        if ax.get_label() == '<colorbar>':
            candidates += ax.get_yticklabels()
        texts += [t for t in candidates if t.get_visible() and t.get_text()]
    return [(t.get_text(), t.get_window_extent(renderer)) for t in texts]


def _assert_layout(fig):
    boxes = _text_boxes(fig)
    W, H = fig.canvas.get_width_height()
    for text, b in boxes:
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= W and b.y1 <= H, f'{text!r} is cut off: {b}'
    for i, (t1, b1) in enumerate(boxes):
        for t2, b2 in boxes[i + 1:]:
            assert not b1.overlaps(b2), f'{t1!r} overlaps {t2!r}'


def _stretched(scale):
    X, labels, _, _ = make_tree(d=3)
    gres = scRL.grids_from_embedding(X * np.array(scale, dtype=np.float32))
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    return gres


SCALES = [(1, 1, 1), (1, 3, 1), (3, 1, 1), (1, 2, 2), (1, 1, 3)]


@pytest.mark.parametrize('scale', SCALES)
def test_plot_views_labels_fit_and_do_not_overlap(scale):
    fig = scRL.plot_views(_stretched(scale), key='pseudotime')
    _assert_layout(fig)
    top = [ax for ax in fig.axes if ax.name == '3d'][2]
    assert not top.zaxis.label.get_visible()


@pytest.mark.parametrize('scale', SCALES)
@pytest.mark.parametrize('view', [(22, -58), (20, 30), (90, -90), (10, 120)])
def test_plot_grids_3d_labels_fit(view, scale):
    _assert_layout(scRL.plot_grids(_stretched(scale), key='pseudotime', view=view).figure)


def test_plot_policy_3d_default_sub_lattice():
    gres = _gres(3)
    t = _policy_trainer(gres, 'TabularQ', 3)
    t.agent.Q.loc[:, :] = 0.0
    t.agent.Q.loc[:, 0] = 1.0
    kept = _lattice(gres, gres.grids['mapped_grids'])
    k = max(1, int(np.ceil((np.ptp(kept, axis=0).max() + 1) / 15)))
    ax = scRL.plot_policy(gres, t)
    arrows = _arrows(ax, True)
    assert len(arrows) == len(_thinned(gres, range(3), k))
    np.testing.assert_allclose(arrows, np.tile(.8 * k * _offsets(gres)[0] * _spacing(gres), (len(arrows), 1)),
                               atol=1e-6)
    assert ax.collections[-1].get_linewidth()[0] > 0


def _slices_with_policy(move):
    import torch
    gres = _gres(3)
    t = _policy_trainer(gres, 'ActorCritic', 3)
    m = gres.grids['actions'].index(move)
    t.agent.actor = lambda x: torch.nn.functional.one_hot(torch.full((len(x),), m, device=x.device), 6).float()
    return gres, scRL.plot_slices(gres, key='pseudotime', axis=2, n_slices=4, policy=t, policy_every=2)


def test_plot_slices_policy_in_plane():
    from matplotlib.quiver import Quiver
    gres, fig = _slices_with_policy('x+')
    h = _spacing(gres)
    total = 0
    for ax in _panels(fig):
        q = [c for c in ax.collections if isinstance(c, Quiver)]
        assert len(q) == 1
        np.testing.assert_allclose(np.column_stack([q[0].U, q[0].V]), np.tile([.8 * 2 * h[0], 0], (len(q[0].U), 1)))
        assert not [l for l in ax.lines if l.get_marker() in ('^', 'v')]
        total += len(q[0].U)
    assert total == len(_thinned(gres, (0, 1), 2))
    _assert_layout(fig)


def test_plot_slices_policy_across():
    from matplotlib.quiver import Quiver
    gres, fig = _slices_with_policy('z+')
    total = 0
    for ax in _panels(fig):
        assert not [c for c in ax.collections if isinstance(c, Quiver)]
        markers = [l for l in ax.lines if l.get_marker() in ('^', 'v')]
        assert {l.get_marker() for l in markers} == {'^'}
        total += sum(len(l.get_xdata()) for l in markers)
    assert total == len(_thinned(gres, (0, 1), 2))


def test_plot_slices_policy_default_thinning():
    from matplotlib.quiver import Quiver
    gres, _ = _slices_with_policy('x+')
    t = _policy_trainer(gres, 'TabularQ', 3)
    t.agent.Q.loc[:, :] = 0.0
    t.agent.Q.loc[:, 0] = 1.0
    fig = scRL.plot_slices(gres, key='pseudotime', policy=t)
    for ax in _panels(fig):
        for q in [c for c in ax.collections if isinstance(c, Quiver)]:
            assert len(np.unique(np.round(q.X, 6))) <= 16 and len(np.unique(np.round(q.Y, 6))) <= 16
    _assert_layout(fig)


def test_plot_trend_restricted_to_clusters():
    gres = _gres(3)
    clusters = np.asarray(gres.embedding['clusters'])
    pt = np.asarray(gres.embedding['pseudotime'])
    inside = np.isin(clusters, ['root', 'branchA'])
    gres.embedding['lineage'] = np.where(inside, pt, 0.0)
    ax = scRL.plot_trend(gres, 'lineage', bins=8, clusters=['root', 'branchA'])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), _bin_medians(pt[inside], pt[inside], 8))
    with pytest.raises(ValueError, match='branchA'):
        scRL.plot_trend(gres, 'lineage', clusters=['nowhere'])


def _drawn_rgb(ax):
    """Face colours (RGB) of the first scatter, as a sorted multiset: 3-D scatters reorder by depth."""
    ax.figure.canvas.draw()
    rgb = ax.collections[0].get_facecolors()[:, :3]
    return rgb[np.lexsort(rgb.T)]


def test_cluster_colouring():
    from matplotlib.colors import to_rgba_array
    gres = _gres(3)
    names = list(gres.embedding['clusters'].cat.categories)
    ax = scRL.plot_grids(gres, key='clusters')
    travel = gres.grids['mapped_travel']
    pos = {g: p for p, g in enumerate(gres.grids['mapped_grids'])}
    stored = to_rgba_array(np.asarray(gres.grids['mapped_grids_colors'])[[pos[g] for g in travel]])[:, :3]
    np.testing.assert_allclose(_drawn_rgb(ax), stored[np.lexsort(stored.T)])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == names
    ax = scRL.plot_grids(gres, key='clusters', cells=True)
    stored = to_rgba_array(gres.embedding['cluster_colors'])[:, :3]
    np.testing.assert_allclose(_drawn_rgb(ax), stored[np.lexsort(stored.T)])
    for fig in (scRL.plot_slices(gres, key='clusters', n_slices=3), scRL.plot_views(gres, key='clusters')):
        assert len(fig.legends) == 1 and [t.get_text() for t in fig.legends[0].get_texts()] == names
        assert not [ax for ax in fig.axes if ax.get_label() == '<colorbar>']


def test_interactive_key_menu():
    pytest.importorskip('plotly')
    fig = scRL.plot_grids(_gres(3), key=['pseudotime', 'clusters'], interactive=True)
    assert len(fig.data) == 2 and [tr.visible for tr in fig.data] == [True, False]
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 2 and buttons[1].args[0]['visible'] == [False, True]


def test_default_cluster_colours_on_nd_grids():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba_array
    X = make_tree(d=3)[0]
    gres = scRL.grids_from_embedding(X)
    labels = pd.Series(pd.Categorical([f'c{i % 19:02d}' for i in range(len(X))]))
    scRL.project_cluster(gres, labels)
    colours = np.unique(to_rgba_array(gres.embedding['cluster_colors'])[:, :3], axis=0)
    tab20 = to_rgba_array(plt.get_cmap('tab20').colors[:19])[:, :3]
    assert len(colours) == 19
    np.testing.assert_allclose(np.unique(tab20, axis=0), colours)


def _pixel_table(gres, key, collapse, how):
    travel = np.asarray(gres.grids['mapped_travel'])
    c = _lattice(gres, travel)
    shown = [d for d in range(3) if d != collapse]
    pos = {g: p for p, g in enumerate(gres.grids['mapped_grids'])}
    v = np.asarray(gres.grids[key])[[pos[g] for g in travel]]
    frame = pd.DataFrame({'a': c[:, shown[0]], 'b': c[:, shown[1]], 'v': v})
    return frame.groupby(['a', 'b'])['v'].agg(how), c[:, shown].min(0)


@pytest.mark.parametrize('how', ['max', 'mean'])
def test_plot_projections(how):
    gres = _gres(3)
    gres.grids['synthetic'] = np.random.default_rng(0).normal(size=len(gres.grids['mapped_grids']))
    fig = scRL.plot_projections(gres, key='synthetic', how=how)
    panels = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    assert len(panels) == 3
    for collapse, ax in enumerate(panels):
        image = ax.collections[0].get_array()
        table, lo = _pixel_table(gres, 'synthetic', collapse, how)
        assert image.count() == len(table)
        for (a, b), v in table.items():
            assert image[b - lo[1], a - lo[0]] == pytest.approx(v)
    _assert_layout(fig)
    for key in (None, 'clusters'):
        _assert_layout(scRL.plot_projections(gres, key=key))


def test_plot_projections_rejects_2d(scan2d):
    with pytest.raises(ValueError):
        scRL.plot_projections(scan2d, key='pseudotime')


def test_save_rotation(tmp_path):
    from PIL import Image
    path = scRL.save_rotation(_gres(3), tmp_path / 'turn.gif', key='pseudotime', frames=6, dpi=40)
    assert Image.open(path).n_frames == 6
    with pytest.raises(ValueError):
        scRL.save_rotation(_gres(3), tmp_path / 'turn.avi')


def test_plot_report():
    gres = _gres(3)
    fig = scRL.plot_report(gres, 'pseudotime', clusters=['root', 'trunk', 'branchA'], title='Pseudotime')
    kinds = [ax.name for ax in fig.axes]
    assert kinds.count('3d') == 1 and len(fig.axes) == 13
    assert len([ax for ax in fig.axes if ax.get_label() == '<colorbar>']) == 1
    _assert_layout(fig)


def test_plot_style_is_scoped():
    import matplotlib as mpl
    before = {k: mpl.rcParams[k] for k in ('font.size', 'pdf.fonttype', 'savefig.dpi', 'font.family')}
    with scRL.plot_style(font='Arial', base_size=7):
        assert mpl.rcParams['font.size'] == 7 and mpl.rcParams['pdf.fonttype'] == 42
        assert mpl.rcParams['ps.fonttype'] == 42 and mpl.rcParams['svg.fonttype'] == 'none'
        assert mpl.rcParams['axes.linewidth'] == .6 and mpl.rcParams['savefig.dpi'] == 300
        assert mpl.rcParams['font.family'] == ['Arial', 'DejaVu Sans']
    assert {k: mpl.rcParams[k] for k in before} == before
    with pytest.raises(RuntimeError):
        with scRL.plot_style(base_size=12):
            raise RuntimeError('inside')
    assert {k: mpl.rcParams[k] for k in before} == before


def test_interactive_policy():
    import torch
    pytest.importorskip('plotly')
    gres = _gres(3)
    t = _policy_trainer(gres, 'ActorCritic', 3)
    t.agent.actor = lambda x: torch.nn.functional.one_hot(torch.full((len(x),), 2, device=x.device), 6).float()
    fig = scRL.plot_policy(gres, t, key='pseudotime', interactive=True, every=2)
    assert [tr.type for tr in fig.data] == ['scatter3d', 'cone']
    cone = fig.data[1]
    move = _offsets(gres)[2] / np.linalg.norm(_offsets(gres)[2])
    np.testing.assert_allclose(np.column_stack([cone.u, cone.v, cone.w]), np.tile(move, (len(cone.u), 1)))
    assert len(cone.u) == len(_thinned(gres, range(3), 2))


def test_interactive_policy_needs_3d(scan2d):
    t = _policy_trainer(scan2d, 'TabularQ', 2)
    with pytest.raises(ValueError, match='3-D'):
        scRL.plot_policy(scan2d, t, interactive=True)


def test_slices_and_views_fit():
    gres = _gres(3)
    for fig in (scRL.plot_slices(gres, key='pseudotime'), scRL.plot_slices(gres, key='clusters'),
                scRL.plot_slices(gres, key='pseudotime', cells=True), scRL.plot_views(gres, key='pseudotime')):
        _assert_layout(fig)


def _string_key(gres, long=False):
    """A string key on the grid: which third of the first axis each kept point lies in."""
    G = gres.grids['grids'][gres.grids['mapped_grids']]
    cuts = np.quantile(G[:, 0], [1 / 3, 2 / 3])
    names = ['early', 'middle', 'late']
    if long:
        names[2] = 'a category name of thirty chars'
    gres.grids['part'] = np.array(names, dtype=object)[np.searchsorted(cuts, G[:, 0])]
    return names


def test_categorical_key_with_stored_colours():
    from matplotlib.colors import to_rgba
    gres = _gres(3)
    _string_key(gres)
    gres.grids['part_colors'] = {'late': '#ff0000', 'early': '#00ff00', 'middle': '#0000ff'}
    ax = scRL.plot_grids(gres, key='part')
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts == ['late', 'early', 'middle']
    for h in ax.get_legend().legend_handles:
        assert to_rgba(h.get_color()) == to_rgba(gres.grids['part_colors'][h.get_label()])
    travel = gres.grids['mapped_travel']
    pos = {g: p for p, g in enumerate(gres.grids['mapped_grids'])}
    expected = np.array([to_rgba(gres.grids['part_colors'][gres.grids['part'][pos[g]]])[:3] for g in travel])
    np.testing.assert_allclose(_drawn_rgb(ax), expected[np.lexsort(expected.T)])


def test_categorical_key_default_colours_and_cells():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    gres = _gres(3)
    gres.embedding['which'] = np.where(np.asarray(gres.embedding['clusters']) == 'root', 'start', 'rest')
    ax = scRL.plot_grids(gres, key='which', cells=True)
    handles = ax.get_legend().legend_handles
    assert [h.get_label() for h in handles] == ['rest', 'start']
    assert [to_rgba(h.get_color()) for h in handles] == [to_rgba(c) for c in plt.get_cmap('tab10').colors[:2]]


def test_vcenter_and_explicit_limits():
    gres = _gres(3)
    gres.grids['shifted'] = np.asarray(gres.grids['pseudotime']) - .2
    pos = {g: p for p, g in enumerate(gres.grids['mapped_grids'])}
    drawn = gres.grids['shifted'][[pos[g] for g in gres.grids['mapped_travel']]]
    ax = scRL.plot_grids(gres, key='shifted', vcenter=0)
    lo, hi = ax.collections[-1].get_clim()
    assert lo == -hi and hi == pytest.approx(np.abs(drawn).max())
    ax = scRL.plot_grids(gres, key='shifted', vcenter=0, vmin=-.1)
    assert ax.collections[-1].get_clim() == (-.1, pytest.approx(np.abs(drawn).max()))
    fig = scRL.plot_slices(gres, key='shifted', vcenter=0)
    lo, hi = [c for ax in fig.axes for c in ax.collections if c.get_array() is not None][0].get_clim()
    assert lo == -hi


def test_missing_values_are_grey_and_do_not_set_limits():
    from matplotlib.colors import to_rgba
    gres = _gres(3)
    values = np.asarray(gres.grids['pseudotime'], dtype=float).copy()
    values[::3] = np.nan
    values[1] = 100.0 if np.isnan(values[1]) else values[1]
    gres.grids['holes'] = values
    ax = scRL.plot_grids(gres, key='holes')
    grey, coloured = ax.collections[0], ax.collections[1]
    travel = gres.grids['mapped_travel']
    pos = {g: p for p, g in enumerate(gres.grids['mapped_grids'])}
    n_missing = int(np.isnan(values[[pos[g] for g in travel]]).sum())
    assert len(grey.get_offsets()) == n_missing > 0
    assert to_rgba(grey.get_facecolor()[0]) == to_rgba('#d9d9d9')
    finite = values[[pos[g] for g in travel]]
    finite = finite[~np.isnan(finite)]
    assert coloured.get_clim() == (finite.min(), finite.max())
    gres.grids['empty'] = np.full(len(values), np.nan)
    with pytest.raises(ValueError):
        scRL.plot_grids(gres, key='empty')
    _assert_layout(scRL.plot_projections(gres, key='holes'))


def test_categorical_projection_and_layouts():
    gres = _gres(3)
    _string_key(gres, long=True)
    fig = scRL.plot_projections(gres, key='part')
    assert [t.get_text() for t in fig.legends[0].get_texts()] == sorted(set(gres.grids['part']))
    for fig in (fig, scRL.plot_views(gres, key='part'), scRL.plot_slices(gres, key='part')):
        _assert_layout(fig)
        assert not [ax for ax in fig.axes if ax.get_label() == '<colorbar>']

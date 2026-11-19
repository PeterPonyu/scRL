import numpy as np
import pandas as pd

import scRL
from scRL.GridCore import Grids_Results
from scRL.GridNDCore import attach_lattice
from conftest import make_tree, seed_all


def test_pseudotime_span_is_the_unscaled_range():
    from scipy.sparse.csgraph import dijkstra
    shape = (4, 7)
    g = Grids_Results()
    g.grids['grids'] = np.indices(shape).reshape(2, -1).T.astype(float)
    g.grids['mapped_grids'] = np.arange(int(np.prod(shape)))
    attach_lattice(g, shape, 'axis')
    clusters = np.array(['other'] * len(g.grids['mapped_grids']), dtype=object)
    clusters[[0, 1, 7]] = 'start'
    g.grids['mapped_grids_clusters'] = clusters
    np.random.seed(0)
    starts = np.random.choice(np.array([0, 1, 7]), 10)
    t = dijkstra(g.grids['mapped_adj'], directed=False, indices=starts).mean(0)
    np.random.seed(0)
    scRL.align_pseudotime(g, 'start', boundary=False)
    assert g.grids['pseudotime_span'] == t.max() - t.min()
    assert t.max() - t.min() > 1


def _train(gres, Z):
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    seed_all(0)
    t = scRL.trainer('ActorCritic', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=300)
    t.train()
    return t


def test_3d_trajectory_runs_from_the_start_cluster():
    X, labels, Z, _ = make_tree(d=3)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    t = _train(gres, Z)
    seed_all(1)
    scRL.get_traj_df(gres, t, n=50, key='tr')
    traj = gres.trajectory['tr_traj']
    assert len(traj) > 2
    start_points = gres.grids['grids'][gres.grids['starts_cluster_grids']]
    dist = lambda p: np.linalg.norm(start_points - p, axis=1).min()
    assert dist(traj[0]) < dist(traj[-1])
    scRL.traj_results(gres, ['branchA'], 'tr')


def test_2d_trajectory_is_unchanged(scan2d):
    t = _train(scan2d, make_tree(d=2)[2])
    seed_all(1)
    scRL.get_traj_df(scan2d, t, n=30, key='tr')
    assert 'pseudotime_span' not in scan2d.grids
    grids, time, n = scan2d.grids['grids'], scan2d.grids['pseudotime'], scan2d.grids['n']
    df = pd.DataFrame(scan2d.trajectory['tr_idx'])
    expected = []
    for i, c in enumerate(df.columns):
        if df[c].isna().sum() > 30 * .9:
            continue
        idx = df[c].dropna().values.astype('int')
        mask = time[idx].values < i / n
        if any(mask):
            expected.append(grids[idx[mask], :].mean(axis=0))
    np.testing.assert_array_equal(scan2d.trajectory['tr_traj'], np.vstack(expected))


def _legend_inside(fig):
    from test_plot import _text_boxes
    legend = [ax.get_legend() for ax in fig.axes if ax.get_legend()][0]
    box = legend.get_window_extent(fig.canvas.get_renderer())
    W, H = fig.canvas.get_width_height()
    assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= W and box.y1 <= H, f'legend is cut off: {box}'
    for text, b in _text_boxes(fig):
        assert not box.overlaps(b), f'legend overlaps {text!r}'


def test_3d_trajectory_figure():
    import matplotlib.pyplot as plt
    from test_plot import _assert_layout
    X, labels, Z, _ = make_tree(d=3)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    t = _train(gres, Z)
    seed_all(1)
    scRL.get_traj_df(gres, t, n=50, key='tr')
    plt.close('all')
    for points, label in ((False, 'Along the path'), (True, 'Step')):
        scRL.traj_results(gres, ['trunk', 'branchA'], 'tr', points=points)
        fig = plt.gcf()
        (ax,) = [a for a in fig.axes if a.name == '3d']
        assert [ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()] == ['Dim 1', 'Dim 2', 'Dim 3']
        (bar,) = [a for a in fig.axes if a.get_label() == '<colorbar>']
        assert bar.get_ylabel() == label
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ['branchA', 'trunk']
        _assert_layout(fig)
        _legend_inside(fig)
    scRL.traj_results(gres, ['trunk', 'branchA'], 'tr')
    fig = plt.gcf()
    assert [t.get_text() for t in fig.axes[1].get_yticklabels()] == ['start', 'end']
    n_figs, n_axes = len(plt.get_fignums()), len(fig.axes)
    scRL.traj_results(gres, ['branchB'], 'tr', multiple=True)
    assert len(plt.get_fignums()) == n_figs and plt.gcf() is fig and len(fig.axes) == n_axes
    plt.close('all')

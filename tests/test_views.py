"""Mesh, cube and trajectory views."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import scRL
from scRL.PlotCore import _lattice_edges, _mean_by_length
from conftest import make_tree, seed_all


def _tree3d(n=12):
    X, labels, Z, genes = make_tree(d=3)
    gres = scRL.grids_from_embedding(X, n=n)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    return gres


def _walks(gres, ends, name, mode='Contribution'):
    scRL.d_rewards(gres, starts=['root'], ends=ends, mode=mode)
    scRL.exact_values(gres, reward_mode=mode, key_add=f'{name}_value')
    return scRL.exact_trajectories(gres, key=f'{name}_value', key_add=name)


def _collections(ax, kind):
    return [c for c in ax.collections if isinstance(c, kind)]


def _walk_lines(ax):
    return [c for c in ax.collections if str(c.get_label()).startswith('_walks')]


def test_lattice_edges_are_the_moves_once():
    gres = _tree3d()
    nb = np.asarray(gres.grids['neighbors'])
    E = _lattice_edges(gres)
    assert len(E) == (nb >= 0).sum() // 2 and (E[:, 0] < E[:, 1]).all()
    assert all(q in nb[p] for p, q in E[:50])


def test_mesh_draws_one_line_per_move(scan2d):
    gres = _tree3d()
    ax = scRL.plot_mesh(gres)
    lines = _collections(ax, Line3DCollection)
    ax.figure.canvas.draw()                                           # 3-D segments are projected on drawing
    assert ax.name == '3d' and len(lines) == 1 and len(lines[0].get_segments()) == len(_lattice_edges(gres))
    plt.close('all')
    scRL.exact_values(scRL.d_rewards(gres, starts=['root'], ends=['branchA']) or gres)
    ax = scRL.plot_mesh(gres, key='exact_branchA', color_edges=True)
    assert len(ax.figure.axes) == 2                                  # the colour bar
    plt.close('all')
    with pytest.raises(ValueError, match='needs a key'):
        scRL.plot_mesh(gres, color_edges=True)
    ax = scRL.plot_mesh(scan2d, s=0)                                 # 2-D scan grid: its eight moves
    assert ax.name != '3d' and len(_collections(ax, LineCollection)) >= 1
    plt.close('all')


def test_mean_by_length_of_two_samplings_of_one_line():
    a = np.array([[0., 0, 0], [1, 0, 0], [4, 0, 0]])
    b = np.array([[0., 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    M = _mean_by_length([a, b], k=5)
    np.testing.assert_allclose(M[:, 0], [0, 1, 2, 3, 4])
    np.testing.assert_allclose(M[:, 1:], 0)


def test_trajectories_in_three_and_two_dimensions(scan2d):
    gres = _tree3d()
    _walks(gres, ['branchA'], 'A')
    _walks(gres, ['branchB'], 'B')
    ax = scRL.plot_trajectories(gres, ['A', 'B'], labels=['branch A', 'branch B'], colors=['#c51b7d', '#01665e'],
                                max_walks=5)
    walks = _walk_lines(ax)
    ax.figure.canvas.draw()
    assert len(walks) == 2 and all(0 < len(w.get_segments()) <= 5 for w in walks)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['branch A', 'branch B']
    assert ax.computed_zorder is False
    plt.close('all')
    _walks(scan2d, ['branchA'], 'A2')
    ax = scRL.plot_trajectories(scan2d, 'A2', context='cells')
    assert ax.name != '3d' and len(_walk_lines(ax)) == 1
    plt.close('all')


def test_trajectories_from_stored_rollouts_and_errors():
    gres = _tree3d()
    mapped = np.asarray(gres.grids['mapped_grids'])
    gres.trajectory['roll_idx'] = [list(mapped[:4]), list(mapped[2:7])]   # the format get_traj_df stores
    ax = scRL.plot_trajectories(gres, 'roll', walks=False)
    assert len(_walk_lines(ax)) == 0 and len(ax.lines) >= 2           # the mean path and its outline only
    plt.close('all')
    with pytest.raises(KeyError, match='No trajectories'):
        scRL.plot_trajectories(gres, 'missing')
    with pytest.raises(ValueError, match='labels'):
        scRL.plot_trajectories(gres, ['roll'], labels=['a', 'b'])
    with pytest.raises(ValueError, match='context'):
        scRL.plot_trajectories(gres, 'roll', context='nothing')


def test_voxels_with_a_cut_and_a_value(scan2d):
    gres = _tree3d()
    ax = scRL.plot_voxels(gres)
    whole = len(_collections(ax, Poly3DCollection))                  # one collection per cube with a visible face
    plt.close('all')
    ax = scRL.plot_voxels(gres, cut=(0, .5))
    half = len(_collections(ax, Poly3DCollection))
    assert whole <= len(gres.grids['mapped_grids']) and 0 < half < whole
    plt.close('all')
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    scRL.exact_values(gres)
    ax = scRL.plot_voxels(gres, key='exact_branchA', label='Decision value')
    assert len(ax.figure.axes) == 2
    plt.close('all')
    with pytest.raises(ValueError, match='3 or more dimensions'):
        scRL.plot_voxels(scan2d)


def test_voxel_colours_follow_the_colour_map_and_the_cut_side():
    gres = _tree3d()
    gres.grids['height'] = np.asarray(gres.grids['grids'])[np.asarray(gres.grids['mapped_grids'])][:, 1].copy()
    ax = scRL.plot_voxels(gres, key='height', robust=False, colorbar=False)
    G = np.asarray(gres.grids['grids'])
    values = np.asarray(gres.grids['height'])
    cmap, norm = plt.get_cmap('plasma'), plt.Normalize(values.min(), values.max())
    worst = 0.
    for poly in _collections(ax, Poly3DCollection):           # each cube's face colour is its value's colour
        fc = poly.get_facecolor()[0][:3]
        expected = [cmap(norm(v))[:3] for v in values]
        worst = max(worst, min(np.abs(np.asarray(expected) - fc).max(1)))
    assert worst < 1e-6
    plt.close('all')
    ax = scRL.plot_voxels(gres, cut=(1, .5))
    # every drawn cube lies on the kept side of the cut
    ys = G[np.asarray(gres.grids['mapped_grids'])][:, 1]
    limit = ys.min() + .5 * np.ptp(ys)
    tops = [np.asarray(c._vec)[1].mean() for c in _collections(ax, Poly3DCollection)]
    assert max(tops) <= limit + np.abs(np.diff(np.unique(ys))).min() + 1e-9
    plt.close('all')
    for bad in ((3, .5), (1, 1.5), (0, -1)):
        with pytest.raises(ValueError, match='cut is'):
            scRL.plot_voxels(gres, cut=bad)


def test_mesh_edges_without_values_stay_plain_and_colour_counts():
    gres = _tree3d()
    v = np.arange(len(gres.grids['mapped_grids']), dtype=float)
    v[:200] = np.nan
    gres.grids['partial'] = v
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ax = scRL.plot_mesh(gres, key='partial', color_edges=True, colorbar=False)
    lines = _collections(ax, Line3DCollection)
    assert len(lines) == 2                                            # plain lines and coloured lines
    plt.close('all')
    _walks(gres, ['branchA'], 'A')
    with pytest.raises(ValueError, match='colours'):
        scRL.plot_trajectories(gres, ['A', 'A'], labels=['a', 'b'], colors=['red'])

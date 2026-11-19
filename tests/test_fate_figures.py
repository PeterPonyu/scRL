import numpy as np
import pandas as pd
import pytest

import scRL
from conftest import make_tree, seed_all
from test_plot import _assert_layout


def _gated(d=3):
    X, labels, Z, genes = make_tree(d=d)
    gres = scRL.grids_from_embedding(X, method='occupancy')
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    for end in ('branchA', 'branchB'):
        scRL.d_rewards(gres, starts=['root'], ends=[end])
        scRL.exact_values(gres)
    scRL.fate_gate(gres, ['exact_branchA', 'exact_branchB'], names=['A', 'B'])
    return gres, labels


@pytest.mark.parametrize('d, cells', [(3, False), (3, True), (2, False)])
def test_plot_fates(d, cells):
    gres, _ = _gated(d)
    fig = scRL.plot_fates(gres, cells=cells)
    maps = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    assert len(maps) == 3 and sum(ax.name == '3d' for ax in maps) == (2 if d == 3 else 0)
    legend = maps[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ['A', 'B']
    bars = maps[2]
    names = list(gres.embedding['clusters'].cat.categories)
    assert [t.get_text() for t in bars.get_yticklabels()] == names
    _assert_layout(fig)


def test_plot_conditions_with_many_long_cluster_names():
    X, labels, Z, genes = make_tree(n_cells=1200, d=3, seed=2)
    gres = scRL.grids_from_embedding(X)
    order = np.argsort(np.argsort(X[:, 0] + X[:, 1] + X[:, 2]))
    names = np.array([f'cluster{i:02d}xyz' for i in range(12)])
    scRL.project_cluster(gres, pd.Series(pd.Categorical(names[order * 12 // len(X)])))
    rng = np.random.default_rng(0)
    conditions = np.where(labels == 'branchB', 'treated', np.where(rng.random(len(X)) < .5, 'control', 'treated'))
    scRL.project_conditions(gres, conditions)
    scRL.compare_abundance(gres, 'control', 'treated', n_shuffles=5)
    fig = scRL.plot_conditions(gres)
    assert len([ax for ax in fig.axes if ax.get_label() != '<colorbar>']) == 4
    _assert_layout(fig)
    colorbar = [ax for ax in fig.axes if ax.get_label() == '<colorbar>'][0]
    assert colorbar.get_ylabel() == 'log2 fold change (control / treated)'


def test_plot_conditions_on_a_2d_grid(scan2d):
    conditions = np.where(np.arange(600) % 3 == 0, 'A', 'B')
    scRL.project_conditions(scan2d, conditions)
    scRL.compare_abundance(scan2d, 'A', 'B', n_shuffles=3)
    _assert_layout(scRL.plot_conditions(scan2d))

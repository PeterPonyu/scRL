import numpy as np
import pandas as pd
import pytest

import scRL
from conftest import make_tree, seed_all


def _gres():
    X, labels, Z, genes = make_tree(d=3)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    return gres


def test_gate_basic_properties():
    gres = _gres()
    M = len(gres.grids['mapped_grids'])
    rng = np.random.default_rng(0)
    for name in ('a', 'b', 'c'):
        gres.grids[name] = rng.normal(size=M)
    gate = scRL.fate_gate(gres, ['a', 'b', 'c'])
    np.testing.assert_allclose(gate.sum(1), 1)
    assert list(gate.columns) == ['a', 'b', 'c'] and list(gate.index) == list(gres.grids['mapped_grids'])
    assert set(gres.grids['gate_top']) <= {'a', 'b', 'c'}
    gres.grids['same'] = gres.grids['a'].copy()
    gate = scRL.fate_gate(gres, ['a', 'same'], key_add='g2')
    np.testing.assert_allclose(gate.to_numpy(), .5)
    np.testing.assert_allclose(gres.grids['g2_entropy'], 1)


def test_gate_top_rank_and_sharpness():
    gres = _gres()
    M = len(gres.grids['mapped_grids'])
    x = np.arange(M, dtype=float)
    gres.grids['up'], gres.grids['down'], gres.grids['down2'] = x, -x, -x
    gate = scRL.fate_gate(gres, ['up', 'down', 'down2'], tau=.1)
    assert gate['up'].iloc[-1] > .99
    sharp = scRL.fate_gate(gres, ['up', 'down'], tau=1e-6, key_add='sharp')
    middle = np.abs(np.arange(M) - (M - 1) / 2) > 1
    assert set(np.round(sharp.to_numpy()[middle], 6).ravel()) <= {0.0, 1.0}


def test_gate_cells_and_lineages():
    gres = _gres()
    for end in ('branchA', 'branchB'):
        scRL.d_rewards(gres, starts=['root'], ends=[end])
        scRL.exact_values(gres)
    gate = scRL.fate_gate(gres, ['exact_branchA', 'exact_branchB'], names=['A', 'B'])
    W = np.column_stack([gres.embedding['gate_A'], gres.embedding['gate_B']])
    np.testing.assert_allclose(W.sum(1), 1)
    X = gres.embedding['embedding']
    clusters = np.asarray(gres.embedding['clusters'])
    tipA = np.flatnonzero(clusters == 'branchA')[np.argmax(X[clusters == 'branchA'][:, 2])]
    tipB = np.flatnonzero(clusters == 'branchB')[np.argmax(X[clusters == 'branchB'][:, 1])]
    assert gres.embedding['gate_top'][tipA] == 'A' and gres.embedding['gate_top'][tipB] == 'B'
    assert set(gres.embedding['gate_top_colors']) == {'A', 'B'}
    assert 0 <= gres.embedding['gate_entropy'].min() and gres.embedding['gate_entropy'].max() <= 1


def test_gate_errors():
    gres = _gres()
    gres.grids['a'] = np.zeros(len(gres.grids['mapped_grids']))
    with pytest.raises(ValueError):
        scRL.fate_gate(gres, ['a'])
    with pytest.raises(KeyError):
        scRL.fate_gate(gres, ['a', 'nope'])
    with pytest.raises(ValueError):
        scRL.fate_gate(gres, ['a', 'a'], names=['x'])


def test_policy_views_accept_move_arrays():
    import torch
    from test_plot import _arrows, _policy_trainer
    gres = _gres()
    t = _policy_trainer(gres, 'ActorCritic', 3)
    t.agent.actor = lambda x: torch.nn.functional.one_hot(torch.full((len(x),), 0, device=x.device), 6).float()
    zeros = np.zeros(len(gres.grids['mapped_grids']), dtype=int)
    np.testing.assert_array_equal(_arrows(scRL.plot_policy(gres, zeros, every=2), True),
                                  _arrows(scRL.plot_policy(gres, t, every=2), True))
    from matplotlib.quiver import Quiver
    a = scRL.plot_slices(gres, key='pseudotime', policy=zeros, policy_every=2)
    b = scRL.plot_slices(gres, key='pseudotime', policy=t, policy_every=2)
    qa = [c for ax in a.axes for c in ax.collections if isinstance(c, Quiver)]
    qb = [c for ax in b.axes for c in ax.collections if isinstance(c, Quiver)]
    assert len(qa) == len(qb) > 0
    for x, y in zip(qa, qb):
        np.testing.assert_array_equal(np.column_stack([x.U, x.V]), np.column_stack([y.U, y.V]))

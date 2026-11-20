import copy
import warnings

import numpy as np
import pandas as pd
import pytest

import scRL
from conftest import make_tree

ENDS = {'A': 'tipA', 'B': 'tipB'}


def _labels(X, labels):
    labels = labels.astype(object)
    far = np.linalg.norm(X - np.array([6, 0, 0]), axis=1) > 3.5
    labels[(labels == 'branchA') & far] = 'tipA'
    labels[(labels == 'branchB') & far] = 'tipB'
    return labels.astype(str)


def _run(Y, labels, seed=0):
    gres = scRL.grids_from_embedding(Y)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    scRL.align_pseudotime(gres, 'root', boundary=False, seed=seed)
    return gres


def _runs(k=4):
    X, labels, Z, genes = make_tree(d=3)
    labels = _labels(X, labels)
    rng = np.random.default_rng(1)
    runs = [_run(X + rng.normal(0, .15, X.shape).astype(np.float32), labels) for _ in range(k)]
    return runs, labels


def test_mean_is_the_average_of_the_runs():
    runs, labels = _runs(4)
    copies = [copy.deepcopy(g) for g in runs]
    out = scRL.fate_over_runs(runs, ENDS, start='root')
    by_hand = []
    for g in copies:
        scRL.fate_probabilities(g, ENDS, key_add='x')
        by_hand.append(np.column_stack([g.embedding['x_A'], g.embedding['x_B']]))
    by_hand = np.array(by_hand)
    assert np.allclose(out['mean'].to_numpy(), by_hand.mean(0))
    assert np.allclose(out['sd'].to_numpy(), by_hand.std(0))
    assert out['runs'] == 4 and list(out['halves'].index) == ['A', 'B']
    assert out['halves'].notna().all() and (out['halves'] > .5).all()
    from scipy.stats import spearmanr
    unc = ~np.isin(labels, ['tipA', 'tipB'])                   # the halves are taken over uncommitted cells
    first, second = by_hand[:2].mean(0), by_hand[2:].mean(0)
    assert np.isclose(out['halves']['A'], spearmanr(first[unc, 0], second[unc, 0]).statistic)
    trunk = scRL.fate_over_runs([copy.deepcopy(g) for g in copies], ENDS, within=['trunk'])['halves']
    assert np.isclose(trunk['B'], spearmanr(first[labels == 'trunk', 1], second[labels == 'trunk', 1]).statistic)
    assert out['reachable'].shape == (4, 2) and np.allclose(out['reachable'].to_numpy(), 1)
    for g in runs:                                             # each run keeps its own shares
        assert 'fate_runs_A' in g.embedding
    g0 = runs[0]
    assert np.allclose(g0.embedding['fate_runs_mean_A'], by_hand.mean(0)[:, 0])
    assert list(g0.grids['fate_runs_mean'].columns) == ['A', 'B']
    reached = by_hand.mean(0).sum(1)
    assert np.allclose(g0.embedding['fate_runs_mean_reached'], reached)
    top = np.where(reached < .5, 'none', np.where(by_hand.mean(0)[:, 0] >= by_hand.mean(0)[:, 1], 'A', 'B'))
    assert (np.asarray(g0.embedding['fate_runs_mean_top']).astype(str) == top).all()


def test_identical_runs_have_no_spread_and_two_runs_no_halves():
    runs, labels = _runs(1)
    two = [runs[0], copy.deepcopy(runs[0])]
    out = scRL.fate_over_runs(two, ENDS)
    assert np.allclose(out['sd'].to_numpy(), 0) and out['halves'].isna().all() and 'reachable' not in out


def test_means_are_compared_between_conditions():
    runs, labels = _runs(4)
    out = scRL.fate_over_runs(runs, ENDS)
    g0 = runs[0]
    n = len(labels)
    cond = np.where(np.arange(n) % 2 == 0, 'ctrl', 'treat')
    samples = np.array([f'{c}{i % 3}' for i, c in enumerate(cond)])
    scRL.project_conditions(g0, cond)
    t = scRL.compare_fates(g0, 'treat', 'ctrl', key='fate_runs_mean', stat='mean', samples=samples,
                           n_permutations=50, n_boot=20)
    M = out['mean'].to_numpy()
    assert np.allclose(t.loc['A', 'difference'], M[cond == 'treat', 0].mean() - M[cond == 'ctrl', 0].mean())
    top = scRL.compare_fates(g0, 'treat', 'ctrl', key='fate_runs_mean_top', n_permutations=20, n_boot=20)
    assert set(top.index) <= {'A', 'B', 'none'}


def test_a_lineage_on_a_separate_part_is_reported_unreachable():
    X, labels, Z, genes = make_tree(d=3)
    labels = _labels(X, labels)
    island = X[:40] + np.array([0, 30, 0], dtype=np.float32)
    Y = np.vstack([X, island])
    lab = np.array(list(labels) + ['island'] * 40)
    runs = [_run(Y + np.random.default_rng(s).normal(0, .1, Y.shape).astype(np.float32), lab) for s in range(2)]
    ends = {'A': 'tipA', 'B': 'tipB', 'I': 'island'}
    out = scRL.fate_over_runs(runs, ends, start='root')
    assert np.allclose(out['reachable']['I'], 0) and np.allclose(out['reachable'][['A', 'B']], 1)
    trunk = lab == 'trunk'
    assert np.allclose(out['mean'].loc[trunk, 'I'], 0)


def test_a_lineage_missing_from_one_run_is_averaged_over_the_others():
    runs, labels = _runs(2)
    ends = {'A': 'tipA', 'B': 'tipB', 'R': 'root'}
    g1 = runs[1]
    cl = np.asarray(g1.grids['mapped_grids_clusters']).astype(object)
    cl[cl == 'root'] = 'trunk'                                  # run 1 has no end point of lineage R
    g1.grids['mapped_grids_clusters'] = cl
    copies = [copy.deepcopy(runs[0])]
    out = scRL.fate_over_runs(runs, ends)
    scRL.fate_probabilities(copies[0], ends, key_add='x')
    assert np.allclose(out['mean']['R'], copies[0].embedding['x_R'])
    assert np.allclose(out['sd']['R'], 0)


def test_errors():
    runs, labels = _runs(2)
    with pytest.raises(ValueError, match='at least two grids'):
        scRL.fate_over_runs(runs[:1], ENDS)
    with pytest.raises(ValueError, match='two lineages'):
        scRL.fate_over_runs(runs, {'A': 'tipA'})
    X, lab2, Z, genes = make_tree(n_cells=500, d=3)
    other = _run(X, _labels(X, lab2))
    with pytest.raises(ValueError, match='numbers of cells'):
        scRL.fate_over_runs([runs[0], other], ENDS)
    with pytest.raises(ValueError, match='start cluster'):
        scRL.fate_over_runs(runs, ENDS, start='nowhere')
    shuffled = copy.deepcopy(runs[1])
    shuffled.embedding['clusters'] = np.asarray(shuffled.embedding['clusters'])[::-1]
    with pytest.raises(ValueError, match='boolean mask'):
        scRL.fate_over_runs(runs, ENDS, within=np.ones(3, bool))
    with pytest.raises(ValueError, match='different cluster labels'):
        scRL.fate_over_runs([runs[0], shuffled], ENDS)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        scRL.fate_over_runs(runs, ENDS)

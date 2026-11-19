import numpy as np
import pandas as pd
import pytest

import scRL
from scRL.ConditionCore import _grid_for_cells, condition_grid
from scRL.EnvironmentNDCore import deepEnvND
from conftest import make_tree, seed_all


def _setup(n_cells=1500):
    X, labels, Z, genes = make_tree(n_cells=n_cells, d=3, seed=1)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    scRL.project(gres, genes)
    rng = np.random.default_rng(0)
    # B everywhere; A everywhere except branchB (the lost branch); A larger than B outside the branch
    conditions = np.where(labels == 'branchB', 'B', np.where(rng.random(len(labels)) < .6, 'A', 'B'))
    return gres, labels, Z, conditions


@pytest.fixture(scope='module')
def setup():
    return _setup()


def _fresh(setup):
    import copy
    gres, labels, Z, conditions = setup
    return copy.deepcopy(gres), labels, Z, conditions


def test_counts_match_brute_force(setup):
    gres, labels, Z, conditions = _fresh(setup)
    counts = scRL.project_conditions(gres, conditions, radius=1.5)
    assert list(counts.columns) == ['A', 'B']
    h = gres.grids['spacing']
    X = gres.embedding['embedding']
    points = gres.grids['grids'][gres.grids['mapped_grids']]
    for p in np.random.default_rng(1).choice(len(points), 20, replace=False):
        near = np.linalg.norm((X - points[p]) / h, axis=1) <= 1.5
        assert counts.iloc[p]['A'] == (near & (conditions == 'A')).sum()
        assert counts.iloc[p]['B'] == (near & (conditions == 'B')).sum()
    with pytest.raises(ValueError):
        scRL.project_conditions(gres, conditions[:-1])
    with pytest.raises(ValueError):
        scRL.project_conditions(gres, np.full(len(conditions), 'A'))


def test_abundance_finds_the_lost_branch(setup):
    gres, labels, Z, conditions = _fresh(setup)
    scRL.project_conditions(gres, conditions)
    table = scRL.compare_abundance(gres, 'A', 'B', n_shuffles=50)
    N_A, N_B = (conditions == 'A').sum(), (conditions == 'B').sum()
    for p in range(3):
        row = table.iloc[p * 97]
        assert row['log2fc'] == pytest.approx(np.log2(((row['n_test'] + .5) / N_A) / ((row['n_reference'] + .5) / N_B)))
    k_t, k_r = table['n_test'].to_numpy(), table['n_reference'].to_numpy()
    expected = np.where((k_t >= 3) & (k_r >= 3), 'shared', np.where(k_r >= 3, 'B only', np.where(k_t >= 3, 'A only',
                                                                                               'too few cells')))
    assert (table['class'].to_numpy() == expected).all()
    branch = gres.grids['mapped_grids_clusters'] == 'branchB'
    occupied = branch & (k_t + k_r >= 6)
    assert (table['class'].to_numpy()[occupied] == 'B only').mean() > .5
    assert (table['call'].to_numpy()[occupied] == 'depleted').mean() > .5
    summary = gres.grids['abundance_clusters']
    assert np.isfinite(summary[['B only by chance', 'A only by chance']].to_numpy()).all()
    assert summary.loc['branchB', 'B only'] >= 3 * summary.loc['branchB', 'B only by chance']
    assert np.isnan(gres.grids['abundance_log2fc'][~(k_t + k_r >= 6)]).all()

    shuffled = np.random.default_rng(2).permutation(conditions)
    scRL.project_conditions(gres, shuffled)
    table = scRL.compare_abundance(gres, 'A', 'B', n_shuffles=5)
    tested = table['call'] != 'too few cells'
    assert table['call'][tested].isin(['depleted', 'enriched']).mean() <= .01


def test_abundance_across_samples(setup):
    gres, labels, Z, conditions = _fresh(setup)
    rng = np.random.default_rng(3)
    same = np.where(rng.random(len(labels)) < .5, 'A', 'B')
    samples = np.array([f'{c}{rng.integers(3)}' for c in same])
    scRL.project_conditions(gres, same)
    table = scRL.compare_abundance(gres, 'A', 'B', samples=samples, n_shuffles=2)
    tested = table['call'] != 'too few cells'
    assert (table['pvalue'][tested] < .05).mean() <= .05 + 1e-9
    with pytest.raises(ValueError):
        scRL.compare_abundance(gres, 'A', 'B', samples=np.array([f'{c}0' for c in same]), n_shuffles=2)


def test_condition_grid_inherits_and_pays_the_same(setup):
    gres, labels, Z, conditions = _fresh(setup)
    scRL.project_conditions(gres, conditions)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    sub = condition_grid(gres, 'A')
    pos = sub.grids['parent_positions']
    assert set(sub.grids['mapped_grids']) <= set(gres.grids['mapped_grids'])
    assert (sub.grids['mapped_grids_clusters'] == gres.grids['mapped_grids_clusters'][pos]).all()
    np.testing.assert_array_equal(sub.grids['pseudotime'].to_numpy(),
                                  gres.grids['pseudotime'].loc[sub.grids['mapped_grids']].to_numpy())
    R, nb = gres.qlearning['d_Decision_rewards'].to_numpy(), gres.grids['neighbors']
    pay = {}
    for p, a in zip(*np.nonzero(nb >= 0)):
        pay.setdefault(nb[p, a], set()).add(R[p, a])
    assert all(len(v) == 1 for v in pay.values())
    Rs, nbs = sub.qlearning['d_Decision_rewards'].to_numpy(), sub.grids['neighbors']
    for p, a in zip(*np.nonzero(nbs >= 0)):
        assert Rs[p, a] == next(iter(pay[pos[nbs[p, a]]]))
    assert (Rs[nbs < 0] == -1).all()
    mask = conditions == 'A'
    env = deepEnvND(sub, Z[mask], KNN=10, reward_type='d', reward_mode='Decision', starts_probs=False)
    env.reset()
    env.step(0)
    assert (np.asarray(sub.grids['mapped_grids_clusters']) != 'branchB').mean() > .9


def test_condition_grid_on_a_scan_grid(scan2d):
    conditions = np.where(np.arange(600) % 3 == 0, 'A', 'B')
    scRL.project_conditions(scan2d, conditions)
    scRL.d_rewards(scan2d, starts=['root'], ends=['branchA'])
    sub = condition_grid(scan2d, 'B')
    assert 'neighbors' in sub.grids and set(sub.grids['mapped_grids']) <= set(scan2d.grids['mapped_grids'])
    assert np.isfinite(scRL.exact_values(sub)).all()
    with pytest.raises(ValueError):
        condition_grid(scan2d, 'C')


def test_compare_values_finds_the_lost_branch(setup):
    gres, labels, Z, conditions = _fresh(setup)
    scRL.project_conditions(gres, conditions)
    scRL.d_rewards(gres, starts=['root'], ends=['branchB'], mode='Contribution')
    table = scRL.compare_values(gres, 'A', 'B', reward_mode='Contribution', n_shuffles=20)
    assert table.loc['trunk', 'effect'] < 0 and abs(table.loc['trunk', 'z']) > 3
    assert table.loc['trunk', 'relative'] < 0 and list(table.columns)[:3] == ['shared_points', 'effect', 'relative']
    M = len(gres.grids['mapped_grids'])
    for suffix in ('A', 'B', 'effect', 'z'):
        assert gres.grids[f'cond_branchB_{suffix}'].shape == (M,)
    shared = ~np.isnan(gres.grids['cond_branchB_effect'])
    assert shared.any() and np.isnan(gres.grids['cond_branchB_z'][~shared]).all()
    pd.testing.assert_frame_equal(table, scRL.compare_values(gres, 'A', 'B', reward_mode='Contribution',
                                                             n_shuffles=20))
    # Decision rewards pay most at the entry of the lineage, and the base of the lost branch is still on A's map
    scRL.d_rewards(gres, starts=['root'], ends=['branchB'])
    table = scRL.compare_values(gres, 'A', 'B', n_shuffles=5)
    assert np.isfinite(table[['effect', 'z']].to_numpy()).all()


@pytest.mark.parametrize('mode', ['Decision', 'Contribution'])
def test_compare_values_present_branch(setup, mode):
    gres, labels, Z, conditions = _fresh(setup)
    scRL.project_conditions(gres, conditions)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'], mode=mode)
    table = scRL.compare_values(gres, 'A', 'B', reward_mode=mode, n_shuffles=20)
    assert abs(table.loc['trunk', 'z']) < 3
    z = gres.grids['cond_branchA_z']
    shared = ~np.isnan(gres.grids['cond_branchA_effect'])
    assert not np.isnan(z[shared]).any()


def test_compare_fates(setup):
    gres, labels, Z, conditions = _fresh(setup)
    scRL.project_conditions(gres, conditions)
    gres.embedding['lineage'] = np.where(np.isin(labels, ['branchA', 'branchB']), labels, 'stem')
    table = scRL.compare_fates(gres, 'A', 'B', key='lineage', n_permutations=500, n_boot=100)
    np.testing.assert_allclose(table[['reference', 'test']].sum(0), 1)
    assert table.loc['branchB', 'test'] < table.loc['branchB', 'reference'] and table.loc['branchB', 'pvalue'] < .01
    scRL.project_conditions(gres, np.random.default_rng(5).permutation(conditions))
    table = scRL.compare_fates(gres, 'A', 'B', key='lineage', n_permutations=500, n_boot=50)
    assert (table['pvalue'] > .05).all()

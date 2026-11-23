import numpy as np
import pandas as pd
import pytest
from scipy import stats

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


def test_condition_grid_labels_from_own_cells(setup):
    gres, labels, Z, conditions = _fresh(setup)
    scRL.project_conditions(gres, conditions)
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    sub = condition_grid(gres, 'A')
    pos = sub.grids['parent_positions']
    assert set(sub.grids['mapped_grids']) <= set(gres.grids['mapped_grids'])
    np.testing.assert_array_equal(sub.grids['pseudotime'].to_numpy(),
                                  gres.grids['pseudotime'].loc[sub.grids['mapped_grids']].to_numpy())
    mask = conditions == 'A'
    X, own = gres.embedding['embedding'][mask], labels[mask]
    points = gres.grids['grids'][sub.grids['mapped_grids']]
    for p in np.random.default_rng(4).choice(len(points), 30, replace=False):
        assert sub.grids['mapped_grids_clusters'][p] == own[np.argmin(np.linalg.norm(X - points[p], axis=1))]
    assert (np.asarray(sub.grids['mapped_grids_clusters']) != 'branchB').all()
    assert not [k for k in sub.qlearning if k.endswith('_rewards')]
    assert 'starts_cluster_grids' not in sub.grids
    scRL.d_rewards(sub, starts=['root'], ends=['branchA'])
    env = deepEnvND(sub, Z[mask], KNN=10, reward_type='d', reward_mode='Decision', starts_probs=False)
    env.reset()
    env.step(0)


def test_condition_grid_on_a_scan_grid(scan2d):
    conditions = np.where(np.arange(600) % 3 == 0, 'A', 'B')
    scRL.project_conditions(scan2d, conditions)
    sub = condition_grid(scan2d, 'B')
    assert 'neighbors' in sub.grids and set(sub.grids['mapped_grids']) <= set(scan2d.grids['mapped_grids'])
    scRL.d_rewards(sub, starts=['root'], ends=['branchA'])
    assert np.isfinite(scRL.exact_values(sub)).all()
    with pytest.raises(ValueError):
        condition_grid(scan2d, 'C')


def _overlapping_loss(seed=0):
    """
    branchB's territory also holds cells of a neutral cluster 'other' in both conditions; condition A has lost
    every remaining branchB cell.
    """
    X, labels, Z, genes = make_tree(n_cells=1500, d=3, seed=1)
    rng = np.random.default_rng(seed)
    labels = labels.astype(object)
    in_b = np.flatnonzero(labels == 'branchB')
    labels[rng.choice(in_b, int(.4 * len(in_b)), replace=False)] = 'other'
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels.astype(str))))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    conditions = np.where(rng.random(len(labels)) < .5, 'A', 'B')
    conditions[labels == 'branchB'] = 'B'
    return gres, labels, conditions


@pytest.fixture(scope='module')
def overlap():
    return _overlapping_loss()


@pytest.mark.parametrize('mode', ['Decision', 'Contribution'])
def test_compare_values_complete_loss_in_shared_territory(overlap, mode):
    import copy
    gres, labels, conditions = copy.deepcopy(overlap[0]), overlap[1], overlap[2]
    scRL.project_conditions(gres, conditions)
    table = scRL.compare_values(gres, 'A', 'B', ends=['branchB'], starts=['root'], mode=mode, n_shuffles=20)
    trunk = table.loc['trunk']
    assert trunk['effect'] < 0 and trunk['relative'] < -.2 and abs(trunk['z']) > 3
    table = scRL.compare_values(gres, 'A', 'B', ends=['branchA'], starts=['root'], mode=mode, n_shuffles=20)
    assert abs(table.loc['trunk', 'z']) < 3
    M = len(gres.grids['mapped_grids'])
    for suffix in ('A', 'B', 'effect', 'z'):
        assert gres.grids[f'cond_branchA_{suffix}'].shape == (M,)
    assert not np.isinf(gres.grids['cond_branchA_z']).any()


def test_compare_values_without_depletion(setup):
    gres, labels, Z, _ = _fresh(setup)
    halves = np.where(np.random.default_rng(7).random(len(labels)) < .5, 'A', 'B')
    scRL.project_conditions(gres, halves)
    for end in ('branchA', 'branchB'):
        table = scRL.compare_values(gres, 'A', 'B', ends=end, starts='root', n_shuffles=20)
        assert (table['z'].abs().fillna(0) <= 3).all()
    first = scRL.compare_values(gres, 'A', 'B', ends='branchA', n_shuffles=5)
    pd.testing.assert_frame_equal(first, scRL.compare_values(gres, 'A', 'B', ends='branchA', n_shuffles=5))


def test_compare_values_same_reward_rule_on_both_maps(overlap):
    import copy
    from scRL.ConditionCore import _lineage_grid
    gres, labels, conditions = copy.deepcopy(overlap[0]), overlap[1], overlap[2]
    scRL.project_conditions(gres, conditions)
    clusters = np.asarray(gres.grids['mapped_grids_clusters'])
    t = gres.grids['pseudotime'].loc[gres.grids['mapped_grids']].to_numpy()[clusters == 'branchA']
    rule = (['branchA'], ['root'], 'Contribution', 1, t.min(), t.max())
    maps = [_lineage_grid(gres, conditions == c, *rule) for c in ('A', 'B')]
    pay = []
    for g in maps:
        R, nb = g.qlearning['d_Contribution_rewards'].to_numpy(), g.grids['neighbors']
        entry = {}
        for p, a in zip(*np.nonzero(nb >= 0)):
            entry[g.grids['mapped_grids'][nb[p, a]]] = R[p, a]
        ends = set(np.asarray(g.grids['mapped_grids'])[np.asarray(g.grids['mapped_grids_clusters']) == 'branchA'])
        pay.append({k: v for k, v in entry.items() if k in ends})
    common = set(pay[0]) & set(pay[1])
    assert len(common) > 20 and all(pay[0][k] == pay[1][k] for k in common)


def test_z_is_never_infinite():
    from scRL.ConditionCore import _z
    z = _z(np.array([0.0, .3, -.2, np.nan]), np.array([0.0, 0.0, .1, 0.0]))
    assert z[0] == 0 and np.isnan(z[1]) and z[2] == pytest.approx(-2) and np.isnan(z[3])


def test_compare_values_errors(setup):
    gres, labels, Z, conditions = _fresh(setup)
    scRL.project_conditions(gres, conditions)
    with pytest.raises(ValueError, match='branchA'):
        scRL.compare_values(gres, 'A', 'B', ends=['nowhere'])


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


def test_cluster_pvalue_counts_only_shuffles_that_entered():
    from scRL.ConditionCore import _cluster_table
    clusters = np.array(['a'] * 4 + ['b'] * 4)
    d = np.array([1., 1., 1., 1., .5, .5, .5, .5])
    null = np.zeros((10, 8))
    null[:, 4:] = np.nan
    null[:6, 4:] = np.linspace(-.1, .1, 6)[:, None]
    null[:, :4] = np.linspace(-.2, .2, 10)[:, None]
    table = _cluster_table(d, null, clusters, ['a', 'b'], spread=2.0)
    assert table.loc['a', 'shuffles'] == 10 and table.loc['a', 'pvalue'] == pytest.approx(1 / 11)
    assert table.loc['b', 'shuffles'] == 6 and table.loc['b', 'pvalue'] == pytest.approx(1 / 7)
    null[4:, 4:] = np.nan
    table = _cluster_table(d, null, clusters, ['a', 'b'], spread=2.0)
    assert table.loc['b', 'shuffles'] == 4 and np.isnan(table.loc['b', ['effect', 'z', 'pvalue']]).all()


def _samples_setup(remove_lineage, seed=0):
    """Three samples per condition from one population; optionally branchB removed from every test sample."""
    X, labels, Z, genes = make_tree(n_cells=1500, d=3, seed=1)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    rng = np.random.default_rng(seed)
    conditions = np.where(rng.random(len(labels)) < .5, 'A', 'B')
    if remove_lineage:
        conditions[labels == 'branchB'] = 'B'
    samples = np.array([f'{c}{i}' for c, i in zip(conditions, rng.integers(3, size=len(labels)))])
    scRL.project_conditions(gres, conditions)
    gres.embedding['lineage'] = np.where(np.isin(labels, ['branchA', 'branchB']), labels, 'stem')
    return gres, labels, conditions, samples


def test_shares_by_sample():
    gres, labels, conditions, samples = _samples_setup(False)
    table = scRL.shares_by_sample(gres, samples, key='lineage')
    assert list(table.columns)[0] == 'condition'
    np.testing.assert_allclose(table.drop(columns='condition').sum(1), 1)
    lineage = gres.embedding['lineage']
    for s in ('A0', 'B2'):
        cells = samples == s
        assert table.loc[s, 'branchA'] == pytest.approx((lineage[cells] == 'branchA').mean())
        assert table.loc[s, 'condition'] == s[0]
    mixed = samples.copy()
    mixed[np.flatnonzero(conditions == 'B')[0]] = 'A0'
    with pytest.raises(ValueError, match='A0'):
        scRL.shares_by_sample(gres, mixed, key='lineage')


def test_compare_fates_across_samples():
    gres, labels, conditions, samples = _samples_setup(False)
    plain = scRL.compare_fates(gres, 'A', 'B', key='lineage', n_permutations=200, n_boot=50)
    table = scRL.compare_fates(gres, 'A', 'B', key='lineage', samples=samples, n_permutations=200, n_boot=50)
    pd.testing.assert_frame_equal(table[plain.columns], plain)
    for column in ('reference_mean', 'test_mean', 'difference_samples', 't', 'df', 'pvalue_samples',
                   'n_reference', 'n_test'):
        assert column in table.columns
    assert (table['pvalue_samples'] > .05).all() and (table['n_test'] == 3).all()
    gres, labels, conditions, samples = _samples_setup(True)
    table = scRL.compare_fates(gres, 'A', 'B', key='lineage', samples=samples, n_permutations=200, n_boot=50)
    assert table.loc['branchB', 'pvalue_samples'] < .01 and table.loc['branchB', 'test_mean'] == 0
    with pytest.raises(ValueError, match='two'):
        scRL.compare_fates(gres, 'A', 'B', key='lineage', samples=np.array([c + '0' for c in conditions]))


def test_compare_abundance_cluster_shares_across_samples():
    gres, labels, conditions, samples = _samples_setup(True)
    scRL.compare_abundance(gres, 'A', 'B', samples=samples, n_shuffles=3)
    table = gres.grids['abundance_clusters']
    for column in ('share_reference', 'share_test', 'pvalue_samples', 'qvalue_samples'):
        assert column in table.columns
    for cluster in ('trunk', 'branchB'):
        per = [np.mean(labels[samples == s] == cluster) for s in ('A0', 'A1', 'A2')]
        assert table.loc[cluster, 'share_test'] == pytest.approx(np.mean(per))
    assert table.loc['branchB', 'qvalue_samples'] < .05


def test_welch_without_spread():
    from scRL.ConditionCore import _welch
    a = np.array([[.2, .5, .3], [.2, .5, .4]])
    b = np.array([[.2, .1, .3], [.2, .1, .5]])
    t, df, p = _welch(a, b)
    assert p[0] == 1 and t[0] == 0 and np.isnan(df[0])
    assert p[1] == 0 and t[1] == np.inf
    from scipy import stats
    assert p[2] == pytest.approx(stats.ttest_ind(a[:, 2], b[:, 2], equal_var=False).pvalue)


def test_cluster_qvalues_include_separated_clusters():
    X, labels, Z, genes = make_tree(n_cells=1500, d=3, seed=1)
    labels = labels.astype(object)
    in_b = np.flatnonzero(labels == 'branchB')
    labels[in_b[:len(in_b) % 3]] = 'trunk'                                # a multiple of 3 branchB cells
    labels = labels.astype(str)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    rng = np.random.default_rng(0)
    conditions = np.where(rng.random(len(labels)) < .5, 'A', 'B')
    conditions[labels == 'branchB'] = 'B'
    rest = np.flatnonzero((conditions == 'B') & (labels != 'branchB'))
    extra = len(rest) % 3
    conditions[rest[:extra]] = 'A'                                        # equal numbers of B cells per sample
    samples = np.empty(len(labels), dtype=object)
    for group in (np.flatnonzero((conditions == 'B') & (labels == 'branchB')),
                  np.flatnonzero((conditions == 'B') & (labels != 'branchB'))):
        samples[group] = [f'B{i % 3}' for i in range(len(group))]
    a_cells = np.flatnonzero(conditions == 'A')
    samples[a_cells] = [f'A{i}' for i in rng.integers(3, size=len(a_cells))]
    scRL.project_conditions(gres, conditions)
    scRL.compare_abundance(gres, 'A', 'B', samples=samples.astype(str), n_shuffles=2, scale='share')
    table = gres.grids['abundance_clusters']                              # equal shares in every B sample
    assert table.loc['branchB', 'pvalue_samples'] == 0 and np.isfinite(table['qvalue_samples']).all()
    assert table.loc['branchB', 'qvalue_samples'] == 0


def test_shares_by_sample_rejects_a_condition_category():
    gres, labels, conditions, samples = _samples_setup(False)
    gres.embedding['lineage'] = np.where(labels == 'root', 'condition', 'other')
    with pytest.raises(ValueError, match='condition'):
        scRL.shares_by_sample(gres, samples, key='lineage')


def _old_shares_by_sample(gres, samples, key):
    """The table of shares_by_sample as computed before the composition options were added."""
    labels = np.asarray(gres.embedding['conditions'])
    samples = np.asarray(samples).astype(str)
    of = {s: labels[samples == s][0] for s in sorted(set(samples.tolist()))}
    cell_key = np.asarray(gres.embedding[key]).astype(str)
    cats = sorted(set(cell_key.tolist()))
    table = pd.DataFrame.from_dict({s: [(cell_key[samples == s] == c).mean() for c in cats] for s in of},
                                   orient='index', columns=cats)
    table.insert(0, 'condition', [of[s] for s in table.index])
    return table


def _old_compare_fates(gres, test, reference, key, samples, n_permutations, n_boot, seed=0):
    """The table of compare_fates as computed before the composition options were added."""
    from scRL.ConditionCore import _welch
    labels = np.asarray(gres.embedding['conditions'])
    cell_key = np.asarray(gres.embedding[key]).astype(str)
    pool = np.flatnonzero((labels == test) | (labels == reference))
    cats = sorted(set(cell_key[pool].tolist()))
    C = (cell_key[pool][:, None] == np.array(cats)[None, :]).astype(float)
    is_test = labels[pool] == test
    rng = np.random.default_rng(seed)
    shares = lambda m: (C[m].mean(0), C[~m].mean(0))
    st, sr = shares(is_test)
    diff = st - sr
    boot_t = np.array([C[is_test][rng.integers(0, is_test.sum(), is_test.sum())].mean(0) for _ in range(n_boot)])
    boot_r = np.array([C[~is_test][rng.integers(0, (~is_test).sum(), (~is_test).sum())].mean(0)
                       for _ in range(n_boot)])
    count = np.zeros(len(cats))
    for _ in range(n_permutations):
        pt, pr = shares(rng.permutation(is_test))
        count += np.abs(pt - pr) >= np.abs(diff)
    table = pd.DataFrame({'reference': sr, 'reference_low': np.percentile(boot_r, 2.5, axis=0),
                          'reference_high': np.percentile(boot_r, 97.5, axis=0),
                          'test': st, 'test_low': np.percentile(boot_t, 2.5, axis=0),
                          'test_high': np.percentile(boot_t, 97.5, axis=0),
                          'difference': diff, 'pvalue': (1 + count) / (n_permutations + 1)}, index=cats)
    if samples is not None:
        per = _old_shares_by_sample(gres, samples, key)
        per = per[per['condition'].isin([test, reference])]
        values = per.reindex(columns=cats, fill_value=0.0)
        a = values[per['condition'] == test].to_numpy(dtype=float)
        b = values[per['condition'] == reference].to_numpy(dtype=float)
        t, df, p = _welch(a, b)
        table['reference_mean'], table['test_mean'] = b.mean(0), a.mean(0)
        table['difference_samples'], table['t'], table['df'], table['pvalue_samples'] = a.mean(0) - b.mean(0), t, df, p
        table['n_reference'], table['n_test'] = len(b), len(a)
    return table


def test_defaults_reproduce_the_previous_tables():
    gres, labels, conditions, samples = _samples_setup(True)
    pd.testing.assert_frame_equal(scRL.shares_by_sample(gres, samples, key='lineage'),
                                  _old_shares_by_sample(gres, samples, 'lineage'))
    for s in (None, samples):
        pd.testing.assert_frame_equal(
            scRL.compare_fates(gres, 'A', 'B', key='lineage', samples=s, n_permutations=200, n_boot=50),
            _old_compare_fates(gres, 'A', 'B', 'lineage', s, 200, 50))


def _with_shares(seed=0):
    gres, labels, conditions, samples = _samples_setup(False, seed)
    rng = np.random.default_rng(seed)
    W = rng.random((len(labels), 2))
    W[rng.random(len(labels)) < .1, 0] = np.nan
    gres.grids['fate'] = pd.DataFrame(np.zeros((len(gres.grids['mapped_grids']), 2)), columns=['x', 'y'])
    gres.embedding['fate_x'], gres.embedding['fate_y'] = W[:, 0], W[:, 1]
    return gres, labels, conditions, samples


def test_mean_statistic_and_selection():
    gres, labels, conditions, samples = _with_shares()
    within = ['trunk', 'root']
    mask = np.isin(labels, within)
    table = scRL.shares_by_sample(gres, samples, key='fate', stat='mean', within=within)
    for s in ('A1', 'B0'):
        cells = (samples == s) & mask
        assert table.loc[s, 'x'] == pytest.approx(np.nanmean(gres.embedding['fate_x'][cells]))
    pd.testing.assert_frame_equal(table, scRL.shares_by_sample(gres, samples, key='fate', stat='mean', within=mask))
    fates = scRL.compare_fates(gres, 'A', 'B', key='fate', stat='mean', within=within, n_permutations=50, n_boot=20)
    a = (conditions == 'A') & mask
    assert fates.loc['y', 'test'] == pytest.approx(np.nanmean(gres.embedding['fate_y'][a]))
    with pytest.raises(ValueError, match='fate_'):
        scRL.shares_by_sample(gres, samples, key='nothing', stat='mean')
    with pytest.raises(ValueError):
        scRL.shares_by_sample(gres, samples, key='fate', stat='median')
    with pytest.raises(ValueError):
        scRL.compare_fates(gres, 'A', 'B', key='lineage', within=np.zeros(len(labels), bool))
    with pytest.raises(ValueError, match='nowhere'):
        scRL.compare_fates(gres, 'A', 'B', key='lineage', within=['nowhere'])


def _composition_case(shift_inside):
    """Two conditions over two clusters: either different composition and equal statistics, or the reverse."""
    from scRL.GridCore import Grids_Results
    rng = np.random.default_rng(1)
    n = 4000
    clusters = np.where(rng.random(n) < .5, 'c1', 'c2')
    conditions = np.where(rng.random(n) < .5, 'A', 'B')
    if not shift_inside:                                   # B holds more of c1; within clusters nothing differs
        flip = (conditions == 'B') & (clusters == 'c2') & (rng.random(n) < .5)
        clusters[flip] = 'c1'
    top = np.where(clusters == 'c1', rng.random(n) < .8, rng.random(n) < .2)
    if shift_inside:                                       # same composition; in c1, A is less often 'yes'
        top = np.where((clusters == 'c1') & (conditions == 'A'), rng.random(n) < .5, top)
    g = Grids_Results()
    g.embedding['embedding'] = rng.random((n, 3))
    g.embedding['clusters'] = pd.Series(pd.Categorical(clusters))
    g.embedding['conditions'] = conditions
    g.embedding['top'] = np.where(top, 'yes', 'no')
    return g


def test_composition_part_and_within_part():
    g = _composition_case(shift_inside=False)
    t = scRL.compare_fates(g, 'A', 'B', key='top', standardise=True, n_permutations=10, n_boot=10)
    assert abs(t.loc['yes', 'difference']) > .05
    assert abs(t.loc['yes', 'difference_std']) < .02 and abs(t.loc['yes', 'within_part']) < .02
    assert t.loc['yes', 'composition_part'] == pytest.approx(t.loc['yes', 'difference'], abs=.02)
    g = _composition_case(shift_inside=True)
    t = scRL.compare_fates(g, 'A', 'B', key='top', standardise=True, n_permutations=10, n_boot=10)
    assert abs(t.loc['yes', 'composition_part']) < .01
    assert t.loc['yes', 'within_part'] == pytest.approx(t.loc['yes', 'difference'], abs=.01)


def _check_parts(gres, samples, kept, stat='top', key='lineage'):
    """The parts add up to the difference over the kept clusters, computed here on those clusters' cells alone."""
    t = scRL.compare_fates(gres, 'A', 'B', key=key, stat=stat, samples=samples, standardise=True,
                           n_permutations=10, n_boot=10)
    alone = scRL.compare_fates(gres, 'A', 'B', key=key, stat=stat, samples=samples, within=kept,
                               n_permutations=10, n_boot=10)
    target = alone['difference' if samples is None else 'difference_samples'].reindex(t.index, fill_value=0.0)
    np.testing.assert_allclose(t['composition_part'] + t['within_part'], t['difference_kept'], atol=1e-12)
    np.testing.assert_allclose(t['difference_kept'], target, atol=1e-12)
    return t


def test_parts_add_up_and_missing_clusters_are_reported():
    gres, labels, conditions, samples = _samples_setup(True)
    kept = ['branchA', 'root', 'trunk']
    t = _check_parts(gres, None, kept)
    assert t.attrs['dropped_clusters'] == ['branchB']                     # no branchB cells in condition A
    assert abs(t.loc['branchB', 'difference_kept']) < 1e-12 and abs(t.loc['branchB', 'difference']) > .05
    t = _check_parts(gres, samples, kept)
    assert 'pvalue_samples_std' in t.columns and t.attrs['dropped_clusters'] == ['branchB']
    per = scRL.shares_by_sample(gres, samples, key='lineage', standardise=True)
    assert 'stem_std' in per.columns and per.attrs['dropped_clusters'] == ['branchB']


def test_parts_with_a_cluster_missing_from_one_sample():
    gres, labels, conditions, samples = _samples_setup(False)
    samples = samples.copy()
    samples[(samples == 'A0') & (labels == 'branchB')] = 'A1'             # branchB absent from sample A0 only
    t = _check_parts(gres, None, None)                                    # across cells nothing is dropped
    assert t.attrs['dropped_clusters'] == []
    np.testing.assert_allclose(t['difference_kept'], t['difference'], atol=1e-12)
    t = _check_parts(gres, samples, ['branchA', 'root', 'trunk'])
    assert t.attrs['dropped_clusters'] == ['branchB']
    gres2, *_ = _with_shares()
    gres2.embedding['fate_x'] = np.where(np.isnan(gres2.embedding['fate_x']), .5, gres2.embedding['fate_x'])
    _check_parts(gres2, samples, ['branchA', 'root', 'trunk'], stat='mean', key='fate')


def test_weighted_within_cluster_statistic_over_samples():
    """A condition's within-cluster statistic is sum w s / sum w over its samples, not the plain mean."""
    from scRL.ConditionCore import _composition, _kept_clusters
    gres, labels, conditions, samples = _with_shares()
    gres.embedding['fate_x'] = np.where(np.isnan(gres.embedding['fate_x']), .5, gres.embedding['fate_x'])
    t = scRL.compare_fates(gres, 'A', 'B', key='fate', stat='mean', samples=samples, standardise=True,
                           n_permutations=10, n_boot=10)
    V = np.column_stack([gres.embedding['fate_x'], gres.embedding['fate_y']])
    clusters = labels.astype(str)
    units = {u: samples == u for u in sorted(set(samples.tolist()))}
    kept, _ = _kept_clusters(clusters, units)
    parts = _composition(V, clusters, units, kept)
    def terms(c):
        members = [u for u in units if u[0] == c]
        w = np.mean([parts[u][0] for u in members], axis=0)
        s = sum(parts[u][0][:, None] * parts[u][1] for u in members) / sum(parts[u][0] for u in members)[:, None]
        return w, s
    (w_a, s_a), (w_b, s_b) = terms('A'), terms('B')
    np.testing.assert_allclose(t['within_part'], ((w_a + w_b) / 2) @ (s_a - s_b), atol=1e-12)
    plain = {c: np.mean([parts[u][1] for u in units if u[0] == c], axis=0) for c in 'AB'}
    assert np.abs(t['within_part'] - ((w_a + w_b) / 2) @ (plain['A'] - plain['B'])).max() > 1e-6


def test_missing_cluster_labels_and_std_name_clash():
    gres, labels, conditions, samples = _samples_setup(False)
    clusters = pd.Series(pd.Categorical(labels))
    clusters[0] = np.nan
    gres.embedding['clusters'] = clusters
    with pytest.raises(ValueError, match='no cluster label'):
        scRL.compare_fates(gres, 'A', 'B', key='lineage', within=['root'])
    with pytest.raises(ValueError, match='no cluster label'):
        scRL.compare_fates(gres, 'A', 'B', key='lineage', standardise=True)
    with pytest.raises(ValueError, match='no cluster label'):
        scRL.shares_by_sample(gres, samples, key='lineage', standardise=True)
    gres, labels, conditions, samples = _samples_setup(False)
    gres.embedding['lineage'] = np.where(labels == 'root', 'stem_std', gres.embedding['lineage'])
    scRL.shares_by_sample(gres, samples, key='lineage')                    # fine without standardise
    with pytest.raises(ValueError, match='stem_std'):
        scRL.shares_by_sample(gres, samples, key='lineage', standardise=True)


def test_paired_test_across_blocks():
    from scipy import stats
    gres, labels, conditions, samples = _samples_setup(False)
    blocks = {f'A{i}': f'block{i}' for i in range(3)} | {f'B{i}': f'block{i}' for i in range(3)}
    t = scRL.compare_fates(gres, 'A', 'B', key='lineage', samples=samples, pairs=blocks, standardise=True,
                           n_permutations=10, n_boot=10)
    per = scRL.shares_by_sample(gres, samples, key='lineage')
    expected = stats.ttest_rel(per.loc[['A0', 'A1', 'A2'], 'stem'].to_numpy(),
                               per.loc[['B0', 'B1', 'B2'], 'stem'].to_numpy()).pvalue
    assert t.loc['stem', 'pvalue_paired'] == pytest.approx(expected)
    assert 'pvalue_paired_std' in t.columns
    cell_blocks = np.array([blocks[s] for s in samples])
    t2 = scRL.compare_fates(gres, 'A', 'B', key='lineage', samples=samples, pairs=cell_blocks, n_permutations=10,
                            n_boot=10)
    np.testing.assert_allclose(t2['pvalue_paired'], t['pvalue_paired'])
    unbalanced = dict(blocks, B2='block0')
    with pytest.raises(ValueError, match='block'):
        scRL.compare_fates(gres, 'A', 'B', key='lineage', samples=samples, pairs=unbalanced)
    with pytest.raises(ValueError, match='samples'):
        scRL.compare_fates(gres, 'A', 'B', key='lineage', pairs=blocks)


def test_fate_comparisons_default_to_the_walk():
    import inspect
    for f in (scRL.compare_fates, scRL.shares_by_sample):
        assert inspect.signature(f).parameters['key'].default == 'fate_top'
    gres, labels, conditions, samples = _samples_setup(False)
    with pytest.raises(KeyError, match="fate_probabilities first, or pass key='gate_top'"):
        scRL.compare_fates(gres, 'A', 'B')
    with pytest.raises(KeyError, match="fate_probabilities first, or pass key='gate_top'"):
        scRL.shares_by_sample(gres, samples)
    gres.embedding['fate_top'] = np.where(labels == 'root', 'A', 'B')
    with pytest.raises(ValueError, match="key='fate'"):
        scRL.compare_fates(gres, 'A', 'B', stat='mean')


def _counts_case(counts, key='clusters'):
    """Cells from a table of counts: {(sample, condition, category): n}; categories as clusters or another key."""
    from scRL.GridCore import Grids_Results
    rows = [(s, c, k) for (s, c, k), n in counts.items() for _ in range(n)]
    samples, conditions, cats = (np.array(x) for x in zip(*rows))
    g = Grids_Results()
    g.embedding['embedding'] = np.random.default_rng(0).random((len(rows), 3))
    g.embedding['conditions'] = conditions
    if key == 'clusters':
        g.embedding['clusters'] = pd.Series(pd.Categorical(cats))
    else:
        g.embedding['clusters'] = pd.Series(pd.Categorical(['c'] * len(rows)))
        g.embedding[key] = cats
    return g, samples


def _halved_cluster_counts():
    """Four blocks; in the test condition the large cluster c1 keeps half its cells, the others are unchanged."""
    rng = np.random.default_rng(3)
    counts = {}
    for b in range(4):
        base = {'c1': 400, 'c2': 100, 'c3': 100, 'c4': 100}
        size = rng.uniform(.6, 1.6)                                        # block effect on every cluster
        for cond, tag in (('ref', 'R'), ('test', 'T')):
            for k, n in base.items():
                m = n * size * rng.uniform(.9, 1.1)
                counts[(f'{tag}{b}', cond, k)] = int(round(m / 2 if (cond == 'test' and k == 'c1') else m))
    return counts


def _transformed(counts, cond, tag, cats, log):
    out = []
    for b in range(4):
        n = np.array([counts[(f'{tag}{b}', cond, k)] for k in cats], dtype=float)
        out.append(np.log2((n + .5) / (n.sum() + 1)) if log else n / n.sum())
    return np.array(out)


def test_scale_default_rule():
    import inspect
    for f in (scRL.compare_fates, scRL.shares_by_sample, scRL.compare_abundance):
        assert inspect.signature(f).parameters['scale'].default is None
    counts = _halved_cluster_counts()
    g, samples = _counts_case(counts)
    run = lambda **kw: scRL.compare_fates(g, 'test', 'ref', samples=samples, n_permutations=10, n_boot=10, **kw)
    assert 'log2fc_samples' in run(key='clusters').columns                # cluster label: log
    assert 'log2fc_samples' not in run(key='clusters', scale='share').columns
    gl, samples_l = _counts_case(counts, key='lineage')
    lin = scRL.compare_fates(gl, 'test', 'ref', key='lineage', samples=samples_l, n_permutations=10, n_boot=10)
    assert 'log2fc_samples' not in lin.columns                            # any other key: share
    assert 'log2fc_samples' in scRL.compare_fates(gl, 'test', 'ref', key='lineage', samples=samples_l, scale='log',
                                                  n_permutations=10, n_boot=10).columns
    assert 'c1_log2' in scRL.shares_by_sample(g, samples, key='clusters').columns
    with pytest.raises(ValueError, match='scale'):
        run(key='clusters', scale='ratio')
    with pytest.raises(ValueError, match='standardise'):
        run(key='clusters', standardise=True, scale='log')
    std = run(key='clusters', standardise=True)                          # default resolves to the share scale
    assert 'log2fc_samples' not in std.columns and 'difference_std' in std.columns
    scRL.compare_fates(g, 'test', 'ref', key='clusters', standardise=True, n_permutations=10, n_boot=10)
    with pytest.raises(ValueError, match="stat='top'"):
        scRL.shares_by_sample(gl, samples_l, key='lineage', stat='mean', scale='log')


def test_halved_large_cluster_on_both_scales():
    from scipy import stats
    counts = _halved_cluster_counts()
    g, samples = _counts_case(counts)
    blocks = {f'{t}{b}': f'block{b}' for b in range(4) for t in 'RT'}
    cats = ['c1', 'c2', 'c3', 'c4']
    out = {}
    for scale in ('log', 'share'):
        t = scRL.compare_fates(g, 'test', 'ref', key='clusters', samples=samples, pairs=blocks, scale=scale,
                               n_permutations=10, n_boot=10)
        yt, yr = (_transformed(counts, c, tag, cats, scale == 'log') for c, tag in (('test', 'T'), ('ref', 'R')))
        np.testing.assert_allclose(t.loc[cats, 'pvalue_paired'], stats.ttest_rel(yt, yr).pvalue, rtol=1e-10)
        np.testing.assert_allclose(t.loc[cats, 'pvalue_samples'],
                                   stats.ttest_ind(yt, yr, equal_var=False).pvalue, rtol=1e-10)
        st, sr = (_transformed(counts, c, tag, cats, False) for c, tag in (('test', 'T'), ('ref', 'R')))
        np.testing.assert_allclose(t.loc[cats, 'difference_samples'], st.mean(0) - sr.mean(0), atol=1e-12)
        if scale == 'log':
            np.testing.assert_allclose(t.loc[cats, 'log2fc_samples'], yt.mean(0) - yr.mean(0), atol=1e-12)
        out[scale] = t
    log = out['log']
    assert log.loc['c1', 'pvalue_paired'] < .001                           # the halved cluster is called
    assert -.6 < log.loc['c1', 'log2fc_samples'] < -.4                     # share 4/7 -> 2/5


def test_share_scale_matches_the_reference_computation():
    gres, labels, conditions, samples = _samples_setup(True)
    for s in (None, samples):
        pd.testing.assert_frame_equal(
            scRL.compare_fates(gres, 'A', 'B', key='clusters', samples=s, scale='share', n_permutations=200,
                               n_boot=50),
            _old_compare_fates(gres, 'A', 'B', 'clusters', s, 200, 50))
    from scRL.ConditionCore import _welch
    scRL.compare_abundance(gres, 'A', 'B', samples=samples, n_shuffles=3, scale='share')
    table = gres.grids['abundance_clusters']
    shares = {s: [np.mean(labels[samples == s] == c) for c in table.index] for s in sorted(set(samples))}
    p = _welch(np.array([shares[s] for s in shares if s[0] == 'A']),
               np.array([shares[s] for s in shares if s[0] == 'B']))[2]
    assert np.array_equal(table['pvalue_samples'].to_numpy(), p)


def test_log_scale_under_the_null():
    from scipy import stats
    gres, labels, conditions, samples = _samples_setup(False)
    t = scRL.compare_fates(gres, 'A', 'B', key='clusters', samples=samples, n_permutations=10, n_boot=10)
    cats = list(t.index)
    y = {s: np.log2((np.array([np.sum(labels[samples == s] == c) for c in cats]) + .5) / ((samples == s).sum() + 1))
         for s in sorted(set(samples))}
    a = np.array([y[s] for s in y if s[0] == 'A'])
    b = np.array([y[s] for s in y if s[0] == 'B'])
    np.testing.assert_allclose(t['pvalue_samples'], stats.ttest_ind(a, b, equal_var=False).pvalue, rtol=1e-10)
    assert np.median(t['pvalue_samples']) > .1
    scRL.compare_abundance(gres, 'A', 'B', samples=samples, n_shuffles=3)
    table = gres.grids['abundance_clusters']
    np.testing.assert_allclose(table.loc[cats, 'pvalue_samples'], t['pvalue_samples'], rtol=1e-12)
    np.testing.assert_allclose(table.loc[cats, 'log2fc_samples'], a.mean(0) - b.mean(0), atol=1e-12)


def test_log_scale_with_a_category_missing_from_some_samples():
    """Explicit scale='log' on a lineage key: a category with no cells in a sample gets the pseudo-count value,
    which moves with the sample's size alone; a category with no cells in the comparison is not tested."""
    from scipy import stats
    counts = {}
    for i, n in enumerate((80, 120, 100)):
        counts[(f'R{i}', 'ref', 'X')] = 5
        counts[(f'R{i}', 'ref', 'Y')] = n
        counts[(f'T{i}', 'test', 'Y')] = n // 2                           # no X cells, smaller selections
    g, samples = _counts_case(counts, key='lineage')
    with pytest.warns(UserWarning, match="category 'X' has no cells in some samples"):
        t = scRL.compare_fates(g, 'test', 'ref', key='lineage', samples=samples, scale='log', n_permutations=10,
                               n_boot=10)
    with pytest.warns(UserWarning, match="scale='share'"):
        scRL.shares_by_sample(g, samples, key='lineage', scale='log')
    assert list(t.index) == ['X', 'Y']
    yt = np.log2((np.array([0, 0, 0]) + .5) / (np.array([40, 60, 50]) + 1))
    yr = np.log2((np.array([5, 5, 5]) + .5) / (np.array([85, 125, 105]) + 1))
    assert t.loc['X', 'log2fc_samples'] == pytest.approx(yt.mean() - yr.mean())
    assert t.loc['X', 'pvalue_samples'] == pytest.approx(stats.ttest_ind(yt, yr, equal_var=False).pvalue)
    assert t.loc['X', 'difference_samples'] == pytest.approx(-np.mean(5 / np.array([85, 125, 105])))


def test_zeros_with_equal_sample_sizes_stay_silent():
    import warnings
    counts = {}
    for i, n in enumerate((80, 120, 100)):
        counts[(f'R{i}', 'ref', 'X')] = 5
        counts[(f'R{i}', 'ref', 'Y')] = n
        counts[(f'T{i}', 'test', 'Y')] = n + 5                             # no X cells, same sizes
    g, samples = _counts_case(counts, key='lineage')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        scRL.compare_fates(g, 'test', 'ref', key='lineage', samples=samples, scale='log', n_permutations=10,
                           n_boot=10)
        scRL.shares_by_sample(g, samples, key='lineage', scale='log')
        gres, labels, conditions, samples = _samples_setup(False)
        scRL.compare_fates(gres, 'A', 'B', key='clusters', samples=samples, n_permutations=10, n_boot=10)
        scRL.shares_by_sample(gres, samples, key='clusters')
        scRL.compare_abundance(gres, 'A', 'B', samples=samples, n_shuffles=3)


def test_halving_a_cluster_is_found_on_the_log_scale_with_pairs():
    """Halving one cluster in every block: the paired log test finds it, the paired share test does not."""
    from scipy import stats
    reference = [400, 200, 50, 10]
    counts = {}
    for b, n in enumerate(reference):
        counts[(f'R{b}', 'ref', 'X')], counts[(f'R{b}', 'ref', 'Y')] = n, 1000 - n
        counts[(f'T{b}', 'test', 'X')], counts[(f'T{b}', 'test', 'Y')] = n // 2, 1000 - n // 2
    g, samples = _counts_case(counts)
    blocks = {f'{t}{b}': f'block{b}' for b in range(4) for t in 'RT'}
    x_ref = np.array(reference, float)
    x_test = x_ref / 2
    share = (x_test / 1000, x_ref / 1000)
    log = (np.log2((x_test + .5) / 1001), np.log2((x_ref + .5) / 1001))
    p_share, p_log = stats.ttest_rel(*share).pvalue, stats.ttest_rel(*log).pvalue
    assert p_share > .05 and p_log < .001                                 # about 0.159 and 8.6e-6
    assert stats.ttest_ind(*share, equal_var=False).pvalue > .05
    assert stats.ttest_ind(*log, equal_var=False).pvalue > .05            # about 0.45 and 0.57: pairing is needed
    for scale, (p_paired, values) in {'share': (p_share, share), 'log': (p_log, log)}.items():
        t = scRL.compare_fates(g, 'test', 'ref', key='clusters', samples=samples, pairs=blocks, scale=scale,
                               n_permutations=10, n_boot=10)
        assert t.loc['X', 'pvalue_paired'] == pytest.approx(p_paired, rel=1e-10)
        welch = stats.ttest_ind(*values, equal_var=False).pvalue
        assert t.loc['X', 'pvalue_samples'] == pytest.approx(welch, rel=1e-10) and welch > .05
    assert t.loc['X', 'pvalue_paired'] < .001
    assert t.loc['X', 'log2fc_samples'] == pytest.approx(-0.978, abs=.001)
    assert t.loc['X', 'log2fc_samples'] == pytest.approx((log[0] - log[1]).mean(), abs=1e-12)


def _clustered(n_samples, n_cells, between_sd, seed):
    rng = np.random.default_rng(seed)
    means = rng.normal(0, between_sd, n_samples)
    values = (means[:, None] + rng.normal(0, 1, (n_samples, n_cells))).ravel()
    return values, np.repeat([f's{i}' for i in range(n_samples)], n_cells)


def test_design_effect_independent_values():
    values, samples = _clustered(10, 200, 0.0, 0)
    out = scRL.design_effect(values, samples)
    assert abs(out['icc']) < .02
    assert .05 <= out['implied_false_positive_rate_normal'] < .15
    c = stats.norm.ppf(.975) / np.sqrt(out['design_effect'])
    assert out['implied_false_positive_rate'] == pytest.approx(2 * stats.t.sf(c * np.sqrt(8) / np.sqrt(9 - c ** 2), 8))
    assert .03 < out['implied_false_positive_rate'] < out['implied_false_positive_rate_normal']


def test_design_effect_strong_clustering():
    values, samples = _clustered(10, 200, 1.0, 4)
    out = scRL.design_effect(values, samples)
    assert abs(out['icc'] - .5) < .15
    assert out['implied_false_positive_rate'] > .8


def test_design_effect_exact_arithmetic():
    out = scRL.design_effect([1, 2, 3, 4, 5, 6], list('aaabbb'), scale='raw')
    icc = 12.5 / 15.5
    deff = 1 + 2 * icc
    z = stats.norm.ppf(.975)
    assert out['n_cells'] == 6 and out['n_samples'] == 2 and out['n_dropped'] == 0
    assert out['cells_per_sample'] == pytest.approx(3)
    assert out['icc'] == pytest.approx(icc)
    assert out['design_effect'] == pytest.approx(deff)
    assert out['implied_false_positive_rate_normal'] == pytest.approx(2 * (1 - stats.norm.cdf(z / np.sqrt(deff))))
    assert out['implied_false_positive_rate'] == out['implied_false_positive_rate_normal']      # two samples
    assert out['effective_cells_per_sample'] == pytest.approx(3 / deff)
    assert out['alpha'] == .05 and out['scale'] == 'raw'
    assert all(type(out[k]) in (int, float, str, bool) for k in out)


def test_design_effect_rank_scale_ignores_monotone_transforms():
    values, samples = _clustered(6, 50, .7, 2)
    a = scRL.design_effect(values, samples)
    b = scRL.design_effect(np.exp(3 * values), samples)
    assert a['icc'] == pytest.approx(b['icc'], abs=1e-12)
    raw_a = scRL.design_effect(values, samples, scale='raw')
    raw_b = scRL.design_effect(np.exp(3 * values), samples, scale='raw')
    assert abs(raw_a['icc'] - raw_b['icc']) > .01


def test_design_effect_negative_icc_gives_no_inflation():
    values = [0, 10, 0, 10, 0, 10, 0, 10.5]
    out = scRL.design_effect(values, list('aaaabbbb'), scale='raw', alpha=.1)
    assert out['icc'] < 0
    assert out['design_effect'] == 1.0
    assert out['implied_false_positive_rate'] == pytest.approx(.1, abs=1e-9)
    assert out['effective_cells_per_sample'] == pytest.approx(out['cells_per_sample'])


def test_design_effect_drops_missing_values():
    values, samples = _clustered(4, 30, .5, 3)
    dirty = pd.Series(values.copy())
    dirty[[0, 5, 40]] = np.nan
    keep = ~dirty.isna().to_numpy()
    out = scRL.design_effect(dirty, pd.Series(samples))
    clean = scRL.design_effect(values[keep], samples[keep])
    assert out['n_dropped'] == 3 and clean['n_dropped'] == 0
    assert out['n_cells'] == 117
    for k in ('icc', 'design_effect', 'implied_false_positive_rate', 'implied_false_positive_rate_normal',
              'cells_per_sample'):
        assert out[k] == pytest.approx(clean[k])


def test_design_effect_errors():
    ok = [1., 2, 3, 4]
    with pytest.raises(ValueError, match='two samples'):
        scRL.design_effect(ok, list('aaaa'))
    with pytest.raises(ValueError, match='two cells'):
        scRL.design_effect([1., np.nan], list('ab'))
    with pytest.raises(ValueError, match='more than one cell'):
        scRL.design_effect(ok, list('abcd'))
    with pytest.raises(ValueError, match='labels'):
        scRL.design_effect(ok, list('aab'))
    for alpha in (0, 1, -.1, 1.5):
        with pytest.raises(ValueError, match='alpha'):
            scRL.design_effect(ok, list('aabb'), alpha=alpha)
    with pytest.raises(ValueError, match='scale'):
        scRL.design_effect(ok, list('aabb'), scale='log')


def test_design_effect_predicts_the_false_call_rate_of_a_cell_level_test():
    rng = np.random.default_rng(7)
    k, n = 16, 150
    values = (rng.normal(0, np.sqrt(.05 / .95), k)[:, None] + rng.normal(0, 1, (k, n)))
    predicted = scRL.design_effect(values.ravel(), np.repeat(np.arange(k), n))['implied_false_positive_rate']
    calls = 0
    for _ in range(400):
        order = rng.permutation(k)
        a, b = values[order[:8]].ravel(), values[order[8:]].ravel()
        calls += stats.mannwhitneyu(a, b, alternative='two-sided').pvalue < .05
    assert abs(calls / 400 - predicted) < .12


def test_design_effect_few_sample_rate_against_the_t_distribution_written_out():
    values, samples = _clustered(8, 100, .37, 34)
    out = scRL.design_effect(values, samples)
    deff, k = out['design_effect'], 8
    assert out['n_samples'] == k and deff == pytest.approx(13, abs=1)
    c = stats.norm.ppf(.975) / np.sqrt(deff)
    expected = 2 * stats.t.sf(c * np.sqrt(k - 2) / np.sqrt(k - 1 - c ** 2), k - 2)
    assert out['implied_false_positive_rate'] == pytest.approx(expected)
    assert out['implied_false_positive_rate_normal'] == pytest.approx(2 * stats.norm.sf(c))
    assert .02 < out['implied_false_positive_rate'] - out['implied_false_positive_rate_normal'] < .06


def test_few_sample_rate_formula():
    rate = scRL.ConditionCore._few_sample_rate
    z = stats.norm.ppf(.975)
    c = z / np.sqrt(13)
    assert rate(c, 8) == pytest.approx(2 * stats.t.sf(c * np.sqrt(6) / np.sqrt(7 - c * c), 6))
    assert rate(c, 8) == pytest.approx(0.6256, abs=5e-4)
    assert rate(c, 3) == pytest.approx(2 * stats.t.sf(c / np.sqrt(2 - c * c), 1))
    assert rate(z, 4) == 0.0                           # c squared above k - 1
    assert rate(2.0, 5) == 0.0 and 0 < rate(2.0 - 1e-6, 5) < 1e-3     # c squared equal to k - 1, and just below
    assert rate(0.0, 5) == 1.0
    assert all(rate(c, k) > rate(c, k + 1) for k in range(3, 40))    # falls towards the normal value
    assert rate(c, 4000) == pytest.approx(2 * stats.norm.sf(c), abs=1e-4)


def test_design_effect_few_sample_rate_is_zero_for_four_samples_without_clustering():
    values, samples = _clustered(4, 200, 0.0, 3)
    out = scRL.design_effect(values, samples)
    assert out['design_effect'] < 1.2
    assert out['implied_false_positive_rate'] == 0.0
    assert .05 <= out['implied_false_positive_rate_normal'] < .08


def test_design_effect_few_sample_rate_tends_to_the_normal_form_with_many_samples():
    values, samples = _clustered(400, 20, .5, 5)
    out = scRL.design_effect(values, samples)
    assert out['n_samples'] == 400 and out['design_effect'] > 3
    gap = out['implied_false_positive_rate'] - out['implied_false_positive_rate_normal']
    assert 0 < gap < .005


def test_design_effect_is_exported():
    assert callable(scRL.design_effect) and 'design_effect' in scRL.__all__


def test_design_effect_unequal_sample_sizes():
    """With samples of 10 and 2 cells the correlation uses the size n0, not the mean size."""
    out = scRL.design_effect(np.arange(1, 13), ['a'] * 10 + ['b'] * 2, scale='raw')
    msb, msw = 60.0, 8.3
    n0 = (12 - (100 + 4) / 12) / 1
    assert out['cells_per_sample'] == pytest.approx(6.0)
    assert out['icc'] == pytest.approx((msb - msw) / (msb + (n0 - 1) * msw))
    assert out['icc'] == pytest.approx(0.6514069718605627)


def test_design_effect_refuses_infinite_raw_values():
    with pytest.raises(ValueError, match='finite'):
        scRL.design_effect([1, 2, np.inf, 4, 5, 6, 7, 8], list('aaaabbbb'), scale='raw')
    out = scRL.design_effect([1, 2, np.inf, 4, 5, 6, 7, 8], list('aaaabbbb'))
    assert np.isfinite(out['icc'])


def test_design_effect_drops_cells_without_a_sample():
    values = [1., 2., 3., 4., 5., 6., 7., 8.]
    samples = ['a', 'a', 'a', None, 'b', 'b', 'b', np.nan]
    out = scRL.design_effect(values, samples, scale='raw')
    ref = scRL.design_effect([1., 2., 3., 5., 6., 7.], list('aaabbb'), scale='raw')
    assert out['n_samples'] == 2 and out['n_dropped'] == 2
    assert out['icc'] == pytest.approx(ref['icc'])


def test_design_effect_ranks_infinite_values_and_says_how_to_drop_them():
    """On the rank scale an infinite value is the largest rank; set it to NaN to leave the cell out."""
    values = np.array([1., 2., 3., np.inf, 5., 6., 7., 8.])
    samples = list('aaaabbbb')
    kept = scRL.design_effect(values, samples)
    assert kept['n_cells'] == 8 and kept['n_dropped'] == 0
    dropped = scRL.design_effect(np.where(np.isfinite(values), values, np.nan), samples)
    assert dropped['n_cells'] == 7 and dropped['n_dropped'] == 1
    assert 'infinite' in scRL.design_effect.__doc__


def _two_conditions_by_hand():
    """Condition x: samples a (1, 2, 3) and b (5, 7); condition y: samples c (10, 14) and d (20, 21, 22, 23)."""
    values = [1, 2, 3, 5, 7, 10, 14, 20, 21, 22, 23]
    samples = list('aaabbccdddd')
    groups = list('xxxxxyyyyyy')
    return values, samples, groups


def test_design_effect_within_conditions_exact_arithmetic():
    values, samples, groups = _two_conditions_by_hand()
    out = scRL.design_effect(values, samples, groups=groups, scale='raw')
    # condition x: mean 3.6; sample means 2 and 6; condition y: mean 110 / 6; sample means 12 and 21.5
    ssb_x = 3 * (2 - 3.6) ** 2 + 2 * (6 - 3.6) ** 2                      # 19.2
    ssb_y = 2 * (12 - 110 / 6) ** 2 + 4 * (21.5 - 110 / 6) ** 2          # 120.33...
    ssw_x, ssw_y = 2 + 2, 8 + 5
    msb = (ssb_x + ssb_y) / (4 - 2)                                      # samples - conditions
    msw = (ssw_x + ssw_y) / (11 - 4)                                     # cells - samples
    n0 = ((5 - (9 + 4) / 5) + (6 - (4 + 16) / 6)) / (4 - 2)
    assert ssb_x == pytest.approx(19.2) and ssb_y == pytest.approx(361 / 3)
    assert n0 == pytest.approx(38 / 15)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    assert out['within_groups'] is True
    assert out['icc'] == pytest.approx(icc)
    assert out['icc'] == pytest.approx(0.9162834, abs=1e-6)
    assert out['n_cells'] == 11 and out['n_samples'] == 4 and out['cells_per_sample'] == pytest.approx(2.75)
    deff = 1 + (2.75 - 1) * icc
    c = stats.norm.ppf(.975) / np.sqrt(deff)
    assert out['design_effect'] == pytest.approx(deff)
    assert out['implied_false_positive_rate'] == pytest.approx(2 * stats.t.sf(c * np.sqrt(2) / np.sqrt(3 - c ** 2), 2))
    assert out['implied_false_positive_rate_normal'] == pytest.approx(2 * stats.norm.sf(c))
    plain = scRL.design_effect(values, samples, scale='raw')
    assert plain['within_groups'] is False
    assert plain['icc'] > out['icc']


def _within_condition_icc(x, samples, groups):
    """The pooled within-condition estimate written out from its definition, on the values as given."""
    ssb = ssw = coefficient = 0.0
    n_samples = 0
    for g in np.unique(groups):
        in_g = groups == g
        centre = x[in_g].mean()
        names = np.unique(samples[in_g])
        n_samples += len(names)
        coefficient += in_g.sum()
        for s in names:
            cells = x[in_g & (samples == s)]
            ssb += len(cells) * (cells.mean() - centre) ** 2
            ssw += ((cells - cells.mean()) ** 2).sum()
            coefficient -= len(cells) ** 2 / in_g.sum()
    msb = ssb / (n_samples - len(np.unique(groups)))
    msw = ssw / (len(x) - n_samples)
    n0 = coefficient / (n_samples - len(np.unique(groups)))
    return (msb - msw) / (msb + (n0 - 1) * msw)


def test_design_effect_within_conditions_is_not_inflated_by_a_difference_between_them():
    rng = np.random.default_rng(11)
    sizes = np.array([40, 55, 30, 70, 45, 60, 35, 50])
    samples = np.repeat([f's{i}' for i in range(8)], sizes)
    groups = np.repeat(['young'] * 4 + ['old'] * 4, sizes)
    values = np.repeat(rng.normal(0, .2, 8), sizes) + rng.normal(0, 1, sizes.sum()) + 3.0 * (groups == 'old')
    out = scRL.design_effect(values, samples, groups=groups)
    ranks = stats.rankdata(values)                                       # over all kept cells
    assert out['icc'] == pytest.approx(_within_condition_icc(ranks, samples, groups), rel=1e-10)
    raw = scRL.design_effect(values, samples, groups=groups, scale='raw')
    assert raw['icc'] == pytest.approx(_within_condition_icc(values, samples, groups), rel=1e-10)
    plain = scRL.design_effect(values, samples)
    assert out['icc'] < .1 and plain['icc'] > .6
    assert plain['icc'] > 5 * out['icc']
    assert plain['implied_false_positive_rate'] > out['implied_false_positive_rate'] + .1
    assert out['n_samples'] == plain['n_samples'] == 8
    assert out['cells_per_sample'] == plain['cells_per_sample']


def test_design_effect_with_one_condition_equals_the_call_without_groups():
    values, samples = _clustered(6, 40, .5, 8)
    samples = samples[:-7]                                               # unequal sample sizes
    values = values[:-7]
    plain = scRL.design_effect(values, samples)
    one = scRL.design_effect(values, samples, groups=['only'] * len(values))
    assert plain['within_groups'] is False and one['within_groups'] is True
    assert set(one) == set(plain)
    for key in plain:
        if key != 'within_groups':
            assert one[key] == pytest.approx(plain[key], rel=1e-12), key


def test_design_effect_refuses_a_sample_lying_in_two_conditions():
    values, samples, groups = _two_conditions_by_hand()
    groups[4] = 'y'                                                      # one cell of sample b
    with pytest.raises(ValueError, match=r"one condition.*\['b'\]"):
        scRL.design_effect(values, samples, groups=groups)
    with pytest.raises(ValueError, match='groups'):
        scRL.design_effect(values, samples, groups=groups[:-1])


def test_design_effect_condition_with_a_single_sample():
    """A condition with one sample adds to the within-sample part only; with no other kind nothing is left."""
    values = [1, 2, 3, 5, 7, 10, 14, 20, 21, 22, 23, 40, 43]
    samples = list('aaabbccddddee')
    groups = list('xxxxxyyyyyyzz')
    out = scRL.design_effect(values, samples, groups=groups, scale='raw')
    msb = (19.2 + 361 / 3) / (5 - 3)
    msw = (4 + 13 + 4.5) / (13 - 5)
    n0 = ((5 - 13 / 5) + (6 - 20 / 6) + (2 - 4 / 2)) / (5 - 3)
    assert out['n_samples'] == 5 and out['within_groups'] is True
    assert out['icc'] == pytest.approx((msb - msw) / (msb + (n0 - 1) * msw))
    with pytest.raises(ValueError, match='two or more samples'):
        scRL.design_effect([1, 2, 3, 5, 7, 9], list('aaabbb'), groups=list('xxxyyy'))


def test_design_effect_drops_cells_without_a_condition():
    values, samples, groups = _two_conditions_by_hand()
    ref = scRL.design_effect(values[1:], samples[1:], groups=groups[1:], scale='raw')
    out = scRL.design_effect(values, samples, groups=[None] + groups[1:], scale='raw')
    assert out['n_dropped'] == 1 and out['n_cells'] == 10
    assert out['icc'] == pytest.approx(ref['icc'])


def test_design_effect_docstring_says_when_the_within_condition_estimate_is_needed():
    sentence = 'with ``groups`` the correlation is estimated within conditions, as it must be when the conditions may differ'
    assert sentence in ' '.join(scRL.design_effect.__doc__.split())
    assert sentence in ' '.join(scRL.compare_readout.__doc__.split())

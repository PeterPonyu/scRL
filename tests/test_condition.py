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
        run(key='clusters', standardise=True)
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


def test_share_scale_reproduces_the_previous_results():
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
    t = scRL.compare_fates(g, 'test', 'ref', key='lineage', samples=samples, scale='log', n_permutations=10,
                           n_boot=10)
    assert list(t.index) == ['X', 'Y']
    yt = np.log2((np.array([0, 0, 0]) + .5) / (np.array([40, 60, 50]) + 1))
    yr = np.log2((np.array([5, 5, 5]) + .5) / (np.array([85, 125, 105]) + 1))
    assert t.loc['X', 'log2fc_samples'] == pytest.approx(yt.mean() - yr.mean())
    assert t.loc['X', 'pvalue_samples'] == pytest.approx(stats.ttest_ind(yt, yr, equal_var=False).pvalue)
    assert t.loc['X', 'difference_samples'] == pytest.approx(-np.mean(5 / np.array([85, 125, 105])))

import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scRL


def _design(means, cells=4, sd=0.0, seed=0, groups=None, k1=None):
    """Cells around given sample means; the first k1 samples are condition 'a', the rest 'b'.

    cells is one number for every sample or one number per sample."""
    rng = np.random.default_rng(seed)
    k = len(means)
    k1 = k // 2 if k1 is None else k1
    sizes = np.broadcast_to(cells, k)
    values = np.concatenate([m + sd * rng.standard_normal(c) for m, c in zip(means, sizes)])
    samples = np.repeat([f's{i}' for i in range(k)], sizes)
    grp = np.repeat(['a'] * k1 + ['b'] * (k - k1), sizes)
    return values, samples, grp


def _clustered(k, cells, rho, rng, shift=0.0):
    sample_effect = np.sqrt(rho) * rng.standard_normal(k)
    values = np.repeat(sample_effect, cells) + np.sqrt(1 - rho) * rng.standard_normal(k * cells)
    grp = np.repeat(['a'] * (k // 2) + ['b'] * (k - k // 2), cells)
    values = values + shift * (grp == 'a')
    return values, np.repeat(np.arange(k), cells), grp


def _on_scale(values, scale):
    """The readout on the named scale, written out here and not taken from the package: mid-ranks over all cells,
    or the standard normal quantile of (rank - 0.5) / number of cells."""
    values = np.asarray(values, dtype=float)
    if scale == 'raw':
        return values
    ranks = stats.rankdata(values, method='average')
    return ranks if scale == 'rank' else stats.norm.ppf((ranks - 0.5) / len(values))


def _sample_means(z, samples):
    """Unweighted mean per sample, in order of first appearance."""
    return pd.Series(np.asarray(z, dtype=float)).groupby(np.asarray(samples), sort=False).mean().to_numpy()


def _hand():
    # sample means 1, 2, 3 against 4, 5, 9
    samples = np.repeat(list('abcdef'), 2)
    values = np.array([0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 8, 10], dtype=float)
    # means: 1, 2, 3, 4, 5, 9
    groups = np.repeat(['x'] * 3 + ['y'] * 3, 2)
    return values, samples, groups


@pytest.mark.parametrize('test, equal_var', [('pooled', True), ('welch', False)])
def test_t_tests_match_scipy_on_sample_means(test, equal_var):
    values, samples, groups = _hand()
    out = scRL.compare_readout(values, samples, groups, test=test, scale='raw')
    ref = stats.ttest_ind([1, 2, 3], [4, 5, 9], equal_var=equal_var)
    assert out['test'] == test
    assert out['statistic'] == pytest.approx(ref.statistic)
    assert out['p_value'] == pytest.approx(ref.pvalue)
    assert out['estimate'] == pytest.approx(2.0 - 6.0)
    if equal_var:
        assert out['df'] == pytest.approx(4)
    else:
        assert out['df'] == pytest.approx(ref.df)
    assert out['condition_1'] == 'x' and out['condition_2'] == 'y'
    assert out['n_samples'] == (3, 3) and out['n_cells'] == (6, 6)


@pytest.mark.parametrize('test, equal_var', [('pooled', True), ('welch', False)])
def test_t_tests_use_unweighted_sample_means_with_unequal_cells(test, equal_var):
    values, samples, groups = _design([0, 1, 5, 2, 6, 4], cells=[2, 5, 9, 9, 2, 5], sd=1, seed=2)
    means = pd.Series(values).groupby(samples).mean().to_numpy()
    expected = means[:3].mean() - means[3:].mean()
    assert abs(expected - (values[groups == 'a'].mean() - values[groups == 'b'].mean())) > 0.1
    ref = stats.ttest_ind(means[:3], means[3:], equal_var=equal_var)
    out = scRL.compare_readout(values, samples, groups, test=test, scale='raw')
    assert out['n_cells'] == (16, 16)
    assert out['estimate'] == pytest.approx(expected)
    assert out['statistic'] == pytest.approx(ref.statistic)
    assert out['p_value'] == pytest.approx(ref.pvalue)


def test_relabel_complete_separation_six_samples():
    values, samples, groups = _design([0, 1, 2, 10, 11, 12], cells=3, sd=.1)
    with pytest.warns(UserWarning, match='cannot reject'):
        out = scRL.compare_readout(values, samples, groups, test='relabel')
    assert out['p_value'] == pytest.approx(2 / 20)
    assert out['smallest_p'] == pytest.approx(0.1)


def test_relabel_complete_separation_eight_samples():
    values, samples, groups = _design(list(range(4)) + list(range(10, 14)), cells=3, sd=.1)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = scRL.compare_readout(values, samples, groups, test='relabel')
    assert out['p_value'] == pytest.approx(2 / 70)
    assert out['smallest_p'] == pytest.approx(2 / 70)


def test_relabel_unequal_sample_numbers_smallest_p():
    values, samples, groups = _design([0, 1, 2, 3, 10, 11, 12], cells=3, sd=.1, k1=3)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = scRL.compare_readout(values, samples, groups, test='relabel')
    assert out['smallest_p'] == pytest.approx(1 / 35)
    assert out['p_value'] >= out['smallest_p']


def _naive_relabel(values, samples, k1):
    """Welch t of the sample mean ranks (first k1 samples against the rest) and its p-value over every relabelling,
    one relabelling at a time."""
    ranks = stats.rankdata(values)
    mean_rank = pd.Series(ranks).groupby(np.asarray(samples)).mean().to_numpy()
    k = len(mean_rank)

    def t(first):
        first = list(first)
        return stats.ttest_ind(mean_rank[first], np.delete(mean_rank, first), equal_var=False).statistic

    observed = t(range(k1))
    reference = np.array([t(c) for c in itertools.combinations(range(k), k1)])
    return observed, float(np.mean(np.abs(reference) >= abs(observed) - 1e-9))


@pytest.mark.parametrize('k1', [4, 3])
def test_relabel_statistic_is_the_welch_t_of_the_sample_mean_ranks(k1):
    means = [0.3, 1.4, -0.2, 0.9, 0.1, 2.0, 0.6, 1.1]
    values, samples, groups = _design(means, cells=[40, 3, 25, 6, 4, 30, 5, 12], sd=1, seed=4, k1=k1)
    out = scRL.compare_readout(values, samples, groups, test='relabel')
    statistic, p = _naive_relabel(values, samples, k1)
    assert out['statistic'] == pytest.approx(statistic)
    assert out['p_value'] == pytest.approx(p)
    # not the difference of the cell-weighted mean ranks
    ranks = stats.rankdata(values)
    assert abs(out['statistic'] - (ranks[groups == 'a'].mean() - ranks[groups == 'b'].mean())) > 1


def test_relabel_holds_its_level_when_one_condition_has_far_fewer_cells_per_sample():
    # zero-inflated readout with a sample effect and no difference between the conditions, four samples against
    # four, 150 to 250 cells per sample in one condition and 8 to 15 in the other
    rng = np.random.default_rng(2026)
    n_data = 300
    condition = np.repeat(['a', 'b'], 4)
    reject = 0
    for _ in range(n_data):
        sizes = np.r_[rng.integers(150, 251, 4), rng.integers(8, 16, 4)]
        samples = np.repeat(np.arange(8), sizes)
        effect = rng.normal(0, .5, 8)
        values = np.where(rng.random(len(samples)) < .6, 0.0, np.exp(rng.normal(effect[samples], 1.0)))
        reject += scRL.compare_readout(values, samples, condition[samples], test='relabel')['p_value'] < .05
    rate = reject / n_data
    print('relabel null rejection rate, tenfold difference in cells per sample', rate)
    assert rate <= 0.09


def test_relabel_statistic_is_infinite_when_the_sample_mean_ranks_vary_in_neither_condition():
    values, samples, groups = _design([1, 1, 1, 2, 2, 2], cells=1)
    with pytest.warns(UserWarning, match='cannot reject'):
        out = scRL.compare_readout(values, samples, groups, test='relabel')
    assert out['statistic'] == -np.inf
    assert out['p_value'] == pytest.approx(2 / 20)      # the observed arrangement and its mirror image
    with pytest.warns(UserWarning, match='cannot reject'):
        swapped = scRL.compare_readout(values, samples, groups[::-1], test='relabel')
    assert swapped['statistic'] == np.inf and swapped['p_value'] == pytest.approx(2 / 20)


def test_relabel_statistic_is_zero_when_every_sample_has_the_same_mean_rank():
    values = np.tile([0.0, 1.0], 6)
    samples = np.repeat(np.arange(6), 2)
    groups = np.repeat(['a', 'b'], 6)
    with pytest.warns(UserWarning, match='cannot reject'):
        out = scRL.compare_readout(values, samples, groups, test='relabel')
    assert out['statistic'] == 0.0
    assert out['p_value'] == 1.0


def test_relabel_counts_infinite_statistics_as_more_extreme_than_a_finite_one():
    # one value per sample; 2 of the 20 relabellings separate the ones from the twos and have an infinite statistic,
    # the other 18 tie with the observed one in absolute value
    values = np.array([1., 2., 1., 2., 1., 2.])
    with pytest.warns(UserWarning, match='cannot reject'):
        out = scRL.compare_readout(values, np.arange(6), np.repeat(['a', 'b'], 3), test='relabel')
    assert np.isfinite(out['statistic']) and out['statistic'] < 0
    assert out['statistic'] == pytest.approx(stats.ttest_ind([1.5, 5, 1.5], [5, 1.5, 5], equal_var=False).statistic)
    assert out['p_value'] == 1.0


def test_relabel_with_a_single_sample_in_a_condition_uses_the_difference_of_unweighted_mean_ranks():
    # compare_readout refuses such a design, so the rule is tested on the function behind it
    from scRL.ConditionCore import _relabel_test
    values, samples, _ = _design([3.0, 0.2, 1.1, 0.5, 0.9], cells=[30, 4, 8, 5, 6], sd=1, seed=7)
    codes = np.unique(samples, return_inverse=True)[1]
    ranks = stats.rankdata(values)
    mean_rank = np.bincount(codes, weights=ranks) / np.bincount(codes)
    reference = np.array([mean_rank[i] - np.delete(mean_rank, i).mean() for i in range(5)])
    for alone in range(5):
        side = np.ones(5, dtype=int)
        side[alone] = 0
        statistic, p, smallest = _relabel_test(ranks, codes, side, 2000, 0)
        assert statistic == pytest.approx(reference[alone])
        assert p == pytest.approx(np.mean(np.abs(reference) >= abs(reference[alone]) - 1e-9))
        assert smallest == pytest.approx(1 / 5)
        mirrored = _relabel_test(ranks, codes, 1 - side, 2000, 0)
        assert mirrored[0] == pytest.approx(-reference[alone]) and mirrored[1] == pytest.approx(p)


@pytest.mark.parametrize('k1, k, smallest', [(4, 8, 2 / 70), (3, 6, 2 / 20), (3, 7, 1 / 35), (2, 7, 1 / 21),
                                             (5, 7, 1 / 21)])
def test_relabel_smallest_p_is_attained_and_never_undercut(k1, k, smallest):
    rng = np.random.default_rng(k1 * 10 + k)
    sizes = rng.integers(3, 30, k)
    samples = np.repeat(np.arange(k), sizes)
    values = np.repeat(rng.normal(0, 1, k), sizes) + rng.normal(0, 1, sizes.sum())
    ps = []
    for first in itertools.combinations(range(k), k1):
        groups = np.where(np.isin(samples, first), 'a', 'b')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = scRL.compare_readout(values, samples, groups, test='relabel')
        assert out['smallest_p'] == pytest.approx(smallest)
        ps.append(out['p_value'])
    assert min(ps) == pytest.approx(smallest)
    assert max(ps) == pytest.approx(1.0)


def test_relabel_random_branch_agrees_with_exhaustive():
    rng = np.random.default_rng(3)
    means = np.r_[rng.normal(0.6, 1, 5), rng.normal(0, 1, 5)]
    values, samples, groups = _design(means, cells=5, sd=.5, seed=1)
    exact = scRL.compare_readout(values, samples, groups, test='relabel')['p_value']
    rand = scRL.compare_readout(values, samples, groups, test='relabel', n_relabel=200, seed=5)
    assert abs(rand['p_value'] - exact) < 0.12
    assert rand['smallest_p'] == pytest.approx(1 / 201)
    again = scRL.compare_readout(values, samples, groups, test='relabel', n_relabel=200, seed=5)
    assert again['p_value'] == rand['p_value']


def test_relabel_random_branch_counts_the_observed_arrangement():
    # complete separation, one relabelling fewer than the 70 distinct ones, so the random branch runs
    values, samples, groups = _design(list(range(4)) + list(range(10, 14)), cells=3, sd=.1)
    for seed in range(5):
        out = scRL.compare_readout(values, samples, groups, test='relabel', n_relabel=69, seed=seed)
        hits = out['p_value'] * 70
        assert out['smallest_p'] == pytest.approx(1 / 70)
        assert out['p_value'] >= 1 / 70
        assert hits == pytest.approx(round(hits)) and round(hits) >= 1


def test_mixed_detects_effect():
    rng = np.random.default_rng(11)
    values, samples, groups = _clustered(10, 100, .1, rng, shift=1.0)
    out = scRL.compare_readout(values, samples, groups, test='mixed')
    assert out['p_value'] < 0.01
    assert np.sign(out['statistic']) == np.sign(out['estimate']) == 1
    assert out['df'] == 8


def test_mixed_p_value_is_the_t_reference_with_samples_minus_two():
    rng = np.random.default_rng(21)
    values, samples, groups = _clustered(8, 40, .2, rng, shift=.4)
    out = scRL.compare_readout(values, samples, groups, test='mixed')
    assert out['fallback'] is False
    assert out['df'] == 6
    assert out['p_value'] == pytest.approx(2 * stats.t.sf(abs(out['statistic']), 6))


def test_mixed_falls_back_to_sample_means_when_no_sample_variance_is_found():
    # Gaussian cells with no sample effect, four samples against four
    sizes = (15, 40, 8, 125, 20, 47, 90, 61)
    values = np.random.default_rng(3).standard_normal(sum(sizes))
    samples = np.repeat(np.arange(8), sizes)
    groups = np.repeat(list('aaaabbbb'), sizes)
    with pytest.warns(RuntimeWarning, match='t-test on the sample means'):
        out = scRL.compare_readout(values, samples, groups, test='mixed')
    assert out['test'] == 'mixed' and out['fallback'] is True
    scores = stats.norm.ppf((stats.rankdata(values) - .5) / len(values))
    means = np.bincount(samples, weights=scores) / np.bincount(samples)
    ref = stats.ttest_ind(means[:4], means[4:])
    assert out['statistic'] == pytest.approx(ref.statistic)
    assert out['p_value'] == pytest.approx(ref.pvalue)
    assert out['df'] == 6


def test_fallback_is_false_for_the_tests_that_ran_as_asked():
    rng = np.random.default_rng(12)
    v, s, g = _clustered(10, 30, .3, rng)
    for t in ('auto', 'pooled', 'welch', 'relabel', 'mixed'):
        assert scRL.compare_readout(v, s, g, test=t)['fallback'] is False
    with pytest.warns(UserWarning):
        assert scRL.compare_readout(v, s, g, test='cells')['fallback'] is False


def test_mixed_null_rejection_rate():
    rng = np.random.default_rng(2024)
    n_data = 200   # eight samples of 50 cells, within-sample correlation 0.1
    reject = 0
    for _ in range(n_data):
        v, s, g = _clustered(8, 50, .1, rng)
        reject += scRL.compare_readout(v, s, g, test='mixed')['p_value'] < .05
    rate = reject / n_data
    print('mixed null rejection rate', rate)
    assert 0.01 <= rate <= 0.10


def test_mixed_caps_cells_per_sample_and_is_seeded():
    rng = np.random.default_rng(5)
    v, s, g = _clustered(8, 120, .1, rng, shift=.5)
    a = scRL.compare_readout(v, s, g, test='mixed', max_cells=30, seed=1)
    b = scRL.compare_readout(v, s, g, test='mixed', max_cells=30, seed=1)
    assert a['p_value'] == b['p_value']
    assert a['n_cells'] == (480, 480)


def test_mixed_without_statsmodels_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, 'statsmodels.regression.mixed_linear_model', None)
    v, s, g = _design(list(range(8)), cells=3, sd=1)
    with pytest.raises(ImportError, match='statsmodels'):
        scRL.compare_readout(v, s, g, test='mixed')


def test_cells_test_warns_and_matches_mannwhitney():
    rng = np.random.default_rng(8)
    v, s, g = _clustered(10, 60, .2, rng)
    with pytest.warns(UserWarning, match='false') as rec:
        out = scRL.compare_readout(v, s, g, test='cells')
    assert any('cells' in str(r.message) for r in rec)
    a, b = v[g == 'a'], v[g == 'b']
    ref = stats.mannwhitneyu(a, b, method='asymptotic')
    assert out['p_value'] == pytest.approx(ref.pvalue)
    assert out['statistic'] == pytest.approx(ref.statistic)
    with pytest.warns(UserWarning):
        scRL.compare_readout(v, s, g, test='cells')


def test_cells_test_far_smaller_than_pooled_on_clustered_null():
    # one seeded clustered null dataset in which the cell-level test wrongly rejects
    for seed in range(50):
        rng = np.random.default_rng(seed)
        v, s, g = _clustered(10, 100, .3, rng)
        with pytest.warns(UserWarning):
            pc = scRL.compare_readout(v, s, g, test='cells')['p_value']
        if pc < 0.001:
            pp = scRL.compare_readout(v, s, g, test='pooled')['p_value']
            assert pp > 0.05 and pp > 100 * pc
            return
    pytest.fail('no clustered null example with a small cell-level p-value found')


def test_auto_is_pooled_and_suitable_tests():
    for k, expect in [(6, ['pooled']), (8, ['pooled', 'mixed']), (10, ['pooled', 'relabel', 'mixed'])]:
        v, s, g = _design(list(np.arange(k) * 0.3), cells=4, sd=1, seed=k)
        auto = scRL.compare_readout(v, s, g)
        pooled = scRL.compare_readout(v, s, g, test='pooled')
        assert auto['test'] == 'pooled' and auto['scale'] == 'normal'
        assert auto['statistic'] == pooled['statistic'] and auto['p_value'] == pooled['p_value']
        means = _sample_means(_on_scale(v, 'normal'), s)
        ref = stats.ttest_ind(means[:k // 2], means[k // 2:])
        assert auto['statistic'] == pytest.approx(ref.statistic) and auto['p_value'] == pytest.approx(ref.pvalue)
        assert auto['suitable_tests'] == expect
    assert 'mixed' not in scRL.compare_readout(*_design(list(range(6)), sd=1))['suitable_tests']
    assert 'relabel' not in scRL.compare_readout(*_design(list(range(8)), sd=1))['suitable_tests']


def _uneven(k, cells_a, cells_b, seed=0):
    """k samples, half in each condition, with the given numbers of cells per sample in 'a' and in 'b'."""
    cells = [cells_a] * (k // 2) + [cells_b] * (k - k // 2)
    return _design(list(np.arange(k) * 0.3), cells=cells, sd=1, seed=seed)


def test_cells_per_sample_ratio_is_the_larger_mean_over_the_smaller():
    values, samples, groups = _design(list(range(8)), cells=[10, 10, 12, 8, 2, 4, 2, 4], sd=1)
    with pytest.warns(UserWarning):
        out = scRL.compare_readout(values, samples, groups, test='pooled')
    assert out['n_cells'] == (40, 12)
    assert out['cells_per_sample_ratio'] == pytest.approx(10 / 3)
    with pytest.warns(UserWarning):
        swapped = scRL.compare_readout(values, samples, np.where(groups == 'a', 'b', 'a'), test='pooled')
    assert swapped['cells_per_sample_ratio'] == pytest.approx(10 / 3)
    assert type(out['cells_per_sample_ratio']) is float
    even = scRL.compare_readout(*_design(list(range(6)), cells=3, sd=1))
    assert even['cells_per_sample_ratio'] == 1.0
    one_each = scRL.compare_readout(np.arange(8.0) ** 2, np.arange(8), np.repeat(['a', 'b'], 4))
    assert one_each['cells_per_sample_ratio'] == 1.0


def test_auto_runs_welch_without_a_warning_when_cells_per_sample_differ_threefold():
    for cells_b, ratio in ((5, 10), (2, 3)):
        values, samples, groups = _uneven(8, cells_b * ratio, cells_b)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = scRL.compare_readout(values, samples, groups)
        assert out['test'] == 'welch' and out['suitable_tests'][0] == 'welch' and out['scale'] == 'normal'
        means = _sample_means(_on_scale(values, 'normal'), samples)
        ref = stats.ttest_ind(means[:4], means[4:], equal_var=False)
        assert out['statistic'] == pytest.approx(ref.statistic) and out['p_value'] == pytest.approx(ref.pvalue)
        raw = scRL.compare_readout(values, samples, groups, scale='raw')
        means = pd.Series(values).groupby(samples).mean().to_numpy()
        assert raw['test'] == 'welch' and raw['scale'] == 'raw'
        assert raw['p_value'] == pytest.approx(stats.ttest_ind(means[:4], means[4:], equal_var=False).pvalue)
    values, samples, groups = _uneven(8, 29, 10)          # below threefold: the pooled t-test
    assert scRL.compare_readout(values, samples, groups)['test'] == 'pooled'


@pytest.mark.parametrize('test', ['pooled'])
def test_pooled_warns_when_cells_per_sample_differ_between_conditions_and_still_runs(test):
    values, samples, groups = _uneven(8, 50, 5)
    with pytest.warns(UserWarning, match='differ about 10-fold') as rec:
        out = scRL.compare_readout(values, samples, groups, test=test, scale='raw')
    message = ' '.join(str(r.message) for r in rec)
    assert "test='welch'" in message and "test='mixed'" in message and 'tenfold' in message
    assert out['test'] == 'pooled'
    means = pd.Series(values).groupby(samples).mean().to_numpy()
    assert out['p_value'] == pytest.approx(stats.ttest_ind(means[:4], means[4:]).pvalue)
    values, samples, groups = _uneven(8, 6, 2)          # exactly threefold
    with pytest.warns(UserWarning, match='differ about 3-fold'):
        assert scRL.compare_readout(values, samples, groups, test=test)['test'] == 'pooled'


def test_no_cells_per_sample_warning_below_threefold_or_for_the_other_tests():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        values, samples, groups = _uneven(8, 29, 10)
        for t in ('auto', 'pooled'):
            assert scRL.compare_readout(values, samples, groups, test=t)['cells_per_sample_ratio'] == pytest.approx(2.9)
        values, samples, groups = _uneven(8, 50, 5)
        for t in ('welch', 'relabel'):
            assert scRL.compare_readout(values, samples, groups, test=t)['cells_per_sample_ratio'] == pytest.approx(10)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        scRL.compare_readout(values, samples, groups, test='mixed')
        with pytest.warns(UserWarning, match='independent unit'):
            scRL.compare_readout(values, samples, groups, test='cells')
    assert not any('differ about' in str(r.message) for r in rec)


def test_suitable_tests_name_welch_in_place_of_pooled_when_cells_per_sample_differ_threefold():
    for k, expect in [(6, ['welch']), (8, ['welch', 'mixed']), (10, ['welch', 'relabel', 'mixed'])]:
        for cells_a, cells_b in ((6, 2), (2, 6), (40, 4)):
            with pytest.warns(UserWarning, match='differ about'):
                out = scRL.compare_readout(*_uneven(k, cells_a, cells_b, seed=k), test='pooled')
            assert out['test'] == 'pooled'
            assert out['suitable_tests'] == expect
            assert scRL.compare_readout(*_uneven(k, cells_a, cells_b, seed=k))['test'] == 'welch'
            assert scRL.compare_readout(*_uneven(k, cells_a, cells_b, seed=k), test='welch')['suitable_tests'] == expect
        below = scRL.compare_readout(*_uneven(k, 29, 10, seed=k))
        assert below['suitable_tests'] == ['pooled'] + expect[1:]


def test_level_of_pooled_and_relabel_under_clustered_null():
    rng = np.random.default_rng(99)
    n_data = 300
    hits = {'pooled': 0, 'relabel': 0}
    for _ in range(n_data):
        v, s, g = _clustered(10, 30, .2, rng)
        for t in hits:
            hits[t] += scRL.compare_readout(v, s, g, test=t)['p_value'] < .05
    for t, h in hits.items():
        print(t, 'null rejection rate', h / n_data)
        assert 0.02 <= h / n_data <= 0.09


def test_missing_entries_are_dropped_and_counted():
    v, s, g = _design(list(range(6)), cells=3, sd=1)
    v = pd.Series(v).astype(float)
    s = pd.Series(s, dtype=object)
    g = pd.Series(g, dtype=object)
    v[0] = np.nan
    s[1] = None
    g[2] = np.nan
    out = scRL.compare_readout(v, s, g, test='pooled')
    assert out['n_dropped'] == 3
    assert sum(out['n_cells']) == 15


def test_input_errors():
    v, s, g = _design(list(range(6)), cells=3, sd=1)
    with pytest.raises(ValueError):                       # sample in both conditions
        scRL.compare_readout(v, s, np.where(np.arange(18) == 0, 'b', g))
    with pytest.raises(ValueError):                       # three conditions
        scRL.compare_readout(v, s, np.repeat(list('aabbcc'), 3))
    with pytest.raises(ValueError):                       # one condition
        scRL.compare_readout(v, s, np.repeat('a', 18))
    with pytest.raises(ValueError):                       # a single sample in a condition
        scRL.compare_readout(v, s, np.repeat(['a', 'b', 'b', 'b', 'b', 'b'], 3))
    for bad in ('t', 'Pooled'):
        with pytest.raises(ValueError):
            scRL.compare_readout(v, s, g, test=bad)
    for alpha in (0, 1, -.1, 2):
        with pytest.raises(ValueError):
            scRL.compare_readout(v, s, g, alpha=alpha)
    for bad in (0, -3, 2.5, 200.0, True, None, '5'):
        with pytest.raises(ValueError, match='n_relabel must be a positive integer'):
            scRL.compare_readout(v, s, g, n_relabel=bad)
        with pytest.raises(ValueError, match='max_cells must be a positive integer'):
            scRL.compare_readout(v, s, g, max_cells=bad)
    assert scRL.compare_readout(v, s, g, n_relabel=np.int64(5), max_cells=np.int32(2))['test'] == 'pooled'


def test_infinite_values_are_refused():
    v, s, g = _design(list(range(6)), cells=3, sd=1)
    for bad in (np.inf, -np.inf):
        w = v.copy()
        w[4] = bad
        with pytest.raises(ValueError, match='infinite'):
            scRL.compare_readout(w, s, g)


def test_equal_values_are_refused_by_every_test():
    _, s, g = _design(list(range(8)), cells=5)
    for t in ('auto', 'pooled', 'welch', 'relabel', 'mixed', 'cells'):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(ValueError, match='all equal'):
                scRL.compare_readout(np.ones(40), s, g, test=t)


@pytest.mark.parametrize('test', ['pooled', 'welch'])
def test_t_tests_refuse_sample_means_without_spread_in_either_condition(test):
    values, samples, groups = _design([1, 1, 1, 2, 2, 2], cells=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='do not vary'):
            scRL.compare_readout(values, samples, groups, test=test)


def test_one_value_per_sample():
    rng = np.random.default_rng(6)
    values = rng.normal(size=8) + np.repeat([0.0, 1.0], 4)
    samples = np.arange(8)
    groups = np.repeat(['a', 'b'], 4)
    for t, equal_var in (('pooled', True), ('welch', False)):
        out = scRL.compare_readout(values, samples, groups, test=t, scale='raw')
        ref = stats.ttest_ind(values[:4], values[4:], equal_var=equal_var)
        assert out['statistic'] == pytest.approx(ref.statistic)
        assert out['p_value'] == pytest.approx(ref.pvalue)
        assert out['design_effect'] is None
        assert out['n_cells'] == (4, 4)
    out = scRL.compare_readout(values, samples, groups, test='relabel')
    assert out['design_effect'] is None
    assert 2 / 70 <= out['p_value'] <= 1
    for t in ('cells', 'mixed'):
        with pytest.raises(ValueError, match='more than one cell per sample'):
            scRL.compare_readout(values, samples, groups, test=t)


def test_a_sample_with_a_single_cell_is_kept():
    values, samples, groups = _design(list(range(8)), cells=[1, 4, 4, 4, 4, 4, 4, 4], sd=.5)
    for t in ('pooled', 'welch', 'relabel', 'mixed'):
        out = scRL.compare_readout(values, samples, groups, test=t)
        assert out['n_samples'] == (4, 4) and out['n_cells'] == (13, 16)
        assert out['design_effect']['n_samples'] == 8
        assert np.isfinite(out['p_value'])


def test_categorical_groups_fix_the_order():
    v, s, g = _design(list(range(6)), cells=3, sd=1)
    default = scRL.compare_readout(v, s, g)
    cat = pd.Series(pd.Categorical(g, categories=['b', 'a']))
    out = scRL.compare_readout(v, s, cat)
    assert (default['condition_1'], default['condition_2']) == ('a', 'b')
    assert (out['condition_1'], out['condition_2']) == ('b', 'a')
    assert out['estimate'] == pytest.approx(-default['estimate'])
    assert out['n_samples'] == (3, 3)


@pytest.mark.parametrize('wrap', [lambda c: c, pd.Series, pd.CategoricalIndex])
def test_category_order_is_honoured_and_unused_categories_are_ignored(wrap):
    v, s, g = _design(list(range(6)), cells=3, sd=1)
    default = scRL.compare_readout(v, s, g)
    out = scRL.compare_readout(v, s, wrap(pd.Categorical(g, categories=['c', 'b', 'a', 'd'])))
    assert (out['condition_1'], out['condition_2']) == ('b', 'a')
    assert out['estimate'] == pytest.approx(-default['estimate'])
    assert out['n_samples'] == (3, 3)


def test_estimate_on_the_readouts_own_scale_for_every_test():
    v, s, g = _design(list(range(8)), cells=3, sd=.5)
    means = pd.Series(v).groupby(s).mean()
    expected = means[:4].mean() - means[4:].mean()
    for t in ('pooled', 'welch', 'relabel', 'mixed'):
        out = scRL.compare_readout(v, s, g, test=t)
        assert out['estimate'] == pytest.approx(expected)
    with pytest.warns(UserWarning):
        assert scRL.compare_readout(v, s, g, test='cells')['estimate'] == pytest.approx(expected)


def test_returned_design_effect_matches_function_on_kept_cells():
    rng = np.random.default_rng(4)
    v, s, g = _clustered(8, 20, .2, rng)
    v = v.copy(); v[:3] = np.nan
    out = scRL.compare_readout(v, s, g)
    ref = scRL.design_effect(v[3:], s[3:], groups=g[3:])
    assert out['design_effect'] == ref
    assert isinstance(out, dict) and isinstance(out['p_value'], float)


def test_returned_design_effect_is_estimated_within_conditions():
    rng = np.random.default_rng(9)
    v, s, g = _clustered(8, 60, .05, rng, shift=3.0)
    out = scRL.compare_readout(v, s, g)
    within = scRL.design_effect(v, s, groups=g)
    plain = scRL.design_effect(v, s)
    assert out['design_effect']['within_groups'] is True
    assert out['design_effect']['icc'] == within['icc']
    assert out['design_effect']['icc'] < .5 * plain['icc']
    with pytest.warns(UserWarning) as rec:
        cells = scRL.compare_readout(v, s, g, test='cells')
    assert cells['design_effect'] == within
    message = next(str(r.message) for r in rec if 'independent unit' in str(r.message))
    assert f"{within['predicted_false_call_rate']:.3g}" in message
    assert f"{plain['predicted_false_call_rate']:.3g}" not in message


SCALES = ('normal', 'rank', 'raw')


def _skewed():
    """Six samples with unequal numbers of cells, a skewed readout and one tied pair of cells."""
    values, samples, groups = _design([0, 1, 5, 2, 6, 4], cells=[2, 5, 9, 9, 2, 5], sd=1, seed=2)
    values = np.exp(values / 2)
    values[20] = values[3]
    return values, samples, groups


@pytest.mark.parametrize('scale', SCALES)
@pytest.mark.parametrize('test, equal_var', [('pooled', True), ('welch', False)])
def test_t_tests_match_scipy_on_sample_means_of_the_named_scale(test, equal_var, scale):
    values, samples, groups = _skewed()
    means = _sample_means(_on_scale(values, scale), samples)
    ref = stats.ttest_ind(means[:3], means[3:], equal_var=equal_var)
    out = scRL.compare_readout(values, samples, groups, test=test, scale=scale)
    assert out['test'] == test and out['scale'] == scale
    assert out['n_cells'] == (16, 16)
    assert out['statistic'] == pytest.approx(ref.statistic)
    assert out['p_value'] == pytest.approx(ref.pvalue)
    assert out['df'] == pytest.approx(4 if equal_var else ref.df)


@pytest.mark.parametrize('test', ['pooled', 'welch'])
def test_the_three_scales_give_three_different_tests(test):
    values, samples, groups = _skewed()
    t = [scRL.compare_readout(values, samples, groups, test=test, scale=scale)['statistic'] for scale in SCALES]
    assert min(abs(t[0] - t[1]), abs(t[0] - t[2]), abs(t[1] - t[2])) > 1e-3


def test_default_scale_is_normal_and_auto_reports_it():
    values, samples, groups = _skewed()
    means = _sample_means(_on_scale(values, 'normal'), samples)
    ref = stats.ttest_ind(means[:3], means[3:])
    for kwargs in ({}, {'test': 'auto'}, {'test': 'pooled'}, {'scale': 'normal'}):
        out = scRL.compare_readout(values, samples, groups, **kwargs)
        assert out['test'] == 'pooled' and out['scale'] == 'normal'
        assert out['statistic'] == pytest.approx(ref.statistic)
        assert out['p_value'] == pytest.approx(ref.pvalue)
    ref = stats.ttest_ind(means[:3], means[3:], equal_var=False)
    out = scRL.compare_readout(values, samples, groups, test='welch')
    assert out['scale'] == 'normal' and out['p_value'] == pytest.approx(ref.pvalue)
    for scale in SCALES:
        assert scRL.compare_readout(values, samples, groups, scale=scale)['scale'] == scale


@pytest.mark.parametrize('scale', SCALES)
@pytest.mark.parametrize('test', ['auto', 'pooled', 'welch'])
def test_estimate_is_the_raw_difference_and_estimate_scaled_is_on_the_scale_of_the_test(test, scale):
    values, samples, groups = _skewed()
    raw = _sample_means(values, samples)
    scaled = _sample_means(_on_scale(values, scale), samples)
    out = scRL.compare_readout(values, samples, groups, test=test, scale=scale)
    assert out['estimate'] == pytest.approx(raw[:3].mean() - raw[3:].mean())
    assert out['estimate_scaled'] == pytest.approx(scaled[:3].mean() - scaled[3:].mean())
    assert type(out['estimate_scaled']) is float
    if scale == 'raw':
        assert out['estimate_scaled'] == out['estimate']
    else:
        assert abs(out['estimate_scaled'] - out['estimate']) > 0.1


@pytest.mark.parametrize('test', ['relabel', 'mixed', 'cells'])
def test_scale_is_none_and_has_no_effect_for_the_tests_that_ignore_it(test):
    rng = np.random.default_rng(12)
    v, s, g = _clustered(10, 30, .3, rng, shift=.5)
    results = []
    for scale in SCALES:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results.append(scRL.compare_readout(np.exp(v), s, g, test=test, scale=scale))
    raw = _sample_means(np.exp(v), s)
    for out in results:
        assert out['test'] == test
        assert out['scale'] is None and out['estimate_scaled'] is None
        assert out['estimate'] == pytest.approx(raw[:5].mean() - raw[5:].mean())
        assert out['statistic'] == results[0]['statistic'] and out['p_value'] == results[0]['p_value']


def test_unknown_scale_is_refused_whatever_the_test():
    v, s, g = _design(list(range(8)), cells=3, sd=1)
    for bad in ('ranks', 'Normal', 'log', None, 1):
        for t in ('auto', 'pooled', 'welch', 'relabel', 'mixed', 'cells'):
            with pytest.raises(ValueError, match=r"scale must be one of \('normal', 'rank', 'raw'\)"):
                scRL.compare_readout(v, s, g, test=t, scale=bad)


@pytest.mark.parametrize('scale', SCALES)
def test_one_value_per_sample_under_each_scale(scale):
    rng = np.random.default_rng(6)
    values = np.exp(rng.normal(size=9) + np.repeat([0.0, 1.0], [4, 5]))
    values[7] = values[2]                                # a tie between two samples
    samples = np.arange(9)
    groups = np.repeat(['a', 'b'], [4, 5])
    z = _on_scale(values, scale)
    for t, equal_var in (('pooled', True), ('welch', False)):
        out = scRL.compare_readout(values, samples, groups, test=t, scale=scale)
        ref = stats.ttest_ind(z[:4], z[4:], equal_var=equal_var)
        assert out['scale'] == scale
        assert out['statistic'] == pytest.approx(ref.statistic)
        assert out['p_value'] == pytest.approx(ref.pvalue)
        assert out['estimate'] == pytest.approx(values[:4].mean() - values[4:].mean())
        assert out['estimate_scaled'] == pytest.approx(z[:4].mean() - z[4:].mean())
        assert out['design_effect'] is None and out['n_cells'] == (4, 5)


@pytest.mark.parametrize('test', ['pooled', 'welch'])
def test_a_monotone_transform_of_the_readout_changes_the_raw_test_only(test):
    values, samples, groups = _design([0, 1, 5, 2, 6, 4], cells=[2, 5, 9, 9, 2, 5], sd=1, seed=2)
    for scale in ('rank', 'normal'):
        plain = scRL.compare_readout(values, samples, groups, test=test, scale=scale)
        moved = scRL.compare_readout(np.exp(values), samples, groups, test=test, scale=scale)
        assert moved['statistic'] == pytest.approx(plain['statistic'], rel=1e-12)
        assert moved['p_value'] == pytest.approx(plain['p_value'], rel=1e-12)
        assert moved['estimate_scaled'] == pytest.approx(plain['estimate_scaled'], rel=1e-12)
        assert abs(moved['estimate'] - plain['estimate']) > 1
    plain = scRL.compare_readout(values, samples, groups, test=test, scale='raw')
    moved = scRL.compare_readout(np.exp(values), samples, groups, test=test, scale='raw')
    assert abs(moved['statistic'] - plain['statistic']) > 0.01
    assert abs(moved['p_value'] - plain['p_value']) > 0.01


@pytest.mark.parametrize('test', ['pooled', 'welch'])
def test_the_check_for_sample_means_without_spread_is_made_on_the_scale_of_the_test(test):
    # raw sample means 1, 1 against 6, 6; the mean ranks and mean normal scores of the samples differ within conditions
    values = np.array([0, 0, 3, 1, 1, 1, 4, 4, 10, 6, 6, 6], dtype=float)
    samples = np.repeat(list('abcd'), 3)
    groups = np.repeat(['x', 'y'], 6)
    with pytest.raises(ValueError, match='do not vary'):
        scRL.compare_readout(values, samples, groups, test=test, scale='raw')
    for scale in ('rank', 'normal'):
        means = _sample_means(_on_scale(values, scale), samples)
        ref = stats.ttest_ind(means[:2], means[2:], equal_var=test == 'pooled')
        out = scRL.compare_readout(values, samples, groups, test=test, scale=scale)
        assert out['p_value'] == pytest.approx(ref.pvalue)
        assert out['estimate'] == pytest.approx(1.0 - 6.0)
    # one value per sample: equal within each condition on every scale
    values, samples, groups = _design([1, 1, 1, 2, 2, 2], cells=1)
    for scale in SCALES:
        with pytest.raises(ValueError, match='do not vary'):
            scRL.compare_readout(values, samples, groups, test=test, scale=scale)


def test_importable_from_top_level():
    from scRL import compare_readout
    assert compare_readout is scRL.compare_readout
    assert 'compare_readout' in scRL.__all__

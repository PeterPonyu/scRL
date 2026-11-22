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
    out = scRL.compare_readout(values, samples, groups, test=test)
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
    out = scRL.compare_readout(values, samples, groups, test=test)
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
        assert auto['test'] == 'pooled'
        assert auto['statistic'] == pooled['statistic'] and auto['p_value'] == pooled['p_value']
        assert auto['suitable_tests'] == expect
    assert 'mixed' not in scRL.compare_readout(*_design(list(range(6)), sd=1))['suitable_tests']
    assert 'relabel' not in scRL.compare_readout(*_design(list(range(8)), sd=1))['suitable_tests']


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
        out = scRL.compare_readout(values, samples, groups, test=t)
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
    ref = scRL.design_effect(v[3:], s[3:])
    assert out['design_effect'] == ref
    assert isinstance(out, dict) and isinstance(out['p_value'], float)


def test_importable_from_top_level():
    from scRL import compare_readout
    assert compare_readout is scRL.compare_readout
    assert 'compare_readout' in scRL.__all__

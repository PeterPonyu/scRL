"""The worked example of the protocol for comparing a per-cell readout between two conditions
(examples/protocol_example.py) reports what the protocol asks for, and its results for seed 0 are fixed."""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import scRL

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'protocol_example.py'

KEYS = {'cells', 'animals', 'conditions', 'test', 'statistic', 'df', 'estimate', 'ci_low', 'ci_high', 'p_value',
        'n_animals', 'n_cells', 'icc', 'design_effect', 'implied_false_positive_rate', 'readouts'}
READOUTS = ['readout', 'null_1', 'null_2', 'null_3', 'null_4', 'null_5']

# The example's results for seed 0, computed once and fixed here, so that its printed output can be checked.
SEED_0 = {'estimate': 0.3726202904708526, 'ci_low': 0.14825376514954622, 'ci_high': 0.596986815792159,
          'p_value': 0.006622403186249988}


def _load():
    spec = importlib.util.spec_from_file_location('protocol_example', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def example():
    return _load()


@pytest.fixture(scope='module')
def result(example):
    return example.run_example(seed=0)


def _by_condition(result, column):
    """A column of the per-animal table, one array per condition, in the order of result['conditions']."""
    animals = result['animals']
    return [animals.loc[animals['condition'] == c, column].to_numpy() for c in result['conditions']]


def test_the_example_returns_the_reported_quantities(result):
    assert set(result) == KEYS
    assert result['conditions'] == ('case', 'control')
    assert result['n_animals'] == (4, 4) and result['n_cells'] == (1600, 1600)
    assert len(result['cells']) == 3200 and len(result['animals']) == 8
    assert list(result['readouts'].index) == READOUTS


def test_every_readout_is_computed_on_all_cells_of_both_conditions_together(result):
    # step 1: one scale for every cell, set on all cells at once (mean 0 and standard deviation 1 over all cells)
    cells = result['cells']
    for name in READOUTS:
        assert cells[name].mean() == pytest.approx(0, abs=1e-12)
        assert cells[name].std() == pytest.approx(1, abs=1e-12)


def test_the_estimate_is_the_difference_of_the_animal_means_and_lies_inside_its_interval(result):
    cells = result['cells']
    for animal, row in result['animals'].iterrows():
        assert row['mean'] == pytest.approx(cells.loc[cells['animal'] == animal, 'readout'].mean(), abs=1e-12)
    case, control = _by_condition(result, 'mean')
    assert result['estimate'] == pytest.approx(case.mean() - control.mean(), abs=1e-12)
    assert result['ci_low'] < result['estimate'] < result['ci_high']


def test_the_test_is_the_pooled_t_test_of_the_raw_animal_means(result):
    # four animals against four with 400 cells each: test='auto' runs the pooled t-test on the animal means
    case, control = _by_condition(result, 'mean')
    ref = stats.ttest_ind(case, control, equal_var=True)
    ci = ref.confidence_interval(confidence_level=0.95)
    assert result['test'] == 'pooled' and result['df'] == 6
    assert result['statistic'] == pytest.approx(ref.statistic, abs=1e-10)
    assert result['p_value'] == pytest.approx(ref.pvalue, abs=1e-12)
    assert (result['ci_low'], result['ci_high']) == pytest.approx((ci.low, ci.high), abs=1e-12)


def test_leaving_one_animal_out_repeats_the_comparison_without_that_animal(result):
    animals = result['animals']
    for animal in animals.index:
        rest = animals.drop(index=animal)
        a, b = (rest.loc[rest['condition'] == c, 'mean'] for c in result['conditions'])
        assert animals.loc[animal, 'difference_without'] == pytest.approx(a.mean() - b.mean(), abs=1e-12)
        # three animals against four: test='auto' runs Welch's t-test
        assert animals.loc[animal, 'p_without'] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue,
                                                                 abs=1e-12)


def test_the_within_animal_correlation_is_estimated_within_conditions_and_is_of_the_simulated_order(result):
    cells = result['cells']
    within = scRL.design_effect(cells['readout'], cells['animal'], groups=cells['condition'])
    assert result['icc'] == within['icc'] and result['design_effect'] == within['design_effect']
    assert 0.005 < result['icc'] < 0.1                 # simulated: 0.03
    # the rate at which a test over cells would call readouts that do not differ lies above its level of 0.05
    assert result['implied_false_positive_rate'] > 0.05


def test_the_p_values_of_the_six_readouts_are_corrected_by_benjamini_hochberg(result):
    table = result['readouts']
    p = table['p_value'].to_numpy()
    m = len(p)
    order = np.argsort(p)
    stepped = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(stepped, 1)
    assert table['q_value'].to_numpy() == pytest.approx(q, abs=1e-12)
    assert (table['q_value'] >= table['p_value']).all()
    assert table.loc['readout', 'p_value'] == result['p_value']
    assert table.loc['readout', 'estimate'] == result['estimate']


def test_a_rank_sum_test_over_cells_gives_a_p_value_far_below_that_of_the_animal_means(result):
    cells = result['cells']
    case, control = (cells.loc[cells['condition'] == c, 'readout'] for c in result['conditions'])
    over_cells = stats.mannwhitneyu(case, control).pvalue
    # Cells of one animal resemble each other, so a test that counts every cell as a unit overstates the evidence.
    # Expected direction: its p-value lies far below that of the test of the animal means (by more than a factor
    # of 1000; its value is not fixed).
    assert over_cells < result['p_value'] / 1000


def test_a_group_of_two_animals_is_tested_with_a_warning_and_a_group_of_one_is_refused(result):
    # compare_readout documents that each condition needs at least two samples, and refuses fewer; a group of two
    # is accepted with a UserWarning (its guidance says to avoid one)
    cells = result['cells']

    def keep(*animals):
        part = cells[cells['animal'].isin(animals)]
        return part['readout'], part['animal'], part['condition']

    with pytest.warns(UserWarning, match='group of two is not recommended'):
        two = scRL.compare_readout(*keep('case1', 'case2', 'control1', 'control2'), test='auto')
    with pytest.warns(UserWarning, match='group of two is not recommended'):
        unequal = scRL.compare_readout(*keep('case1', 'case2', 'control1', 'control2', 'control3', 'control4'),
                                       test='auto')
    assert two['test'] == 'pooled' and two['n_samples'] == (2, 2) and two['df'] == 2
    assert unequal['test'] == 'welch' and unequal['n_samples'] == (2, 4)
    with pytest.raises(ValueError, match='each condition needs at least two samples; found 1 and 3'):
        scRL.compare_readout(*keep('case1', 'control1', 'control2', 'control3'), test='auto')


def test_the_printed_report_states_the_test_the_difference_and_its_interval(example, result, capsys):
    example.report(result)
    text = capsys.readouterr().out
    assert 'pooled t-test' in text
    for key in ('estimate', 'ci_low', 'ci_high'):
        assert f'{result[key]:+.3f}' in text


def test_the_results_for_seed_0_are_fixed(result):
    for key, value in SEED_0.items():
        assert result[key] == pytest.approx(value, abs=1e-10), key

"""A worked example of the protocol for comparing a per-cell readout between two conditions.

The data are simulated: two conditions ('case' and 'control') of four animals each, 400 cells per animal, and six
readouts per cell. The first readout is shifted in the case animals; the other five do not differ between the
conditions and are there for the step that corrects for testing many readouts. Cells of one animal resemble each
other (a within-animal correlation of 0.03), which is why the animal, not the cell, is the unit of the comparison.

Run from the repository root:

    python examples/protocol_example.py

The documentation page "Protocol: comparing a per-cell readout between conditions" walks through it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # use the scRL package of this repository
import scRL

ANIMALS = 4             # animals per condition
CELLS = 400             # cells per animal
ICC = 0.03              # within-animal correlation of every readout
SHIFT = 0.4             # effect of the condition on the first readout, in standard deviations of the cells
READOUTS = ['readout'] + [f'null_{i}' for i in range(1, 6)]     # the five further readouts have no effect


def simulate(seed=0):
    """One row per cell: its animal, its condition and the raw value of each readout."""
    rng = np.random.default_rng(seed)
    animals = [f'{c}{i}' for c in ('case', 'control') for i in range(1, ANIMALS + 1)]
    cells = pd.DataFrame({'animal': np.repeat(animals, CELLS),
                          'condition': np.repeat(['case', 'control'], ANIMALS * CELLS)})
    for name in READOUTS:
        # every animal moves all its cells by one amount (variance ICC) and every cell adds its own noise
        # (variance 1 - ICC), so two cells of the same animal correlate by ICC
        animal_shift = rng.normal(0, np.sqrt(ICC), len(animals))
        cells[name] = np.repeat(animal_shift, CELLS) + rng.normal(0, np.sqrt(1 - ICC), len(cells))
    cells['readout'] += SHIFT * (cells['condition'] == 'case')     # the condition effect, first readout only
    return cells


def compare(cells, readout='readout'):
    """Test the animal means of one readout. test='auto' runs the pooled t-test when both conditions have the same
    number of animals and similar numbers of cells per animal (less than threefold apart), and Welch's t-test
    otherwise; scale='raw' tests the animals' mean readout itself."""
    return scRL.compare_readout(cells[readout], cells['animal'], cells['condition'], test='auto', scale='raw')


def run_example(seed=0):
    """Run the four steps of the protocol on simulated data and return the quantities it reports."""
    cells = simulate(seed)

    # Step 1. Compute each readout once, on all cells of both conditions together, so that both conditions are on
    # one scale. Here the raw values are centred and scaled over all cells; a pseudotime, a fate probability or a
    # signature score is likewise computed on all cells at once. Computing or scaling it within each condition
    # would remove the very difference to be tested.
    for name in READOUTS:
        cells[name] = (cells[name] - cells[name].mean()) / cells[name].std()

    # Step 2. Average the readout per animal and look at the animal means. With four animals per condition one
    # outlying animal can decide the result, so repeat the comparison with each animal left out in turn.
    animals = cells.groupby('animal').agg(condition=('condition', 'first'), mean=('readout', 'mean'))
    without = [compare(cells[cells['animal'] != animal]) for animal in animals.index]
    animals['difference_without'] = [r['estimate'] for r in without]
    animals['p_without'] = [r['p_value'] for r in without]

    # Step 3. Test the animal means: the pooled t-test here, four animals against four with 400 cells each. Never
    # compare a group of two animals; compare_readout refuses a group of one and warns for a group of two.
    res = compare(cells)

    # Step 4. Report the difference of the animal means (case minus control: the conditions are taken in
    # alphabetical order) with its 95% interval, the animals and cells per condition, and the within-animal
    # correlation, estimated within conditions so that the difference between them is not counted as resemblance.
    within = scRL.design_effect(cells['readout'], cells['animal'], groups=cells['condition'])
    # With several readouts, correct their animal-level p-values for the number tested (Benjamini-Hochberg).
    tests = [compare(cells, name) for name in READOUTS]
    readouts = pd.DataFrame({'estimate': [t['estimate'] for t in tests], 'p_value': [t['p_value'] for t in tests]},
                            index=READOUTS)
    readouts['q_value'] = stats.false_discovery_control(readouts['p_value'])
    return {'cells': cells, 'animals': animals, 'conditions': (res['condition_1'], res['condition_2']),
            'test': res['test'], 'statistic': res['statistic'], 'df': res['df'], 'estimate': res['estimate'],
            'ci_low': res['ci_low'], 'ci_high': res['ci_high'], 'p_value': res['p_value'],
            'n_animals': res['n_samples'], 'n_cells': res['n_cells'],
            'icc': within['icc'], 'design_effect': within['design_effect'],
            # the share of readouts that do not differ at all which a test over cells would call at level 0.05:
            # it explains why cells are not the unit, and never permits a test over cells, however small it is
            'implied_false_positive_rate': within['implied_false_positive_rate'], 'readouts': readouts}


def report(r):
    """Print the reported quantities as a short plain-text report."""
    first, second = r['conditions']
    test = {'pooled': 'pooled t-test', 'welch': "Welch's t-test"}[r['test']]
    print(f'Simulated: two conditions of {ANIMALS} animals, {CELLS} cells per animal, within-animal correlation '
          f'{ICC};\nthe first readout is shifted by {SHIFT} in the case animals.')
    print(f"Step 1. Six readouts per cell, computed on all {len(r['cells'])} cells of both conditions together.")
    print('Step 2. Animal means of the readout, and the comparison repeated without each animal:')
    print(r['animals'].to_string(formatters={'mean': '{:+.3f}'.format, 'difference_without': '{:+.3f}'.format,
                                             'p_without': '{:.3g}'.format}))
    print(f"Step 3. The {test} of the animal means: t = {r['statistic']:.2f} on {r['df']:.0f} df, "
          f"p = {r['p_value']:.3g}.")
    print(f"Step 4. {first} - {second}: {r['estimate']:+.3f} (95% interval {r['ci_low']:+.3f} to "
          f"{r['ci_high']:+.3f});\n  {r['n_animals'][0]} and {r['n_animals'][1]} animals, {r['n_cells'][0]} and "
          f"{r['n_cells'][1]} cells.")
    print(f"  Within-animal correlation, within conditions: {r['icc']:.3f} (design effect "
          f"{r['design_effect']:.1f}).\n  The design implies that a test over cells would call about "
          f"{r['implied_false_positive_rate']:.0%} of readouts that do not differ at all (level 0.05).")
    print('  The six readouts, animal-level p-values and Benjamini-Hochberg q-values:')
    print(r['readouts'].to_string(formatters={'estimate': '{:+.3f}'.format, 'p_value': '{:.3g}'.format,
                                              'q_value': '{:.3g}'.format}))


if __name__ == '__main__':
    report(run_example(seed=0))

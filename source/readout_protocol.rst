Protocol: comparing a per-cell readout between conditions
=========================================================

A per-cell readout, such as a pseudotime, a fate probability or a signature score, is often compared between two
conditions that each hold a few animals (or donors). Cells of one animal resemble each other more than they
resemble cells of another animal, so they are not independent observations: the animal, not the cell, is the unit
of the comparison. The protocol has four steps.

1. Compute the readout once, on all cells of both conditions together.
2. Average it per animal and look at the animal means. One outlying animal can decide a result, so repeat the
   comparison with each animal left out in turn.
3. Test the animal means with ``scRL.compare_readout(values, samples, groups, test='auto')``: the pooled t-test
   when both conditions have the same number of animals and similar numbers of cells per animal, Welch's t-test
   otherwise, on the raw animal means. Never compare a group of two animals.
4. Report the difference of the animal means with its 95% interval, the animals and cells per condition, and the
   within-animal correlation γ, estimated within conditions by
   ``scRL.design_effect(values, samples, groups=groups)``. With many readouts, correct the animal-level p-values
   for their number (Benjamini-Hochberg, ``scipy.stats.false_discovery_control``).

The worked example ``examples/protocol_example.py`` runs these steps on simulated data and prints a short report.
From the repository root:

.. code-block:: bash

    python examples/protocol_example.py

Its function ``run_example(seed=0)`` returns the reported quantities in a dict. The output shown below is the one
for seed 0; ``tests/test_protocol_example.py`` fixes its difference, interval and p-value, so the output can be
checked.

Simulated data
--------------

Two conditions, ``'case'`` and ``'control'``, of four animals each, with 400 cells per animal and six readouts per
cell. Every animal moves all its cells by one amount and every cell adds its own noise, so that two cells of the
same animal correlate by 0.03. The first readout is shifted by 0.4 standard deviations of the cells in the case
animals; the five others do not differ between the conditions and are there for the correction for many readouts.

.. code-block:: python

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

Step 1: one readout, computed on all cells
------------------------------------------

Compute the readout once, on all cells of both conditions together, so that both conditions are on one scale: one
pseudotime over all cells, the fate probabilities of one grid built from all cells, one signature score with the
same genes for every cell. Computing the readout for each condition on its own, or scaling it within each
condition, can remove the very difference to be tested. In the example each readout is centred and scaled over all
cells:

.. code-block:: python

    for name in READOUTS:
        cells[name] = (cells[name] - cells[name].mean()) / cells[name].std()

.. code-block:: text

    Step 1. Six readouts per cell, computed on all 3200 cells of both conditions together.

Step 2: the animal means, and each animal left out
--------------------------------------------------

The animal means are the data of the test, so look at them. With four animals per condition a single animal that
lies apart from the others can decide the result; leaving each animal out in turn and repeating the comparison
shows how much rests on one animal. ``compare`` is the test of step 3.

.. code-block:: python

    animals = cells.groupby('animal').agg(condition=('condition', 'first'), mean=('readout', 'mean'))
    without = [compare(cells[cells['animal'] != animal]) for animal in animals.index]
    animals['difference_without'] = [r['estimate'] for r in without]
    animals['p_without'] = [r['p_value'] for r in without]

.. code-block:: text

    Step 2. Animal means of the readout, and the comparison repeated without each animal:
             condition   mean difference_without p_without
    animal
    case1         case +0.146             +0.386    0.0149
    case2         case +0.125             +0.393    0.0152
    case3         case +0.230             +0.358    0.0192
    case4         case +0.244             +0.353    0.0209
    control1   control -0.435             +0.290    0.0025
    control2   control -0.174             +0.377    0.0838
    control3   control -0.062             +0.414    0.0524
    control4   control -0.074             +0.410     0.057

control1 lies furthest from the other animals of its condition. Without any one animal the difference stays
positive, between +0.290 and +0.414, so its direction does not rest on one animal. Whether its p-value falls below
0.05 does: without control2, control3 or control4 it rises to 0.084, 0.052 and 0.057 (with one animal fewer the test
has less to go on, and three animals against four are compared with Welch's t-test). Without control1 it falls,
because the remaining control animals agree more closely.

Step 3: test the animal means
-----------------------------

.. code-block:: python

    def compare(cells, readout='readout'):
        return scRL.compare_readout(cells[readout], cells['animal'], cells['condition'], test='auto', scale='raw')


    res = compare(cells)

``test='auto'`` runs the pooled t-test when both conditions have the same number of animals and their mean numbers
of cells per animal are less than threefold apart, and Welch's t-test otherwise. ``scale='raw'`` (the default)
tests the animals' mean readout itself, which is the quantity reported; every animal counts once, whatever its
number of cells. Here both conditions have four animals of 400 cells each, so the pooled t-test runs:

.. code-block:: text

    Step 3. The pooled t-test of the animal means: t = 4.06 on 6 df, p = 0.00662.

Never compare a group of two animals: the spread between the animals of that condition is then estimated from a
single difference. ``compare_readout`` refuses a condition with one animal; a condition with two animals is still compared, but
the function issues a ``UserWarning`` that a group of two is not recommended. Use at least three animals per
condition, and report the animal values descriptively if no more are available.

Step 4: what to report
----------------------

.. code-block:: python

    within = scRL.design_effect(cells['readout'], cells['animal'], groups=cells['condition'])
    tests = [compare(cells, name) for name in READOUTS]
    readouts = pd.DataFrame({'estimate': [t['estimate'] for t in tests], 'p_value': [t['p_value'] for t in tests]},
                            index=READOUTS)
    readouts['q_value'] = stats.false_discovery_control(readouts['p_value'])

.. code-block:: text

    Step 4. case - control: +0.373 (95% interval +0.148 to +0.597);
      4 and 4 animals, 1600 and 1600 cells.
      Within-animal correlation, within conditions: 0.013 (design effect 6.2).
      The design implies that a test over cells would call about 47% of readouts that do not differ at all (level 0.05).
      The six readouts, animal-level p-values and Benjamini-Hochberg q-values:
            estimate p_value q_value
    readout   +0.373 0.00662  0.0397
    null_1    -0.067   0.725   0.785
    null_2    +0.036   0.785   0.785
    null_3    +0.075   0.492   0.738
    null_4    -0.278   0.204   0.613
    null_5    -0.157   0.409   0.738

Report:

- the difference of the animal means with its 95% interval (``estimate``, ``ci_low`` and ``ci_high`` of
  ``compare_readout``): case minus control, +0.373 (+0.148 to +0.597) standard deviations of the cells. The
  conditions are taken in alphabetical order unless ``groups`` is a categorical, which fixes their order;
- the animals and the cells per condition (``n_samples`` and ``n_cells``): four and four animals, 1,600 and 1,600
  cells;
- the within-animal correlation γ (``icc`` of ``design_effect``, on the ranks of the readout by default),
  estimated within conditions by passing ``groups``, so that the difference between the conditions is not counted
  as resemblance within animals: 0.013. It lies below the simulated 0.03, because from eight animals it can only
  be estimated roughly;
- with many readouts, the Benjamini-Hochberg q-values of the animal-level p-values. The shifted readout keeps
  q = 0.040; the five readouts without an effect have q-values of 0.61 or more.

The design effect and ``implied_false_positive_rate`` of ``design_effect`` say what a test over cells would do with
these data: with this correlation and 400 cells per animal, it would call about 47% of readouts that do
not differ at all, at level 0.05. The rate explains why a test over cells over-calls. It is not a correction to apply to such a
test, and a small rate never permits one. On these data a rank-sum test over the 3,200 cells gives p = 1.1e-25 for
the shifted readout, against 0.0066 for the animal means, and a p-value below 0.05 for four of the five readouts
that do not differ:

.. code-block:: python

    # in a Python session started in the examples folder
    from scipy import stats
    from protocol_example import READOUTS, run_example

    cells = run_example(seed=0)['cells']
    for name in READOUTS:                    # cells as units: shown only to see what the over-calling looks like
        case, control = (cells.loc[cells['condition'] == c, name] for c in ('case', 'control'))
        print(name, stats.mannwhitneyu(case, control).pvalue)

Running it on public multi-donor data
-------------------------------------

The protocol is meant for data in which each condition holds several donors or animals, and it is most useful
when there are many. Public cohorts of this kind exist. One is a cohort of blood cells (peripheral blood
mononuclear cells) from patients with systemic lupus erythematosus and from healthy donors, GEO accession
GSE174188, which is also distributed by CZ CELLxGENE Discover. On such data, compute the readout on all cells of
one cell type at once (comparing within one cell type keeps a difference in cell-type composition from being read
as a difference in the readout), pass the donor of each cell as ``samples`` and its disease status as ``groups``,
and follow steps 2 to 4. Take the donor as the sample, so that each donor counts once even when its cells come from
several libraries or batches. With unequal numbers of donors in the two conditions, ``test='auto'`` runs Welch's
t-test.

These data are not part of scRL, and nothing here downloads them. Before using them, check their terms of use
(licence and conditions of access) where you obtain them.

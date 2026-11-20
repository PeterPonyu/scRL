Comparing conditions on one grid
================================

Two conditions (for example control and knockout cells) are compared on one grid built from all cells. The
lattice and the pseudotime are common to both; each condition's map takes its clusters from its own cells and
is paid by the same reward rule.

.. code-block:: python

    gres = scRL.grids_from_embedding(X3)
    scRL.project_cluster(gres, clusters)
    scRL.align_pseudotime(gres, early_cluster='4')
    scRL.project_conditions(gres, adata.obs['condition'])
    table = scRL.compare_abundance(gres, test='knockout', reference='control')
    scRL.plot_conditions(gres)
    scRL.plot_projections(gres, key='abundance_call')   # the calls seen along each axis
    values = scRL.compare_values(gres, 'knockout', 'control', ends=['3'], starts=['4'], mode='Contribution')

Where the cells are
-------------------

``scRL.project_conditions`` counts the cells of each condition within ``radius`` grid steps of every grid point
and stores the counts as ``gres.grids['condition_counts']``.

``scRL.compare_abundance`` classes every grid point as shared, held by one condition only, or too sparse to
tell, and tests where the numbers of cells differ (a two-sided binomial test across cells, twice the smaller
tail capped at 1, or a Welch t-test across samples
when ``samples`` is given, with Benjamini-Hochberg q-values). A grid point held by one condition only is not
yet a finding: with unequal numbers of cells, sampling alone leaves many sparse grid points with cells of one
condition only. The table ``gres.grids['abundance_clusters']`` therefore gives, per cluster, the number of such
points observed next to the number expected after shuffling the condition labels over the cells (the
"by chance" columns). Only an excess over the by-chance number points to a real difference.

``scRL.plot_conditions`` draws where each condition is, the log2 fold change of the cell numbers, the calls
per cluster and the observed against the shuffled one-condition points. ``scRL.plot_projections(gres,
key='abundance_call')`` is the clearest single summary: along each axis it shows the most frequent call
(depleted, enriched, no difference, too few cells) in the colours of the calls.

What the cells can still reach
------------------------------

``scRL.condition_grid`` builds the map of one condition on the same lattice: the grid points near its cells,
each labelled with the cluster of its nearest cell of that condition, on the pseudotime of the pooled grid.

``scRL.compare_values`` builds a lineage reward on each condition's map from that map's own labels (the grid
points labelled with ``ends`` pay the time weight of their pseudotime, scaled the same way on every map),
solves it exactly, and compares the values where both maps have travel points against the differences
obtained after shuffling the condition labels (the null). It reports per grid point an effect and a z-score,
and per cluster the difference of medians, its size relative to the spread of the reference values, a z-score
and a p-value. With 'Decision' rewards it asks whether a condition's cells still reach the entry of their own
territory of the lineage; with 'Contribution' rewards whether they reach its far end. Where a condition has no
cells of the lineage there is nothing to reach, and its value toward the lineage falls to zero.

``scRL.compare_fates`` compares the share of each condition's cells in each lineage (for example the heaviest
lineage from ``fate_probabilities``, ``key='fate_top'``), with bootstrap intervals and a label-permutation p-value.

Composition and shifts within cell states
-----------------------------------------

A difference in lineage shares between conditions is first of all a difference in which cell states are
present: a condition with more cells in the clusters near one lineage has more cells whose heaviest lineage is
that one.
Read such a difference beside the cluster composition. ``standardise=True`` reweights each condition (or each
sample) to one cluster composition and splits the difference into a composition part and a within-cluster
part, which add up to it. To look for a shift within cell states, compare the uncommitted cells only
(``within=['...']``, a list of clusters or a mask) with the mean of the lineage shares (``stat='mean'`` and
``key='fate'``), which keeps the graded shares instead of counting heaviest lineages:

.. code-block:: python

    scRL.compare_fates(gres, 'knockout', 'control', standardise=True)
    scRL.compare_fates(gres, 'knockout', 'control', key='fate', stat='mean', within=['progenitor'])

Replicate samples
-----------------

Without ``samples`` every test is across cells. It cannot tell a difference between conditions from a
difference between two libraries, so with one library per condition read the calls as a description of these
two libraries. When each condition has several libraries, pass one sample label per cell as ``samples``:

.. code-block:: python

    scRL.shares_by_sample(gres, adata.obs['library'], key='gate_top')      # one row per library
    scRL.compare_fates(gres, 'knockout', 'control', samples=adata.obs['library'])
    scRL.compare_abundance(gres, 'knockout', 'control', samples=adata.obs['library'])

The grid-point test of ``compare_abundance`` is then made across samples. It is valid but has little power
with a handful of libraries: three or four per condition can leave almost no grid point called after
correction. Where each library contributes one number the comparison has power: the share of each library's
cells per lineage (``compare_fates``, which adds means over samples and a Welch t-test across samples) and,
when clusters are projected, per cluster (the ``share_reference``, ``share_test``, ``pvalue_samples`` and
``qvalue_samples`` columns of ``gres.grids['abundance_clusters']``). With replicates, read the grid map as the
picture and these tables as the test. When the libraries form blocks (for example replicate A of each
condition prepared together), give the blocks as ``pairs`` (a block label per cell, or a dict library ->
block) to add a paired test across blocks.

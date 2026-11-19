Comparing conditions on one grid
================================

Two conditions (for example control and knockout cells) are compared on one grid built from all cells, so that
grid points, clusters, pseudotime and rewards are the same for both.

.. code-block:: python

    gres = scRL.grids_from_embedding(X3)
    scRL.project_cluster(gres, clusters)
    scRL.align_pseudotime(gres, early_cluster='4')
    scRL.project_conditions(gres, adata.obs['condition'])
    table = scRL.compare_abundance(gres, test='knockout', reference='control')
    scRL.plot_conditions(gres)
    scRL.d_rewards(gres, starts=['4'], ends=['3'], mode='Contribution')
    values = scRL.compare_values(gres, 'knockout', 'control', reward_mode='Contribution')

Where the cells are
-------------------

``scRL.project_conditions`` counts the cells of each condition within ``radius`` grid steps of every grid point
and stores the counts as ``gres.grids['condition_counts']``.

``scRL.compare_abundance`` classes every grid point as shared, held by one condition only, or too sparse to
tell, and tests where the numbers of cells differ (binomial test across cells, or Welch t-test across samples
when ``samples`` is given, with Benjamini-Hochberg q-values). A grid point held by one condition only is not
yet a finding: with unequal numbers of cells, sampling alone leaves many sparse grid points with cells of one
condition only. The table ``gres.grids['abundance_clusters']`` therefore gives, per cluster, the number of such
points observed next to the number expected after shuffling the condition labels over the cells (the
"by chance" columns). Only an excess over the by-chance number points to a real difference.

``scRL.plot_conditions`` draws where each condition is, the log2 fold change of the cell numbers, the calls
per cluster and the observed against the shuffled one-condition points.

What the cells can still reach
------------------------------

``scRL.condition_grid`` builds the map of one condition on the same lattice: the grid points near its cells,
with the cluster labels, pseudotime and rewards of the pooled grid, so that every grid point pays the same
reward on every map.

``scRL.compare_values`` solves the same reward exactly on each condition's map and compares the values where
both maps have travel points, against the differences obtained after shuffling the condition labels (the
null). It reports per grid point an effect and a z-score, and per cluster the difference of medians, its size
relative to the spread of the reference values, a z-score and a p-value. With 'Decision' rewards it asks
whether a condition still reaches the entry of a lineage; with 'Contribution' rewards whether it still reaches
its far end. A condition that lost a branch but keeps a few grid points at its base shows the loss in
'Contribution' mode.

``scRL.compare_fates`` compares the share of each condition's cells in each lineage (for example the heaviest
lineage from ``fate_gate``), with bootstrap intervals and a label-permutation p-value.

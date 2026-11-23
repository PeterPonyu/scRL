# Changelog

## 2.0.0 (2026-11-23)

This release is scRL 2, so the version moves from 0.0.9 to 2.0.0. It holds everything added since 0.0.9: a protocol for comparing a per-cell readout between conditions with the sample as the unit, grids for 3-D to 5-D embeddings, exact values and fate maps without training, and comparisons between conditions on one grid.

### Comparing a per-cell readout between conditions

- `scRL.compare_readout(values, samples, groups)` compares a per-cell readout (a pseudotime, a fate probability, a signature score) between two conditions with the sample (animal, donor) as the unit. By default (`test='auto'`) it runs a t-test on the samples' mean readout: the pooled t-test when both conditions have the same number of samples and their mean numbers of cells per sample are less than threefold apart, Welch's t-test otherwise.
- It returns the difference of the sample means (`estimate`, first condition minus second) and, for the t-tests, its 95% confidence interval (`ci_low`, `ci_high`; the level is 1 - `alpha`), with the p-value, the degrees of freedom and the numbers of samples and cells per condition.
- It refuses a condition with one sample and warns (`UserWarning`) when a condition has only two samples.
- A relabelling test, a linear mixed model (it needs statsmodels, which is imported only for this test) and a test over cells (shown only for comparison; it always warns) can be asked for by name. `scale='rank'` or `scale='normal'` tests the samples' mean ranks or mean normal scores instead of their mean readout.
- `scRL.design_effect(values, samples, groups=None)` estimates how strongly cells resemble each other within samples (the within-sample correlation), the design effect, and the false-positive rate that this correlation implies for a test taking cells as units (`implied_false_positive_rate`, with a t reference for a correlation estimated from few samples, and `implied_false_positive_rate_normal`). With `groups` the correlation is estimated within conditions.
- A worked example, `examples/protocol_example.py`, runs the protocol on simulated animals, and the documentation page "Protocol: comparing a per-cell readout between conditions" walks through it.

### Grids in 3-D to 5-D, exact values and fate maps

- `grids_from_embedding` accepts embeddings with 2 to 5 dimensions (new arguments `method`, `radius` and `moves`; 2-D grids are built as before). Projection, pseudotime, rewards, training, trajectories and the simulator run on these grids. 4-D and 5-D grids are experimental.
- `trainer` has a fourth algorithm, `'SAC'` (a soft actor-critic for discrete moves), and a method `state_values()`. `eval_fate` and `get_state_value` work for every algorithm.
- `exact_values` computes, without training, the best return from every grid point for a relaxed problem in which returning to a grid point does not end the walk; `exact_trajectories` gives trajectories that follow these values.
- `fate_gate` ranks the lineages at every grid point from one value map per lineage; `fate_probabilities` gives the share of walks from every grid point that reach each lineage first, and `fate_over_runs` averages these shares over several embedding runs of the same cells. `fate_corridors`, `fate_decisions`, `corridor_profile` and `plot_corridors` describe the corridors from a start region to each lineage and where on the grid each lineage's fate is decided.
- `place_cells` and `project_cells` read the values of a stored grid for cells that were not used to build it.
- `project_conditions`, `compare_abundance`, `condition_grid`, `compare_values`, `compare_fates` and `shares_by_sample` compare two conditions on one grid, with the samples as units when `samples=` is given.
- `reachability` checks which rewarded grid points an episode can reach.
- New views: `plot_grids`, `plot_slices`, `plot_views`, `plot_trend`, `plot_clusters`, `plot_policy`, `plot_training`, `plot_projections`, `plot_report`, `save_rotation`, `plot_style`, `plot_fates`, `plot_conditions`, `plot_mesh`, `plot_trajectories` and `plot_voxels`. Interactive 3-D figures need plotly, which is not installed with scRL.
- `align_pseudotime` takes `seed` to make its start points reproducible.
- New examples `examples/grid3d_paul15.py` and `examples/grid3d_walkthrough.py` (with expected results in `examples/expected/` and a `--check` option), and new documentation pages on 3-D grids, the exact solution, comparing conditions and the test data.

### Fixes

- The tabular environment (`'TabularQ'`) assumed a grid width of 50. It now uses the grid's own width, so its results change for grids built with another `n`.
- `eval_fate` and `get_state_value` failed for `'DDQN'` and for double-precision latent spaces.
- Without start probabilities on the grid, the trainer raises a `ValueError` that says how to provide them, instead of a `KeyError`.
- The reward functions accept a single cluster name as a string and cluster names that are not strings, and treat an empty start name as no start.
- `align_pseudotime` samples its start points from all grid points of the early cluster when none of them lies on the grid boundary, instead of failing, and names the clusters on the grid when the early cluster has none.

### Requirements

- scipy 1.11 or later is now a listed requirement: `compare_readout` reads t-test results that scipy provides from version 1.11.

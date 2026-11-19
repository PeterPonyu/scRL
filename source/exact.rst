Exact solution and lineage gate
===============================

The grid environment is small and deterministic, so the best return from every grid point can be computed
exactly instead of being learned. One exact solution per lineage, combined by a gate, gives one fate map.

.. code-block:: python

    gres = scRL.grids_from_embedding(X3)
    scRL.project_cluster(gres, clusters)
    scRL.align_pseudotime(gres, early_cluster='4')
    for end in ['0', '3']:
        scRL.d_rewards(gres, starts=['4'], ends=[end], mode='Decision')
        scRL.exact_values(gres)                       # stored as gres.grids['exact_<end>']
    scRL.fate_gate(gres, ['exact_0', 'exact_3'], names=['lineage 0', 'lineage 3'])
    scRL.plot_fates(gres)
    scRL.plot_slices(gres, key='exact_3', policy=gres.grids['exact_3_move'])

Exact solution
--------------

``scRL.exact_values`` computes, for every kept grid point, the best discounted return under the rules of the
learners (a move off the grid costs -2 and ends the episode, a step onto the boundary outside the start
cluster ends it, at most ``max_step + 1`` moves) by backward induction over the moves. It has no random seed and
takes well under a second. It stores the values as ``gres.grids['exact_<reward key>']`` and the best move from
every grid point as ``gres.grids['exact_<reward key>_move']``, which ``plot_policy`` and
``plot_slices(policy=...)`` can draw.

One rule of the learners is relaxed: returning to a grid point does not end the episode. Inside a rewarded
region the best path may therefore go back and forth between neighbouring rewarded points, and the value there
approaches reward / (1 - gamma). Read the value of a grid point as the discounted distance to the best-paying
part of the lineage: its entry with 'Decision' rewards, its far end with 'Contribution' rewards. It is not the
amount of rewarded territory.

Lineage gate
------------

``scRL.fate_gate`` combines one value map per lineage (exact values or the values of trained learners) into
lineage weights. Each map is rank-scaled to 0..1 and the weights are a softmax across lineages with temperature
``tau``: a grid point gets most of its weight from the lineage in whose map it ranks highest. The weights, the
heaviest lineage (``gate_top``) and how undecided a point is (``gate_entropy``, 0 decided, 1 undecided) are
stored for the grid points and, brought back with the Gaussian kernel of ``project_back``, for the cells.

``scRL.plot_fates`` draws the heaviest lineage, how undecided each point is, and the mean lineage weights per
cluster.

Reachability
------------

``scRL.reachability`` checks before training whether any rewarded grid point can be reached from the start
points within the step limit under the stop rules. The environments warn when none can; a larger ``max_step``,
a larger grid ``radius`` or other start points help.

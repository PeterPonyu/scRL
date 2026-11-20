Exact solution, lineage gate and lineage shares
===============================================

The grid environment is small and deterministic, so the best return from every grid point can be computed
exactly instead of being learned. One exact solution per lineage, combined by a gate, gives one fate map.

.. code-block:: python

    gres = scRL.grids_from_embedding(X3)
    scRL.project_cluster(gres, clusters)
    scRL.align_pseudotime(gres, early_cluster='4', seed=0)
    for end in ['0', '3']:
        scRL.d_rewards(gres, starts=['4'], ends=[end], mode='Decision')
        scRL.exact_values(gres)                       # stored as gres.grids['exact_<end>']
    scRL.fate_gate(gres, ['exact_0', 'exact_3'], names=['lineage 0', 'lineage 3'])
    scRL.plot_fates(gres)
    scRL.plot_slices(gres, key='exact_3', policy=gres.grids['exact_3_best'])
    scRL.fate_probabilities(gres, {'lineage 0': '0', 'lineage 3': '3'})
    scRL.plot_fates(gres, key_add='fate')

Exact solution
--------------

``scRL.exact_values`` computes, for every kept grid point, the best discounted return under the rules of the
learners (a move off the grid costs -2 and ends the episode, a step onto the boundary outside the start
cluster ends it, at most ``max_step + 1`` moves) by backward induction over the moves. The solver has no
random element and takes well under a second. It stores the values as ``gres.grids['exact_<reward key>']``
and the best move from every grid point as ``gres.grids['exact_<reward key>_move']``. On a regular lattice
several moves are often equally good; ``gres.grids['exact_<reward key>_best']`` marks all of them, and
``plot_policy`` and ``plot_slices(policy=...)`` draw their mean, which is the reading to use.

Its input does have one: ``align_pseudotime`` samples its start points from the early cluster, so
pseudotime, and with it the rewards, depends on the draw. Give it ``seed`` (for example ``seed=0``) to make
pseudotime reproducible without touching numpy's global generator. Once pseudotime is fixed, the exact
solution, the lineage gate and the lineage shares are deterministic.

One rule of the deep environment is relaxed: returning to a grid point does not end the episode (in the
deep environment a revisit pays -1 and ends it). The exact solution is therefore that of a relaxed problem.
When the rewards are not negative, its values are an upper bound on what a policy can collect under the deep
environment's rules, and a reference for the shape of the value landscape, not a return that a policy
reaches. Lineage rewards ('Decision' and 'Contribution') are never negative, and neither are gene rewards
without ``punish_keys``; with ``punish_keys`` gene rewards can be negative and the bound does not hold. The
tabular environment follows other rules (it also pays the pseudotime change on every move and handles a
revisit differently), so the bound is not for it. Inside a rewarded region the best relaxed path may go back
and forth between neighbouring rewarded points, and the value there approaches reward / (1 - gamma). A
greedy policy read from the exact solution and run under the deep environment's rules reaches the lineage
and then ends by revisiting a point, so its return is lower than the exact value. Read the value of a grid
point as the discounted distance to the best-paying part of the lineage: its entry with 'Decision' rewards,
its far end with 'Contribution' rewards. It is not the amount of rewarded territory.

Lineage gate
------------

``scRL.fate_gate`` combines one value map per lineage (exact values or the values of trained learners) into
lineage weights. Each map is rank-scaled to 0..1 and the weights are a softmax across lineages with temperature
``tau``: a grid point gets most of its weight from the lineage in whose map it ranks highest. The weights, the
heaviest lineage (``gate_top``) and how undecided a point is (``gate_entropy``, 0 decided, 1 undecided) are
stored for the grid points and, brought back with the Gaussian kernel of ``project_back``, for the cells.

The gate is a ranking: it orders grid points within a lineage and marks the heaviest lineage of each point.
Its weights are close to one-hot and are not shares of cells; for the share of walks reaching each lineage use
``fate_probabilities``.

``scRL.plot_fates`` draws the heaviest lineage, how undecided each point is, and the mean lineage weights per
cluster.

Lineage shares
--------------

``scRL.fate_probabilities`` gives, for every grid point, the share of walks starting there that enter the end
points of each lineage first within ``steps`` moves. ``forward`` biases the walk toward later pseudotime (0
is the unbiased walk). The default is a mild bias, ``forward=0.25``, with 5,000 moves. At ``forward`` of 1
and above each move is strongly drawn to later pseudotime: at 1, a move one median pseudotime step later
weighs about seven times (e squared) a move the same step earlier, so walks turn back toward a neighbouring
branch less often. A smaller value lets them do so more often, which can keep fate signals of neighbouring
lineages at early grid points, and needs more moves to arrive, hence the larger budget. When many grid points
are still collecting arriving walks in the last tenth of the moves, a warning asks for more ``steps`` (not
for fewer than 20 moves, a deliberately short horizon). The shares are not
renormalised: they sum to the share of walks that arrive at any listed lineage (``fate_reached``), so a region
whose walks do not arrive, for example a tip that is not among the listed lineages, shows up as 'none'. At
early grid points the split among lineages depends on ``forward`` and on which clusters are given as the ends
of each lineage, so state both when reporting it. The results are stored like those of the gate, so
``plot_fates(gres, key_add='fate')`` (with a grey 'none' segment for walks that do not arrive) and
``compare_fates(..., key='fate_top')`` read them. For value maps of trained learners use the gate
(``fate_gate``).

Reachability
------------

``scRL.reachability`` checks before training whether any rewarded grid point can be reached from the start
points within the step limit under the stop rules. The environments warn when none can; a larger ``max_step``,
a larger grid ``radius`` or other start points help.

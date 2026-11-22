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
random element and takes well under a second. It stores the values as ``gres.grids['exact_<reward key>']``,
the best move from every grid point as ``gres.grids['exact_<reward key>_move']`` and its settings as
``gres.grids['exact_<reward key>_settings']``. On a regular lattice
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

Trajectories from the exact solution
------------------------------------

``scRL.exact_trajectories`` gives trajectories for a lineage without training. From each start point (by
default every grid point of the start cluster) the walk takes, at every move, the move of largest exact move
value among the moves into grid points it has not visited yet; equally good moves go to the first one, as in
the best-move table. It follows the rules the values were solved with (stop points, ``gamma``, at most
``max_step + 1`` moves; the settings are read from what ``exact_values`` stored). When every neighbour has
been visited the walk ends where it is ('trapped') instead of returning. Each start point gives one walk, and
no random seed is involved.

.. code-block:: python

    scRL.d_rewards(gres, starts=['4'], ends=['3'])
    scRL.exact_values(gres)                           # gres.grids['exact_3']
    walks = scRL.exact_trajectories(gres)             # one row per walk
    gres.trajectory['exact_3_walks_summary']          # share entering the lineage, moves, share of the bound
    scRL.exact_trajectories(gres, until_entry=True, key_add='to_3')
    scRL.traj_results(gres, ['4', '3'], 'to_3')       # 2-D or 3-D view of the mean path
    scRL.plot_grids(gres, key='to_3_visits')          # how many walks pass each grid point

The walk is the greedy reading of the relaxed solution with returns ruled out, not the optimum of the
training problem. It is the reading to use for a lineage whose end cluster lies far from the start cluster:
trained learners often fail to reach such a lineage, while these walks reach it whenever a path without returns
leads there. With rewards that are never negative, no walk from the start cluster (or from any grid point that is
not a stop point) collects more than the exact value at its start point, and the share of that bound is reported
per walk; with gene rewards that include ``punish_keys``, or with stop points given as starts, a warning says that
the bound does not hold. Each row gives the grid points visited, how the walk ended (stop
point, step cap, trapped, left the grid, or, with ``until_entry``, entered), whether and at which move it
first entered the lineage's end cluster by a move (for a gene reward: a grid point whose entry pays a positive
reward), and the discounted return. The paths and their mean path are stored where ``get_traj_df`` stores
the rollouts of a trained learner, so ``traj_results`` draws them in 2-D and 3-D. The simulator figures need
a trained learner and do not take these walks.

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

Shares over embedding runs
--------------------------

A UMAP or t-SNE run is one random layout of the cells, and the grid over it is one of many. The walk shares of
uncommitted cells can change between runs of the same cells, most for a lineage whose end clusters sit on a
separate part of the grid, where no walk from the start arrives. ``scRL.fate_over_runs`` takes several grids of
the same cells (for example one per UMAP seed, each after ``project_cluster`` and ``align_pseudotime``), computes
the walk shares on each and averages every cell's shares over the runs::

    runs = []
    for seed in range(10):
        Y = umap_3d(cells, seed)                     # any embedding of the same cells, in the same order
        g = scRL.grids_from_embedding(Y)
        scRL.project_cluster(g, clusters)
        scRL.align_pseudotime(g, 'Stem', seed=0)
        runs.append(g)
    out = scRL.fate_over_runs(runs, {'Goblet': 'Goblet', 'Tuft': 'Tuft'}, start='Stem')
    out['halves']        # agreement of the first and second half of the runs, uncommitted cells
    out['reachable']     # per run: share of each lineage's end points on the start's part of the grid

The means are written for the cells of the first run (``fate_runs_mean_<lineage>``, ``_top``, ``_entropy``), so
``compare_fates(runs[0], test, reference, key='fate_runs_mean', stat='mean', samples=...)`` compares conditions on
them. When the halves agree below about 0.8 for a lineage, add runs; when a lineage's end points are not on the
start's part of the grid, its share from the start rests on a few stray points and should not be read.

Fate corridors from a start region
----------------------------------

A multipotent state is a region of the grid, not a point, and a differentiation path is a corridor that holds many
grid points, and cells, at each stage. ``scRL.fate_corridors`` starts the walk of ``fate_probabilities`` from a start
region (one or more clusters, or any mask over the kept grid points; each point weighted by the cells nearest to it)
and computes, exactly and without sampling, for every lineage the expected number of visits to each grid point by the
walks that end in that lineage (its corridor), the fate potential of every grid point (the probability of each
lineage, the limit of the walk shares) and its potency (the effective number of fates still open, from 1 to the
number of lineages). In reinforcement-learning terms the fate potential is the value of the walk policy for a reward
of 1 on reaching the lineage, and the corridor is that policy's state occupancy over the episodes that reach it.

.. code-block:: python

    scRL.fate_probabilities(gres, ends)                       # walk shares, as before
    summary = scRL.fate_corridors(gres, ends, start='HSC_1')  # probability of each lineage from the start region
    profile = scRL.corridor_profile(gres)                     # stage by stage along pseudotime
    profile['pairs']        # how much of two corridors is shared, and where they part
    profile['decisions']    # where each corridor crosses an even chance of its lineage
    profile['tree']         # the order of the fate decisions (average linkage on 1 - overlap)
    scRL.plot_corridors(gres)                                 # corridors in 2-D or 3-D, start region as rings

Corridors of several lineages run together where they leave the start region and part where the fate is decided. A
mask such as ``gres.grids['corridor_potency'] >= 2`` defines the multipotent region from the data instead of a
cluster.

``scRL.fate_decisions`` adds where each lineage's fate is decided. The decision intensity of lineage l at a grid point
is the variance of the one-step change of its fate potential, which is zero at the end points and wherever only one
outcome for l remains open; summed along the walk it gives back the uncertainty, h (1 - h) = (I - Q)^-1 d, and its
map from the start region (intensity times expected visits) sums to the variance of the outcome from the start
region. Any stored value can colour the grid view.

.. code-block:: python

    scRL.fate_decisions(gres, ends, start='HSC_1')            # same walk and start region as fate_corridors
    scRL.plot_grids(gres, key='decision_<lineage>')           # the decision map of one lineage on the grid

Reachability
------------

``scRL.reachability`` checks before training whether any rewarded grid point can be reached from the start
points within the step limit under the stop rules. The environments warn when none can; a larger ``max_step``,
a larger grid ``radius`` or other start points help.

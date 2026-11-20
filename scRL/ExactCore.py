"""Exact solution of a reward on a grid by backward induction over the move table."""
import numpy as np

from .EnvironmentNDCore import _stop_mask
from .GridNDCore import is_nd, lattice_from_scan


def exact_values(gres, reward_type='d', reward_mode='Decision', gamma=.9, max_step=50, key_add=None):
    """
    Best discounted return from every kept grid point, computed exactly by backward induction.

    The rules are those of the deep environment: a move off the grid pays -2 and ends the episode; a move onto
    a stop point (a boundary point outside the start points) pays the reward of the move plus the pseudotime
    increment and ends the episode; any other move pays its reward and continues, discounted by ``gamma``. At
    most ``max_step + 1`` moves are made. One rule is relaxed: returning to a grid point already visited does
    not end the episode (with that rule the best return would depend on the whole path).

    The result is the exact solution of this relaxed problem (in the deep environment a revisit pays -1 and
    ends the episode). When the rewards are not negative, its values are an upper bound on what a policy can
    collect under the deep environment's rules, and a reference for the shape of the value landscape. Lineage
    rewards ('Decision', 'Contribution') are never negative, nor are gene rewards without ``punish_keys``; with
    ``punish_keys`` they can be, and the bound does not hold. The tabular environment pays the pseudotime
    change on every move and handles a revisit differently, so the bound is not for it. Inside a rewarded
    region the best relaxed path may go back and forth between neighbouring rewarded points, so the value there
    approaches reward / (1 - gamma). A greedy policy read from these values and run under the deep
    environment's rules reaches the lineage and then ends by revisiting a point, so its return is lower than
    the value. The value of a grid point is therefore the discounted distance to the best-paying
    part of the lineage (its entry with 'Decision' rewards, its far end with 'Contribution' rewards), not the
    amount of rewarded territory.

    Parameters
    ----------
    gres
        Grids results after ``d_rewards`` or ``c_rewards``.
    reward_type
        'd' or 'c'.
        (Default: 'd')
    reward_mode
        'Decision' or 'Contribution'.
        (Default: 'Decision')
    gamma
        Discount factor.
        (Default: .9)
    max_step
        Maximum steps of an episode.
        (Default: 50)
    key_add
        Key under which the values are stored.
        (Default: None, 'exact_<reward key>')

    Returns
    ----------
    The values, one per kept grid point in stored order. They are stored as ``gres.grids[key]``; the index of
    the first best move from each kept grid point as ``gres.grids[key + '_move']``, and every best move (a
    boolean array, kept grid points x moves, True for each move within 1e-9 of the best; on a regular lattice
    several moves are often equally good) as ``gres.grids[key + '_best']``. Pass ``_best`` to the policy views
    to draw the mean of the equally good moves.
    """
    table = f'{reward_type}_{reward_mode}_rewards'
    if table not in gres.qlearning:
        raise ValueError(f'No {table!r} reward table: call d_rewards (reward_type="d") or c_rewards '
                         f'(reward_type="c") with mode={reward_mode!r} first.')
    lattice = gres if is_nd(gres) else lattice_from_scan(gres)
    nb = lattice.grids['neighbors']
    R = gres.qlearning[table].to_numpy(dtype=float)
    stop = _stop_mask(lattice, lattice.grids.get('starts_cluster_grids'))
    t = gres.grids['pseudotime'].loc[np.asarray(gres.grids['mapped_grids'])].to_numpy(dtype=float)
    valid = nb >= 0
    q = np.where(valid, nb, 0)
    ends_here = valid & stop[q]
    r_stop = R + (t[q] - t[:, None])
    V = np.zeros(len(nb))
    for _ in range(max_step + 1):
        Q = np.where(valid, np.where(ends_here, r_stop, R + gamma * V[q]), -2.0)
        V = Q.max(1)
    key = key_add or 'exact_' + gres.qlearning['reward_key']
    gres.grids[key] = V
    gres.grids[key + '_move'] = Q.argmax(1)
    gres.grids[key + '_best'] = Q >= V[:, None] - 1e-9
    return V

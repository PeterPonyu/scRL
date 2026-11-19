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

    Because of that relaxation, inside a rewarded region the best path may go back and forth between
    neighbouring rewarded points, so the value there approaches reward / (1 - gamma). The value of a grid point
    is therefore the discounted distance to the best-paying part of the lineage (its entry with 'Decision'
    rewards, its far end with 'Contribution' rewards), not the amount of rewarded territory.

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
    The values, one per kept grid point in stored order. They are stored as ``gres.grids[key]``, and the index
    of the best move from each kept grid point as ``gres.grids[key + '_move']``.
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
    return V

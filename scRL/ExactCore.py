"""Exact solution of a reward on a grid by backward induction over the move table, and walks read from it."""
import warnings

import numpy as np
import pandas as pd

from .EnvironmentNDCore import _stop_mask
from .GridNDCore import is_nd, lattice_from_scan

TIE = 1e-9                                  # moves within this of the best count as equally good
WALK_ENDS = ('stop point', 'step cap', 'trapped', 'left the grid', 'entered')


def _problem(gres, reward_type, reward_mode):
    """Neighbour table, reward table, stop points and pseudotime of the deep environment's rules."""
    table = f'{reward_type}_{reward_mode}_rewards'
    if table not in gres.qlearning:
        raise ValueError(f'No {table!r} reward table: call d_rewards (reward_type="d") or c_rewards '
                         f'(reward_type="c") with mode={reward_mode!r} first.')
    lattice = gres if is_nd(gres) else lattice_from_scan(gres)
    nb = lattice.grids['neighbors']
    R = gres.qlearning[table].to_numpy(dtype=float)
    stop = _stop_mask(lattice, lattice.grids.get('starts_cluster_grids'))
    t = gres.grids['pseudotime'].loc[np.asarray(gres.grids['mapped_grids'])].to_numpy(dtype=float)
    return nb, R, stop, t


def _solve(nb, R, stop, t, gamma, max_step):
    """Backward induction: final move values Q (kept grid points x moves) and values V."""
    valid = nb >= 0
    q = np.where(valid, nb, 0)
    ends_here = valid & stop[q]
    r_stop = R + (t[q] - t[:, None])
    V = np.zeros(len(nb))
    for _ in range(max_step + 1):
        Q = np.where(valid, np.where(ends_here, r_stop, R + gamma * V[q]), -2.0)
        V = Q.max(1)
    return Q, V


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
    to draw the mean of the equally good moves. The settings of the solution (reward type and mode, ``gamma``,
    ``max_step``, reward key) are stored as ``gres.grids[key + '_settings']``; ``exact_trajectories`` reads
    them.
    """
    nb, R, stop, t = _problem(gres, reward_type, reward_mode)
    Q, V = _solve(nb, R, stop, t, gamma, max_step)
    key = key_add or 'exact_' + gres.qlearning['reward_key']
    gres.grids[key] = V
    gres.grids[key + '_move'] = Q.argmax(1)
    gres.grids[key + '_best'] = Q >= V[:, None] - TIE
    gres.grids[key + '_settings'] = dict(reward_type=reward_type, reward_mode=reward_mode, gamma=gamma,
                                         max_step=max_step, reward_key=gres.qlearning['reward_key'])
    return V


def _walk(nb, R, stop, t, Q, target, starts, gamma, max_step, until_entry):
    """
    Walks from the stored positions ``starts``: at every move the allowed move of largest exact move value, the
    allowed moves being those into kept grid points not visited yet in the walk (ties: the first move within
    ``TIE`` of the best). Returns the visited positions per walk (-1 padded), moves made, end code (index into
    ``WALK_ENDS``), move of first entry into ``target`` (-1 if none) and discounted return.
    """
    n = len(starts)
    path = np.full((n, max_step + 2), -1, dtype=np.int64)
    path[:, 0] = starts
    pos = np.asarray(starts, dtype=np.int64).copy()
    G, disc = np.zeros(n), np.ones(n)
    moves = np.zeros(n, dtype=np.int64)
    entry = np.full(n, -1, dtype=np.int64)
    end = np.full(n, WALK_ENDS.index('step cap'), dtype=np.int64)
    alive = np.arange(n)
    for k in range(max_step + 1):
        if not alive.size:
            break
        p = pos[alive]
        nxt = nb[p]
        seen = (nxt[:, :, None] == path[alive, None, :k + 1]).any(2)
        allowed = (nxt >= 0) & ~seen
        stuck = ~allowed.any(1)
        kept = (nxt >= 0).any(1)
        i = alive[stuck & ~kept]                                   # no kept neighbour: the only move leaves the grid
        G[i] -= 2.0 * disc[i]
        moves[i] = k + 1
        end[i] = WALK_ENDS.index('left the grid')
        end[alive[stuck & kept]] = WALK_ENDS.index('trapped')     # every kept neighbour visited: end without a move
        free = ~stuck
        i, p, nxt, allowed = alive[free], p[free], nxt[free], allowed[free]
        score = np.where(allowed, Q[p], -np.inf)
        a = np.argmax(score >= score.max(1, keepdims=True) - TIE, axis=1)
        q = nxt[np.arange(len(i)), a]
        r = R[p, a]
        moves[i] = k + 1
        path[i, k + 1] = q
        new = target[q] & (entry[i] < 0)
        entry[i[new]] = k + 1
        st = stop[q]
        j = i[st]
        G[j] += disc[j] * (r[st] + t[q[st]] - t[p[st]])
        end[j] = WALK_ENDS.index('stop point')
        go = ~st
        i, q, r, new = i[go], q[go], r[go], new[go]
        G[i] += disc[i] * r
        pos[i] = q
        disc[i] *= gamma
        if until_entry:
            end[i[new]] = WALK_ENDS.index('entered')
            i = i[~new]
        alive = i
    return path, moves, end, entry, G


def _start_positions(gres, mapped, starts, seed):
    """Stored positions of the start points: the start cluster, a number drawn with ``seed``, or given points."""
    cluster = gres.grids.get('starts_cluster_grids') or []
    on_cluster = np.flatnonzero(np.isin(mapped, list(cluster)))
    if starts is None:
        if not len(on_cluster):
            raise ValueError('The reward has no start cluster: pass start grid points with `starts`, or a number of '
                             'start points to draw.')
        return on_cluster
    if isinstance(starts, (bool, np.bool_)) or (np.isscalar(starts) and not isinstance(starts, (int, np.integer))):
        raise TypeError('`starts` must be None, a whole number of start points to draw, or grid point indices.')
    if np.isscalar(starts):
        k = int(starts)
        if k < 1:
            raise ValueError('`starts` as a number must be at least 1.')
        rng = np.random.default_rng(seed)
        probs = gres.grids.get('starts_probs')
        if len(on_cluster):
            pool, p = on_cluster, None
        elif probs is not None:
            pool = np.flatnonzero(np.asarray(probs) > 0)
            p = np.asarray(probs, dtype=float)[pool]
            p = p / p.sum()
        else:
            pool, p = np.arange(len(mapped)), None
        if k > len(pool):
            raise ValueError(f'{k} start points asked for, but only {len(pool)} grid points can start a walk.')
        return np.sort(rng.choice(pool, k, replace=False, p=p))
    given = np.asarray(starts).ravel()
    position = pd.Index(mapped).get_indexer(given)
    if (position < 0).any():
        raise ValueError(f'Start points that are not kept grid points: {given[position < 0][:10].tolist()} '
                         f'(give indices as in gres.grids["mapped_grids"]).')
    return position


def exact_trajectories(gres, key=None, starts=None, seed=None, until_entry=False, key_add=None):
    """
    Trajectories that follow the exact solution of a reward, one per start point, without training.

    From each start point the walk takes, at every move, the move of largest exact move value (the move values
    of the backward induction of ``exact_values``) among the moves into kept grid points it has not visited
    yet; equally good moves (within 1e-9) go to the first one, the rule of the best-move table. It follows the
    rules the values were solved with: a move onto a stop point (a boundary point outside the start points)
    pays its reward plus the pseudotime increment and ends the walk; any other move pays its reward, discounted
    by ``gamma``; at most ``max_step + 1`` moves are made. When every kept neighbour has been visited, the walk
    ends where it is ('trapped') instead of returning to a visited point; a move off the grid (-2) is made only
    from a grid point without kept neighbours. Each start point gives one walk; the result depends on no
    random seed (``seed`` is used only to draw start points when ``starts`` is a number).

    The walk is the greedy reading of the relaxed solution with returns ruled out, not the optimum of the
    training problem (in which a return to a visited point ends the episode). The greedy policy of the
    best-move table itself goes back and forth between neighbouring rewarded points and, under the training
    rules, ends by a return. Prefer these walks to a trained learner's rollouts for a lineage whose end cluster
    lies far from the start cluster: trained learners often fail to reach such a lineage, these walks reach it
    whenever a path without returns leads there. When the rewards are never negative, no walk that starts on a
    grid point other than a stop point (the start cluster's points never are) collects more than the exact
    value at its start point, which bounds every return under the training rules; the share of that bound is
    reported per walk. Rewards from ``c_rewards`` with ``punish_keys`` can be negative, and walks given a stop
    point as start can end with a negative pseudotime increment that the bound does not cover; the walks are
    made the same way, but a warning says that the exact value is not a bound and ``bound_holds`` is False in
    the summary.

    Parameters
    ----------
    gres
        Grids results after ``exact_values``, with the reward table it was solved for.
    key
        Key under which ``exact_values`` stored the values.
        (Default: None, ``'exact_<reward key>'`` of the current reward)
    starts
        Start points: None for every grid point of the start cluster; a number of start points drawn without
        replacement with ``seed`` (from the start cluster; without one, by the start probabilities of
        ``c_rewards``, otherwise from all kept grid points); or grid point indices as in
        ``gres.grids['mapped_grids']``.
        (Default: None)
    seed
        Random seed for drawing start points.
        (Default: None)
    until_entry
        End a walk when it first enters the target ('entered'), to draw the path up to the lineage.
        (Default: False)
    key_add
        Prefix under which the walks are stored.
        (Default: None, ``'<key>_walks'``)

    Returns
    ----------
    DataFrame, one row per walk: ``start`` (grid point), ``path`` (the grid points visited, start included),
    ``moves`` (moves made), ``end`` ('stop point', 'step cap', 'trapped', 'left the grid' or, with
    ``until_entry``, 'entered'), ``entered`` (whether a move entered the target: the end cluster of a lineage
    reward, or a grid point whose entry pays a positive reward for a gene reward; a start point inside the
    target does not count by itself), ``entry_move`` (move of the first entry), ``return`` (discounted),
    ``bound`` (exact value at the start point) and ``bound_share``
    (return / bound where the bound is positive). Stored: ``gres.trajectory['<key_add>_idx']`` (the paths) and
    ``'<key_add>_traj'`` (their mean path by pseudotime, as ``get_traj_df`` makes it), so ``traj_results(gres,
    clusters, '<key_add>')`` draws them in 2-D and 3-D; ``gres.trajectory['<key_add>_summary']`` (share of walks
    that enter the target, median moves, median moves to the entry, median share of the bound, counts of the
    ends, ``bound_holds``, the settings); and ``gres.grids['<key_add>_visits']``, the number of walks
    through each kept grid point.
    """
    from .Trajectory.Core import _mean_path_points
    key = key or 'exact_' + gres.qlearning['reward_key']
    if key not in gres.grids:
        raise KeyError(f'No exact values under {key!r}: call exact_values first (with key_add={key!r} or the '
                       f'default key).')
    settings = gres.grids.get(key + '_settings')
    if not isinstance(settings, dict):
        raise ValueError(f'No settings are stored for {key!r}: solve it again with exact_values.')
    if settings['reward_key'] != gres.qlearning.get('reward_key'):
        raise ValueError(f'{key!r} was solved for the reward {settings["reward_key"]!r}, but the current reward is '
                         f'{gres.qlearning.get("reward_key")!r}: call the reward function for '
                         f'{settings["reward_key"]!r} and exact_values again.')
    nb, R, stop, t = _problem(gres, settings['reward_type'], settings['reward_mode'])
    Q, V = _solve(nb, R, stop, t, settings['gamma'], settings['max_step'])
    stored = np.asarray(gres.grids[key], dtype=float)
    if stored.shape != V.shape or not np.allclose(stored, V, rtol=0, atol=1e-9):
        raise ValueError(f'The values under {key!r} no longer match the grid and reward table they were solved for '
                         f'(the reward was changed after exact_values): call exact_values again.')
    mapped = np.asarray(gres.grids['mapped_grids'])
    valid = nb >= 0
    if settings['reward_type'] == 'd':
        target = np.isin(mapped, list(gres.grids['ends_cluster_grids']))
    else:
        target = np.zeros(len(mapped), bool)
        target[nb[valid & (R > 0)]] = True
    S = _start_positions(gres, mapped, starts, seed)
    negative, on_stop = bool((R[valid] < 0).any()), int(stop[S].sum())
    bound_holds = not negative and not on_stop
    if negative:
        warnings.warn(f'The reward of {key!r} has negative values, so the exact value is not an upper bound on a '
                      f'return here; bound_share can exceed 1.', UserWarning)
    elif on_stop:
        warnings.warn(f'{on_stop} start point(s) are stop points (boundary points outside the start cluster); for '
                      f'walks from them the exact value is not an upper bound on the return.', UserWarning)
    path, moves, end, entry, G = _walk(nb, R, stop, t, Q, target, S, settings['gamma'], settings['max_step'],
                                       until_entry)
    paths = [mapped[row[row >= 0]] for row in path]
    Vs = V[S]
    share = np.where(Vs > TIE, G / np.where(Vs > TIE, Vs, 1.0), np.nan)
    walks = pd.DataFrame({'start': mapped[S], 'path': paths, 'moves': moves,
                          'end': np.asarray(WALK_ENDS, dtype=object)[end], 'entered': entry > 0,
                          'entry_move': pd.array([int(e) if e > 0 else pd.NA for e in entry], dtype='Int64'),
                          'return': G, 'bound': Vs, 'bound_share': share})
    prefix = key_add or key + '_walks'
    idx = [p.tolist() for p in paths]
    gres.trajectory[prefix + '_idx'] = idx
    points = _mean_path_points(gres, idx)
    gres.trajectory[prefix + '_traj'] = np.vstack(points) if points else np.empty((0, gres.grids['grids'].shape[1]))
    visits = np.bincount(path[path >= 0], minlength=len(mapped))
    gres.grids[prefix + '_visits'] = visits
    entered = walks['entered'].to_numpy()
    gres.trajectory[prefix + '_summary'] = {
        'walks': int(len(walks)), 'entered': float(entered.mean()) if len(walks) else float('nan'),
        'moves_median': float(np.median(moves)) if len(walks) else float('nan'),
        'entry_move_median': float(np.median(entry[entered])) if entered.any() else float('nan'),
        'bound_share_median': float(np.nanmedian(share)) if np.isfinite(share).any() else float('nan'),
        'ends': {e: int((walks['end'] == e).sum()) for e in WALK_ENDS if (walks['end'] == e).any()},
        'bound_holds': bound_holds, 'key': key, 'settings': dict(settings), 'until_entry': bool(until_entry)}
    return walks

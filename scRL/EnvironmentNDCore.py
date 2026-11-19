"""Reward tables and environments on n-D grids.

The rules are those of ``EnvironmentCore`` and ``TabularQCore``; moves are looked up in the neighbour table
of the grid (grid point x move -> next grid point, -1 for a move off the grid).
"""
import random
import time
import warnings

import numpy as np
import pandas as pd

from .GridNDCore import check_labels, get_state_nd


MISSING_STARTS_PROBS = ('starts_probs=True needs start probabilities: call c_rewards(..., starts_keys=[...]) '
                        'first, or use starts_probs=False (starts_prob=False in trainer) to start from the start '
                        'cluster.')


def _as_list(names):
    return [names] if np.isscalar(names) else list(names)


def _time_weight(scaled, beta, mode):
    if mode == 'Decision':
        return np.exp(-beta * scaled)
    if mode == 'Contribution':
        return 1 - np.exp(-beta * scaled)
    raise ValueError('Mode must be one of "Decision" and "Contribution"!')


def _cluster_grids(gres, names):
    clusters = np.asarray(gres.grids['mapped_grids_clusters'])
    mapped = np.asarray(gres.grids['mapped_grids'])
    return list(mapped[np.concatenate([np.flatnonzero(clusters == c) for c in names])])


def _table(gres, bonus):
    """Reward table: -1 for a move off the grid, otherwise the bonus of the grid point moved into."""
    nb = gres.grids['neighbors']
    valid = nb >= 0
    R = np.where(valid, 0.0, -1.0)
    R[valid] += bonus[nb[valid]]
    return pd.DataFrame(R, index=list(gres.grids['mapped_grids']), columns=gres.grids['actions'])


def _kept_mask(gres):
    """
    Kept grid points as an array. On 2-D grids it has the layout of the 2-D code (float, indexed
    [c1, c0]); on 3-D and higher grids it is a boolean mask in the lattice shape, indexed [c0, c1, ...].
    """
    mask = np.zeros(int(np.prod(gres.grids['shape'])), bool)
    mask[gres.grids['mapped_grids']] = True
    mask = mask.reshape(gres.grids['shape'])
    return mask.T.astype(float) if mask.ndim == 2 else mask


def _scaled(t):
    return (t - t.min()) / (t.max() - t.min())


def d_rewards_nd(gres, starts, ends, beta=1, mode='Decision'):
    """
    n-D version of ``d_rewards``: lineage reward table.

    Moving into a grid point of an end cluster earns exp(-beta * s) ('Decision') or 1 - exp(-beta * s)
    ('Contribution'), s being the pseudotime scaled within the end cluster grid points. The kept grid
    points are stored in ``gres.qlearning['matrix']``: on 2-D grids as in the 2-D code (float, indexed
    [c1, c0]), on 3-D and higher grids as a boolean mask in the lattice shape.
    """
    start_time = time.time()
    starts, ends = _as_list(starts), _as_list(ends)
    _time_weight(0.0, beta, mode)
    check_labels(gres, starts + ends)
    mapped = np.asarray(gres.grids['mapped_grids'])
    clusters = np.asarray(gres.grids['mapped_grids_clusters'])
    pseudotime = gres.grids['pseudotime'].loc[mapped].to_numpy(dtype=float)
    is_end = np.isin(clusters, ends)
    bonus = np.zeros(len(mapped))
    bonus[is_end] = _time_weight(_scaled(pseudotime[is_end]), beta, mode)
    gres.qlearning['reward_key'] = '.'.join(map(str, ends))
    gres.grids['ends_cluster_grids'] = _cluster_grids(gres, ends)
    gres.grids['starts_cluster_grids'] = _cluster_grids(gres, starts)
    gres.qlearning[f'd_{mode}_rewards'] = _table(gres, bonus)
    gres.qlearning['matrix'] = _kept_mask(gres)
    print(f'Time used for generating rewards : {(time.time() - start_time):.2f} seconds')


def c_rewards_nd(gres, reward_keys, starts=None, starts_keys=None, punish_keys=None, beta=1, mode='Decision'):
    """
    n-D version of ``c_rewards``: reward table from projected gene expression.

    Moving into a grid point with positive mean reward expression earns that expression weighted by the
    time weight of its pseudotime (scaled within those grid points); punished genes subtract in the same way.
    """
    start_time = time.time()
    _time_weight(0.0, beta, mode)
    if 'proj' not in gres.grids:
        raise ValueError(
            "No projected data found. Please call scRL.project(gres, data) first to project "
            "gene expression data onto the grid embedding before using c_rewards."
        )
    proj = gres.grids['proj']
    missing_keys = [k for k in reward_keys if k not in set(proj.columns)]
    if missing_keys:
        raise KeyError(
            f"reward_keys {missing_keys} not found in projected data columns.\n"
            f"Available columns (first 10): {list(proj.columns)[:10]}\n\n"
            f"Note: c_rewards (gene_rewards) expects gene names from projected expression data.\n"
            f"If you want to use cluster/lineage names for rewards, use d_rewards (lineage_rewards) instead:\n"
            f"  scRL.d_rewards(gres, starts=['cluster1'], ends=['cluster2'], mode='Contribution')"
        )
    mapped = np.asarray(gres.grids['mapped_grids'])
    pseudotime = gres.grids['pseudotime'].loc[mapped].to_numpy(dtype=float)
    bonus = np.zeros(len(mapped))
    reward = proj[reward_keys].mean(axis=1).to_numpy(dtype=float)
    pos = reward > 0
    bonus[pos] += _time_weight(_scaled(pseudotime[pos]), beta, mode) * reward[pos]
    if punish_keys:
        punish = proj[punish_keys].mean(axis=1).to_numpy(dtype=float)
        neg = punish > 0
        bonus[neg] -= _time_weight(_scaled(pseudotime[neg]), beta, mode) * punish[neg]
    if starts:
        starts = _as_list(starts)
        check_labels(gres, starts)
        gres.grids['starts_cluster_grids'] = _cluster_grids(gres, starts)
    else:
        gres.grids['starts_cluster_grids'] = None
    if starts_keys:
        starts_probs = proj[starts_keys].mean(axis=1).to_numpy(dtype=float)
        gres.grids['starts_probs'] = (starts_probs / starts_probs.sum()).astype(float)
    gres.qlearning['reward_key'] = '.'.join(reward_keys)
    gres.qlearning[f'c_{mode}_rewards'] = _table(gres, bonus)
    gres.qlearning['matrix'] = _kept_mask(gres)
    print(f'Time used for reward generation: {(time.time() - start_time):.2f} seconds')


def _reach(neighbors, R, starts, stop, max_step):
    """Breadth-first search over the moves an episode can make; see ``reachability``."""
    dist = np.full(len(neighbors), -1)
    dist[starts] = 0
    first_entry = np.full(len(neighbors), np.inf)
    frontier = np.unique(starts)
    for step in range(1, max_step + 2):
        if not len(frontier):
            break
        nb = neighbors[frontier]
        entered = nb[(nb >= 0) & (R[frontier] > 0)]
        first_entry[entered] = np.minimum(first_entry[entered], step)
        nxt = np.unique(nb[nb >= 0])
        nxt = nxt[dist[nxt] < 0]
        dist[nxt] = step
        frontier = nxt[~stop[nxt]]
    reached = np.isfinite(first_entry)
    return {'rewarded': int(len(np.unique(neighbors[(neighbors >= 0) & (R > 0)]))),
            'reached': int(reached.sum()),
            'min_steps': int(first_entry[reached].min()) if reached.any() else None}


def _stop_mask(gres, starts_grids):
    """Grid points where an episode stops when entered: boundary points outside the start points."""
    stop = (gres.grids['neighbors'] < 0).any(1)
    if starts_grids:
        stop[np.isin(gres.grids['mapped_grids'], list(starts_grids))] = False
    return stop


def reachability(gres, reward_type='d', reward_mode='Decision', max_step=50):
    """
    Check which rewarded grid points an episode can reach under the stop rules and the step cap (any grid,
    2-D scan grids included).

    Episodes start at the start cluster grid points (or, without them, at the grid points with a positive
    start probability), stop when they leave the grid, enter a boundary point outside the start points or
    revisit a point, and make at most ``max_step + 1`` moves.

    Parameters
    ----------
    gres
        Grids results after reward generating.
    reward_type
        'c' or 'd'.
        (Default: 'd')
    reward_mode
        'Decision' or 'Contribution'.
        (Default: 'Decision')
    max_step
        Maximum steps the agent can travel in the environment.
        (Default: 50)

    Returns
    ----------
    dict with 'rewarded' (grid points earning a positive reward when entered), 'reached' (how many of them
    can be entered) and 'min_steps' (fewest moves to enter one, or None).
    """
    mapped = np.asarray(gres.grids['mapped_grids'])
    starts_grids = gres.grids.get('starts_cluster_grids')
    if starts_grids:
        starts = np.flatnonzero(np.isin(mapped, list(starts_grids)))
    else:
        starts = np.flatnonzero(np.asarray(gres.grids['starts_probs']) > 0)
    R = gres.qlearning[f'{reward_type}_{reward_mode}_rewards'].to_numpy(dtype=float)
    from .GridNDCore import is_nd, lattice_from_scan
    lattice = gres if is_nd(gres) else lattice_from_scan(gres)       # 2-D scan grids: the same lattice, with moves
    return _reach(lattice.grids['neighbors'], R, starts, _stop_mask(lattice, starts_grids), max_step)


class _gridEnvND:
    """State shared by the n-D environments: reward table, neighbour table, start and stop points."""

    def __init__(self, gres, max_step, reward_type, reward_mode, starts_probs):
        self.N = gres.grids['n']
        self.rewards_df = gres.qlearning[f'{reward_type}_{reward_mode}_rewards']
        self.mapped_grids = np.asarray(gres.grids['mapped_grids'])
        self.starts_probs = None
        if starts_probs:
            if 'starts_probs' not in gres.grids:
                raise ValueError(MISSING_STARTS_PROBS)
            self.starts_probs = gres.grids['starts_probs']
        self.starts_grids = gres.grids['starts_cluster_grids']
        self.mapped_boundary = gres.grids['mapped_boundary']
        self.pseudotime = gres.grids['pseudotime']
        self.reset_ = False
        self.max_step = max_step
        self._R = self.rewards_df.to_numpy(dtype=float)
        self._next = gres.grids['neighbors']
        self._time = self.pseudotime.loc[self.mapped_grids].to_numpy(dtype=float)
        self._pos = dict(zip(self.mapped_grids.tolist(), range(len(self.mapped_grids))))
        self._stop = _stop_mask(gres, self.starts_grids)
        if self.starts_probs is not None:
            starts = np.flatnonzero(np.asarray(self.starts_probs) > 0)
        else:
            starts = np.flatnonzero(np.isin(self.mapped_grids, list(self.starts_grids or [])))
        if _reach(self._next, self._R, starts, self._stop, max_step)['reached'] == 0:
            warnings.warn(f'No rewarded grid point can be reached from the start points within {max_step + 1} '
                          f'moves under the stop rules; the agent cannot learn this reward. Consider a larger '
                          f'max_step, a larger radius or other start points.', UserWarning)

    def _start(self):
        if self.starts_probs is not None:
            return np.random.choice(self.mapped_grids, 1, p=self.starts_probs).item()
        return random.sample(self.starts_grids, 1)[0]


class deepEnvND(_gridEnvND):
    """
    n-D version of ``deepEnv``: environment for deep reinforcement learning on a grid with a neighbour table.

    Parameters
    ----------
    gres
        Grids results after reward generating.
    X_pca
        The input latent space.
    max_step
        Maximum steps the agent can travel in the environment.
        (Default: 50)
    KNN
        Nearest neighbors for considering each grid points state value.
        (Default: 100)
    reward_type
        The reward type generated, 'c' or 'd'.
        (Default: 'c')
    reward_mode
        The reward mode selected when generating the reward, 'Decision' or 'Contribution'.
        (Default: 'Decision')
    starts_probs
        Whether to sample start points from the start probabilities of ``c_rewards``.
        (Default: True)
    """
    def __init__(self, gres, X_pca, max_step=50, KNN=100, reward_type='c', reward_mode='Decision', starts_probs=True):
        super().__init__(gres, max_step, reward_type, reward_mode, starts_probs)
        self.state_space = get_state_nd(gres, X_pca, KNN)

    def reset(self):
        self.state_idx = self._start()
        self._p = self._pos[self.state_idx]
        self.state = np.mean(self.state_space[self._p], axis=0)
        self.time_step = 0
        self.reset_ = True
        self.trajectory = [self.state_idx]
        self._visited = {self.state_idx}
        return self.state

    def step(self, action):
        if not self.reset_:
            print('Warning: reset the environment first!')
            return
        p = self._p
        q = self._next[p, action]
        self.time_step += 1
        if q < 0:
            return self.state, -2, True, False
        reward = self._R[p, action]
        self._p = q
        self.state_idx = self.mapped_grids[q]
        self.state = np.mean(self.state_space[q], axis=0)
        if self._stop[q]:
            return self.state, reward + (self._time[q] - self._time[p]), True, False
        if self.state_idx in self._visited:
            return self.state, -1, True, False
        self._visited.add(self.state_idx)
        self.trajectory.append(self.state_idx)
        return self.state, reward, False, self.time_step > self.max_step


class tabularEnvND(_gridEnvND):
    """
    n-D version of ``tabularEnv``: environment for tabular Q-learning on a grid with a neighbour table.

    Parameters
    ----------
    gres
        Grids results after reward generating.
    max_step
        Maximum steps the agent can travel in the environment.
        (Default: 50)
    reward_type
        The reward type generated, 'c' or 'd'.
        (Default: 'c')
    reward_mode
        The reward mode selected when generating the reward, 'Decision' or 'Contribution'.
        (Default: 'Decision')
    starts_probs
        Whether to sample start points from the start probabilities of ``c_rewards``.
        (Default: True)
    """
    def reset(self):
        self.state = self._start()
        self._p = self._pos[self.state]
        self.time_step = 0
        self.reset_ = True
        self.trajectory = [self.state]
        self._visited = {self.state}
        return self.state

    def step(self, action):
        if not self.reset_:
            print('Warning: reset the environment first!')
            return
        p = self._p
        q = self._next[p, action]
        if q < 0:
            return self.state, -2, True, False
        reward = self._R[p, action] + (self._time[q] - self._time[p])
        self._p = q
        self.state = self.mapped_grids[q]
        if self._stop[q]:
            return self.state, reward, True, False
        truncation = False
        if self.state in self._visited:
            truncation = True
            reward = -1
        else:
            self._visited.add(self.state)
            self.trajectory.append(self.state)
        self.time_step += 1
        if self.time_step > self.max_step:
            truncation = True
        return self.state, reward, False, truncation

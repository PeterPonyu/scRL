"""Reward tables and environments on n-D grids.

The rules are those of ``EnvironmentCore`` and ``TabularQCore``; moves are looked up in the neighbour table
of the grid (grid point x move -> next grid point, -1 for a move off the grid).
"""
import time

import numpy as np
import pandas as pd

from .GridNDCore import check_labels


def _as_list(names):
    return [names] if isinstance(names, str) else list(names)


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
    mask = np.zeros(int(np.prod(gres.grids['shape'])), bool)
    mask[gres.grids['mapped_grids']] = True
    return mask.reshape(gres.grids['shape'])


def _scaled(t):
    return (t - t.min()) / (t.max() - t.min())


def d_rewards_nd(gres, starts, ends, beta=1, mode='Decision'):
    """
    n-D version of ``d_rewards``: lineage reward table.

    Moving into a grid point of an end cluster earns exp(-beta * s) ('Decision') or 1 - exp(-beta * s)
    ('Contribution'), s being the pseudotime scaled within the end cluster grid points.
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
    gres.qlearning['reward_key'] = '.'.join(ends)
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

"""Combine value maps of several lineages into lineage weights."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.stats import rankdata


def _kept_values(gres, key):
    """Values of a grid key, one per kept grid point in stored order."""
    if key not in gres.grids:
        raise KeyError(f'{key!r} is not stored on the grid; available keys: {sorted(map(str, gres.grids))}')
    values = gres.grids[key]
    mapped = np.asarray(gres.grids['mapped_grids'])
    if isinstance(values, pd.Series) and len(values) == len(mapped) and set(values.index) == set(mapped):
        return values.loc[mapped].to_numpy(dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    if len(values) != len(mapped):
        raise ValueError(f'gres.grids[{key!r}] has {len(values)} values but the grid has {len(mapped)} points.')
    return values


def _summaries(W, names):
    """Heaviest lineage ('none' where a row has no weights) and normalised entropy (NaN there) of each row."""
    missing = np.isnan(W).any(1)
    filled = np.where(np.isnan(W), 0, W)
    top = np.asarray(names, dtype=object)[filled.argmax(1)].astype(str).astype(object)
    top[missing] = 'none'
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.where(filled > 0, filled * np.log(filled), 0).sum(1) / np.log(W.shape[1])
    entropy[missing] = np.nan
    return top, entropy


def _store_weights(gres, W, names, key_add):
    """
    Store lineage weights (kept grid points x lineages, NaN rows allowed) on the grid and, brought back with
    ``project_back``, on the cells (rows without weights count as 0 there; cell rows renormalised).
    """
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as plt
    from .GridCore import project_back
    colors = {n: to_hex(plt.get_cmap('tab10')(i % 10)) for i, n in enumerate(names)}
    colors['none'] = '#d9d9d9'
    table = pd.DataFrame(W, index=list(gres.grids['mapped_grids']), columns=names)
    top, entropy = _summaries(W, names)
    for i, n in enumerate(names):
        gres.grids[f'{key_add}_{n}'] = np.where(np.isnan(W[:, i]), 0, W[:, i])
        project_back(gres, f'{key_add}_{n}', negative=True)
        gres.grids[f'{key_add}_{n}'] = W[:, i]
    gres.grids[key_add] = table
    gres.grids[f'{key_add}_top'], gres.grids[f'{key_add}_entropy'] = top, entropy
    gres.grids[f'{key_add}_top_colors'] = colors
    Wc = np.clip(np.column_stack([gres.embedding[f'{key_add}_{n}'] for n in names]), 0, None)
    with np.errstate(invalid='ignore', divide='ignore'):
        Wc = Wc / Wc.sum(1, keepdims=True)
    for i, n in enumerate(names):
        gres.embedding[f'{key_add}_{n}'] = Wc[:, i]
    gres.embedding[f'{key_add}_top'], gres.embedding[f'{key_add}_entropy'] = _summaries(Wc, names)
    gres.embedding[f'{key_add}_top_colors'] = dict(colors)
    return table


def fate_gate(gres, keys, names=None, tau=.1, key_add='gate'):
    """
    Combine one value map per lineage into lineage weights.

    Each map is rank-scaled to [0, 1] over the kept grid points that have a value in every map, and the
    weights are the softmax of the scaled values divided by ``tau`` across lineages: a grid point gets most of
    its weight from the lineage in whose map it ranks highest. A small ``tau`` gives nearly one-hot weights.
    Grid points without a value in some map get no weights (NaN), ``'none'`` as heaviest lineage and NaN
    entropy.

    Parameters
    ----------
    gres
        Grids results holding the value maps.
    keys
        Grid keys of the value maps, one per lineage, at least two (for example from ``exact_values``).
    names
        Lineage names.
        (Default: None, the keys)
    tau
        Temperature of the softmax.
        (Default: .1)
    key_add
        Prefix of the stored results.
        (Default: 'gate')

    Returns
    ----------
    DataFrame of the weights (index kept grid indices, one column per lineage). Also stored on the grid:
    ``<key_add>`` (the DataFrame), ``<key_add>_<name>`` (weights), ``<key_add>_top`` (heaviest lineage),
    ``<key_add>_entropy`` (Shannon entropy divided by log of the number of lineages: 0 decided, 1 undecided)
    and ``<key_add>_top_colors`` (lineage colours). The same are stored for the cells in ``gres.embedding``,
    from the weights brought to the cells with ``project_back`` and renormalised.
    """
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as plt
    from .GridCore import project_back
    keys = list(keys)
    if len(keys) < 2:
        raise ValueError('fate_gate needs at least two value maps.')
    names = keys if names is None else [str(n) for n in names]
    if len(names) != len(keys):
        raise ValueError(f'names has {len(names)} entries but there are {len(keys)} keys.')
    values = np.column_stack([_kept_values(gres, k) for k in keys])
    finite = np.isfinite(values).all(1)
    if finite.sum() < 2:
        raise ValueError('fate_gate needs at least two grid points with a value in every map.')
    scaled = np.column_stack([(rankdata(v) - 1) / (finite.sum() - 1) for v in values[finite].T])
    z = scaled / tau
    w = np.exp(z - z.max(1, keepdims=True))
    W = np.full(values.shape, np.nan)
    W[finite] = w / w.sum(1, keepdims=True)
    return _store_weights(gres, W, names, key_add)


def fate_probabilities(gres, ends, forward=1.0, policy=None, key_add='fate'):
    """
    For every kept grid point, the probability that a walk starting there enters the end points of each
    lineage before those of any other lineage.

    The walk moves from a grid point to a neighbouring kept point with weight
    exp(forward * (t_q - t_p) / dt), t being pseudotime and dt the median pseudotime difference of a move, so
    ``forward`` biases the walk toward later pseudotime independently of the pseudotime scale (``forward=0``:
    the unbiased walk). With ``policy`` the move probabilities come from an array or from the actor of a
    trained learner instead. The end points of a lineage (the grid points labelled with its clusters) absorb
    the walk. Grid points from which no end point can be reached get no shares (NaN), and the walk never
    steps onto them. Solved exactly with one sparse factorisation.

    Parameters
    ----------
    gres
        Grids results after ``project_cluster`` (and ``align_pseudotime`` unless ``forward`` is 0).
    ends
        Dict lineage name -> cluster name or list of cluster names (at least two lineages; a cluster belongs to
        one lineage only).
    forward
        Bias of the walk toward later pseudotime.
        (Default: 1.0)
    policy
        Move probabilities instead of the walk: an array (kept grid points x moves; rows need not be
        normalised, moves off the grid are ignored), or a trained ``scRL.trainer`` with an actor ('ActorCritic',
        'SAC').
        (Default: None)
    key_add
        Prefix of the stored results.
        (Default: 'fate')

    Returns
    ----------
    DataFrame of the shares (index kept grid indices, one column per lineage), stored as ``fate_gate`` stores
    its weights (``<key_add>``, ``<key_add>_<name>``, ``<key_add>_top`` with 'none' where there are no shares,
    ``<key_add>_entropy``, ``<key_add>_top_colors``, and the same for the cells), so that ``plot_fates`` and
    ``compare_fates`` read either.
    """
    from .GridNDCore import check_labels, is_nd, lattice_from_scan
    if not isinstance(ends, dict) or len(ends) < 2:
        raise ValueError('ends must be a dict of at least two lineages: name -> cluster name(s).')
    names = [str(n) for n in ends]
    lineage_clusters = {n: [c] if np.isscalar(c) else list(c) for n, c in zip(names, ends.values())}
    seen = {}
    for n, cs in lineage_clusters.items():
        for c in cs:
            if c in seen:
                raise ValueError(f'Cluster {c!r} is in two lineages ({seen[c]!r} and {n!r}).')
            seen[c] = n
    check_labels(gres, list(seen))
    lattice = gres if is_nd(gres) else lattice_from_scan(gres)
    nb = lattice.grids['neighbors']
    M = len(nb)
    labels = np.asarray(gres.grids['mapped_grids_clusters'])
    owner = np.full(M, -1)
    for i, n in enumerate(names):
        owner[np.isin(labels, lineage_clusters[n])] = i
    rows, cols = np.nonzero(nb >= 0)
    q = nb[rows, cols]
    if policy is None:
        if forward == 0:
            w = np.ones(len(rows))
        else:
            t = gres.grids['pseudotime'].loc[np.asarray(gres.grids['mapped_grids'])].to_numpy(dtype=float)
            diff = t[q] - t[rows]
            dt = np.median(np.abs(diff[diff != 0]))
            w = np.exp(forward * diff / dt)
    else:
        w = _policy_weights(gres, policy, nb)[rows, cols]
    W = sparse.csr_matrix((w, (rows, q)), shape=(M, M))
    reach = owner >= 0
    while True:
        grown = reach | (np.asarray((W > 0) @ reach.astype(float)).ravel() > 0)
        if (grown == reach).all():
            break
        reach = grown
    Wr = W.multiply(reach[None, :]).tocsr()
    P = sparse.diags(1 / np.maximum(np.asarray(Wr.sum(1)).ravel(), 1e-300)) @ Wr
    free = (owner < 0) & reach
    H = np.full((M, len(names)), np.nan)
    for i in range(len(names)):
        H[owner >= 0, i] = (owner[owner >= 0] == i).astype(float)
    if free.any():
        P_free = P[free][:, free].tocsc()
        lu = splu((sparse.eye(free.sum(), format='csc') - P_free).tocsc())
        for i in range(len(names)):
            H[free, i] = lu.solve(np.asarray(P[free][:, owner == i].sum(1)).ravel())
    return _store_weights(gres, H, names, key_add)


def _policy_weights(gres, policy, nb):
    """Move weights (kept grid points x moves) from an array or from the actor of a trained learner."""
    if isinstance(policy, np.ndarray):
        weights = np.asarray(policy, dtype=float)
        if weights.shape != nb.shape:
            raise ValueError(f'policy has shape {weights.shape}; it needs one row per kept grid point and one '
                             f'column per move {nb.shape}.')
        return weights
    agent = getattr(policy, 'agent', None)
    if agent is None or not hasattr(agent, 'actor'):
        raise ValueError('policy needs an actor (an ActorCritic or SAC trainer) or an array of move weights.')
    import torch
    net = agent.actor
    device = next(iter(net.parameters())).device if hasattr(net, 'parameters') else None
    states = torch.tensor(policy.env.state_space.mean(axis=1), dtype=torch.float, device=device)
    return net(states).detach().cpu().numpy()

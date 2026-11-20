"""Combine value maps of several lineages into lineage weights."""
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
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


RESERVED = ('top', 'entropy', 'none', 'reached')


def _check_names(names):
    clash = [n for n in names if str(n) in RESERVED]
    if clash:
        raise ValueError(f'Lineage name(s) {clash} are reserved for the stored results; choose other names.')


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
    entropy. The gate ranks grid points within a lineage and marks the heaviest lineage; its weights are not
    shares of cells (for those use ``fate_probabilities``).

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
    _check_names(names)
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


def fate_probabilities(gres, ends, forward=.25, steps=5000, policy=None, key_add='fate', min_reached=.5):
    """
    For every kept grid point, the share of walks starting there that enter the end points of each lineage
    first, within ``steps`` moves.

    The walk moves from a grid point to a neighbouring kept point with weight
    exp(forward * (t_q - t_p) / dt), t being pseudotime and dt the median pseudotime difference of a move, so
    ``forward`` biases the walk toward later pseudotime independently of the pseudotime scale (``forward=0``:
    the unbiased walk). With ``policy`` the move probabilities come from an array or from the actor of a trained
    learner instead. The end points of a lineage (the grid points labelled with its clusters) absorb the walk.
    The shares are not renormalised: a grid point's shares sum to the share of its walks that arrive at any
    listed lineage within ``steps`` moves (stored as ``<key_add>_reached``). A region whose walks do not arrive,
    for example a tip that is not among the listed lineages, therefore gets low shares and ``'none'`` as
    heaviest lineage. At early grid points the split among lineages depends on ``forward`` and on which
    clusters are given as the ends of each lineage.

    The default is a mild bias (``forward=0.25``) with a large move budget (``steps=5000``). At ``forward``
    of 1 and above each move is strongly drawn to later pseudotime: at 1, a move one median pseudotime step
    later weighs about seven times (e squared) a move the same step earlier, so walks turn back toward a
    neighbouring branch less often. A smaller value lets them do so more often, which can keep fate signals of
    neighbouring lineages at early grid points, and needs more moves to arrive, hence the larger budget. When
    more than 10% of the grid points still gain over 0.01 of arriving share in the last tenth of the moves, a
    warning says so; then raise ``steps`` or accept that those points show no lineage (check
    ``<key_add>_reached``). The check is skipped for ``steps`` below 20, a deliberately short horizon.

    Parameters
    ----------
    gres
        Grids results after ``project_cluster`` (and ``align_pseudotime`` unless ``forward`` is 0).
    ends
        Dict lineage name -> cluster name or list of cluster names (at least two lineages; a cluster belongs to
        one lineage only).
    forward
        Bias of the walk toward later pseudotime; 0 is the unbiased walk. Not used with ``policy``.
        (Default: 0.25)
    steps
        Number of moves within which a walk must arrive; a milder bias needs more moves. With ``policy`` it
        only bounds the number of moves.
        (Default: 5000)
    policy
        Move probabilities instead of the walk: an array (kept grid points x moves; finite and not negative;
        rows need not be normalised, moves off the grid are ignored, a row of zeros means the walk stops
        there), or a trained
        ``scRL.trainer`` with an actor ('ActorCritic', 'SAC'). With ``policy``, ``forward`` is not used and
        ``steps`` only bounds the number of moves.
        (Default: None)
    key_add
        Prefix of the stored results.
        (Default: 'fate')
    min_reached
        Share of arriving walks below which a grid point (or cell) has no heaviest lineage ('none') and no
        entropy (NaN).
        (Default: .5)

    Returns
    ----------
    DataFrame of the shares (index kept grid indices, one column per lineage). Stored on the grid: the table
    ``<key_add>``, ``<key_add>_<name>``, ``<key_add>_reached``, ``<key_add>_top``, ``<key_add>_entropy`` (of the
    shares renormalised over the lineages) and ``<key_add>_top_colors``; the same for the cells, brought back
    with the kernel of ``project_back``. ``plot_fates`` and ``compare_fates`` read them.
    """
    from .GridNDCore import check_labels, is_nd, lattice_from_scan
    if not isinstance(ends, dict) or len(ends) < 2:
        raise ValueError('ends must be a dict of at least two lineages: name -> cluster name(s).')
    if steps < 1:
        raise ValueError('steps must be at least 1.')
    names = [str(n) for n in ends]
    _check_names(names)
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
    total = np.asarray(W.sum(1)).ravel()
    P = (sparse.diags(np.where(total > 0, 1 / np.where(total > 0, total, 1), 0)) @ W).tocsr()
    free = owner < 0
    P_ff = P[free][:, free].tocsr()
    B = np.column_stack([np.asarray(P[free][:, owner == i].sum(1)).ravel() for i in range(len(names))])
    H_free = np.zeros_like(B)
    steps = int(steps)
    snapshot_at = steps - max(1, steps // 10)               # the last tenth of the moves
    for k in range(steps):
        if k == snapshot_at:
            before = H_free.copy()
        H_free = P_ff @ H_free + B
    still = int(((H_free - before).sum(1) > .01).sum()) / M
    if steps >= 20 and still > .1:
        warnings.warn(f"{still:.0%} of grid points are still collecting arriving walks at the last of {steps} "
                      f"moves; raise steps (check '{key_add}_reached') or accept that these points show no "
                      f"lineage.", UserWarning, stacklevel=2)
    H = np.zeros((M, len(names)))
    H[free] = H_free
    for i in range(len(names)):
        H[owner == i, i] = 1.0
    return _store_shares(gres, H, names, key_add, min_reached)


def _store_shares(gres, H, names, key_add, min_reached):
    """Store lineage shares that need not sum to one, with the share of arriving walks, on the grid and cells."""
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as plt
    from .GridCore import project_back
    colors = {n: to_hex(plt.get_cmap('tab10')(i % 10)) for i, n in enumerate(names)}
    colors['none'] = '#d9d9d9'

    def summaries(S):
        reached = S.sum(1)
        top = np.asarray(names, dtype=object)[S.argmax(1)].astype(str).astype(object)
        low = reached < min_reached
        top[low] = 'none'
        with np.errstate(divide='ignore', invalid='ignore'):
            R = S / reached[:, None]
            entropy = -np.where(R > 0, R * np.log(R), 0).sum(1) / np.log(S.shape[1])
        entropy[low] = np.nan
        return reached, top, entropy

    table = pd.DataFrame(H, index=list(gres.grids['mapped_grids']), columns=names)
    for i, n in enumerate(names):
        gres.grids[f'{key_add}_{n}'] = H[:, i]
        project_back(gres, f'{key_add}_{n}', negative=True)
    gres.grids[key_add] = table
    gres.grids[f'{key_add}_reached'], gres.grids[f'{key_add}_top'], gres.grids[f'{key_add}_entropy'] = summaries(H)
    gres.grids[f'{key_add}_top_colors'] = colors
    C = np.clip(np.column_stack([gres.embedding[f'{key_add}_{n}'] for n in names]), 0, 1)
    over = C.sum(1, keepdims=True)
    C = np.where(over > 1, C / np.where(over > 1, over, 1), C)        # the kernel average cannot exceed 1 in theory
    for i, n in enumerate(names):
        gres.embedding[f'{key_add}_{n}'] = C[:, i]
    reached, top, entropy = summaries(C)
    gres.embedding[f'{key_add}_reached'], gres.embedding[f'{key_add}_top'] = reached, top
    gres.embedding[f'{key_add}_entropy'] = entropy
    gres.embedding[f'{key_add}_top_colors'] = dict(colors)
    return table


def _policy_weights(gres, policy, nb):
    """Move weights (kept grid points x moves) from an array or from the actor of a trained learner."""
    if isinstance(policy, np.ndarray):
        weights = np.asarray(policy, dtype=float)
        if weights.shape != nb.shape:
            raise ValueError(f'policy has shape {weights.shape}; it needs one row per kept grid point and one '
                             f'column per move {nb.shape}.')
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('policy weights must be finite and not negative (a row of zeros stops the walk).')
        return weights
    agent = getattr(policy, 'agent', None)
    if agent is None or not hasattr(agent, 'actor'):
        raise ValueError('policy needs an actor (an ActorCritic or SAC trainer) or an array of move weights.')
    import torch
    net = agent.actor
    device = next(iter(net.parameters())).device if hasattr(net, 'parameters') else None
    states = policy.env.state_space.mean(axis=1)
    if len(states) != len(nb):
        raise ValueError(f'The trainer was built on a grid of {len(states)} kept points; this grid has {len(nb)}. '
                         f'Pass a trainer built on this grid.')
    out = net(torch.tensor(states, dtype=torch.float, device=device)).detach().cpu().numpy().astype(np.float64)
    if out.shape != nb.shape:
        raise ValueError(f'The actor gives {out.shape[1]} moves; this grid has {nb.shape[1]}.')
    return out

"""Fate corridors: where the walks that leave a multipotent start region travel on their way to each lineage."""
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu

from .FateCore import _check_names, _policy_weights


def _lineages(gres, ends):
    """Lineage names, and the lineage owning each kept grid point (-1: none), as in ``fate_probabilities``."""
    from .GridNDCore import check_labels
    if not isinstance(ends, dict) or len(ends) < 2:
        raise ValueError('ends must be a dict of at least two lineages: name -> cluster name(s).')
    names = [str(n) for n in ends]
    _check_names(names)
    clusters = {n: [c] if np.isscalar(c) else list(c) for n, c in zip(names, ends.values())}
    seen = {}
    for n, cs in clusters.items():
        for c in cs:
            if c in seen:
                raise ValueError(f'Cluster {c!r} is in two lineages ({seen[c]!r} and {n!r}).')
            seen[c] = n
    check_labels(gres, list(seen))
    labels = np.asarray(gres.grids['mapped_grids_clusters'])
    owner = np.full(len(labels), -1)
    for i, n in enumerate(names):
        owner[np.isin(labels, clusters[n])] = i
    return names, owner


def _walk(gres, nb, forward, policy):
    """Row-stochastic move matrix of the walk of ``fate_probabilities`` (kept grid points)."""
    M = len(nb)
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
    return (sparse.diags(np.where(total > 0, 1 / np.where(total > 0, total, 1), 0)) @ W).tocsr()


def _reaching(P, absorbing):
    """Grid points from which the walk can reach an absorbing point."""
    reach = absorbing.copy()
    PT = P.T.tocsr()
    frontier = absorbing.copy()
    while frontier.any():
        new = (np.asarray(PT[frontier].sum(0)).ravel() > 0) & ~reach
        reach |= new
        frontier = new
    return reach


def _cell_counts(gres, dims=None):
    """Number of cells whose nearest kept grid point is each kept grid point."""
    from scipy.spatial import cKDTree
    mapped = np.asarray(gres.grids['mapped_grids'])
    G = np.asarray(gres.grids['grids'], dtype=float)[mapped]
    X = np.asarray(gres.embedding['embedding'], dtype=float)
    d = min(G.shape[1], X.shape[1])
    nearest = cKDTree(G[:, :d]).query(X[:, :d])[1]
    return np.bincount(nearest, minlength=len(mapped)).astype(float)


def _start_distribution(gres, start, weights, owner):
    mapped = np.asarray(gres.grids['mapped_grids'])
    M = len(mapped)
    if isinstance(start, np.ndarray) and start.dtype == bool:
        if len(start) != M:
            raise ValueError(f'A start region given as a mask needs one value per kept grid point ({M}).')
        mask = start.copy()
    else:
        names = [start] if np.isscalar(start) else list(start)
        labels = np.asarray(gres.grids['mapped_grids_clusters']).astype(str)
        unknown = [c for c in names if str(c) not in set(labels.tolist())]
        if unknown:
            raise ValueError(f'Start cluster(s) {unknown} label no kept grid point.')
        mask = np.isin(labels, [str(c) for c in names])
    if (mask & (owner >= 0)).any():
        warnings.warn(f'{int((mask & (owner >= 0)).sum())} start grid point(s) lie in an end cluster and are left '
                      'out of the start region.', UserWarning, stacklevel=3)
        mask &= owner < 0
    if not mask.any():
        raise ValueError('The start region has no grid point outside the end clusters.')
    if weights == 'cells':
        mu = _cell_counts(gres) * mask
        if mu.sum() == 0:
            warnings.warn('No cell lies nearest to the start region; its grid points are weighted equally.',
                          UserWarning, stacklevel=3)
            mu = mask.astype(float)
    elif weights == 'uniform':
        mu = mask.astype(float)
    else:
        raise ValueError("weights is 'cells' or 'uniform'.")
    return mu / mu.sum()


def fate_corridors(gres, ends, start, weights='cells', forward=.25, policy=None, key_add='corridor'):
    """
    Fate corridors from a start region: for every lineage, how often the walks that leave the start region and end
    in that lineage pass through each grid point; and the fate potential and potency of every grid point.

    A multipotent state is a region of the grid, not a point, and a differentiation path is a corridor that holds
    many grid points (and cells) at each stage. The walk is that of ``fate_probabilities`` (``forward`` biases it
    toward later pseudotime; the end points of each lineage absorb it), started from the start region with the
    weights below. Everything is computed exactly by sparse linear solves, without sampling and without a move
    limit: the fate potential h_l(p) is the probability that the walk from p ends in lineage l (the limit of the
    walk shares for many moves); the visits g(p) are the expected number of visits to p by a walk from the start
    region; the corridor of lineage l is C_l(p) = g(p) h_l(p) / P_l, the expected number of visits to p by a walk
    from the start region that ends in l, where P_l is the probability of ending in l. At the end points of l,
    C_l is the probability of entering the lineage there. In reinforcement-learning terms h_l is the value of the
    walk policy for a reward of 1 on reaching l, and C_l is that policy's state occupancy over the episodes that
    reach l. The potency of a point is exp(entropy of its fate potentials renormalised over the lineages): the
    effective number of fates still open there, from 1 to the number of lineages.

    Corridors of several lineages coincide where they leave the start region (the shared multipotent state) and
    part where the fate is decided; ``corridor_profile`` measures this stage by stage and ``plot_corridors`` draws
    the corridors in two or three dimensions.

    Parameters
    ----------
    gres
        Grids results after ``project_cluster`` and ``align_pseudotime``.
    ends
        Dict lineage name -> end cluster(s), as in ``fate_probabilities``.
    start
        The start region: a cluster name or a list of cluster names (their kept grid points), or a boolean mask
        over the kept grid points (for example ``gres.grids['<key_add>_potency'] >= 2`` from an earlier call).
        Grid points in end clusters are left out.
    weights
        'cells': each start grid point weighted by the number of cells whose nearest kept grid point it is (the
        region as the cells occupy it); 'uniform': all start grid points weighted equally.
        (Default: 'cells')
    forward
        Forward bias of the walk, as in ``fate_probabilities``.
        (Default: 0.25)
    policy
        Move probabilities instead of the walk, as in ``fate_probabilities`` (an array of kept grid points x moves,
        or a trained ``scRL.trainer`` with an actor): the corridors of that policy.
        (Default: None)
    key_add
        Prefix of the stored results.
        (Default: 'corridor')

    Returns
    ----------
    DataFrame, one row per lineage: probability that a walk from the start region ends in the lineage, mean
    number of moves of such a walk, and the start region's size. Stored on the grid: ``<key_add>_<name>`` (the
    corridor C_l per kept grid point, also brought to the cells), ``<key_add>_fate`` (fate potentials, a table),
    ``<key_add>_potency`` (also for the cells), ``<key_add>_start`` (start weights), ``<key_add>_settings``.
    """
    from .GridCore import project_back
    from .GridNDCore import is_nd, lattice_from_scan
    names, owner = _lineages(gres, ends)
    lattice = gres if is_nd(gres) else lattice_from_scan(gres)
    nb = lattice.grids['neighbors']
    M, L = len(nb), len(names)
    P = _walk(gres, nb, forward, policy)
    absorbing = owner >= 0
    ok = _reaching(P, absorbing) & ~absorbing                # transient points that can still arrive
    mu = _start_distribution(gres, start, weights, owner)
    trapped = (mu > 0) & ~ok & ~absorbing
    if trapped.any():
        warnings.warn(f'{trapped.sum()} start grid point(s) cannot reach any lineage and are left out.',
                      UserWarning, stacklevel=2)
        mu = np.where(trapped, 0, mu)
        if mu.sum() == 0:
            raise ValueError('No start grid point can reach a lineage.')
        mu = mu / mu.sum()
    idx = np.flatnonzero(ok)
    Q = P[idx][:, idx].tocsc()
    lu = splu((sparse.identity(len(idx), format='csc') - Q).tocsc())
    to_end = P[idx]
    B = np.column_stack([np.asarray(to_end[:, owner == i].sum(1)).ravel() for i in range(L)])
    H_ok = lu.solve(B)                                       # fate potentials of the transient points
    g = lu.solve(mu[idx], trans='T')                         # expected visits from the start region
    H = np.zeros((M, L))
    H[idx] = H_ok
    for i in range(L):
        H[owner == i, i] = 1.0
    prob = mu[idx] @ H_ok
    entry = np.asarray(to_end.T @ g).ravel()                 # probability of entering each point from outside
    C = np.zeros((M, L))
    for i in range(L):
        if prob[i] > 0:
            C[idx, i] = g * H_ok[:, i] / prob[i]
            on = owner == i
            C[on, i] = entry[on] / prob[i]
    if prob.sum() < .99:
        warnings.warn(f'Walks from the start region end in a listed lineage with probability {prob.sum():.3f}; the '
                      'rest stay on grid parts without an end point.', UserWarning, stacklevel=2)
    total = H.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        R = H / total[:, None]
        ent = -np.where(R > 0, R * np.log(R), 0).sum(1)
    potency = np.where(total >= .5, np.exp(ent), np.nan)
    mapped = np.asarray(gres.grids['mapped_grids'])
    for i, n in enumerate(names):
        gres.grids[f'{key_add}_{n}'] = C[:, i]
        project_back(gres, f'{key_add}_{n}', negative=True)
    gres.grids[f'{key_add}_fate'] = pd.DataFrame(H, index=mapped, columns=names)
    gres.grids[f'{key_add}_potency'] = potency
    project_back(gres, f'{key_add}_potency', negative=True)
    gres.grids[f'{key_add}_start'] = mu
    gres.grids[f'{key_add}_settings'] = dict(names=names, start=start if not isinstance(start, np.ndarray)
                                             else 'mask', weights=weights, forward=forward,
                                             policy=policy is not None)
    moves = C[idx].sum(0)
    return pd.DataFrame({'probability': prob, 'moves': moves, 'start_points': int((mu > 0).sum()),
                         'start_cells': float(_cell_counts(gres)[mu > 0].sum())}, index=pd.Index(names, name='lineage'))


def _decision_terms(P, owner, mu):
    """
    Fate potentials, decision intensities and visits of the walk ``P`` whose end points (``owner`` >= 0) absorb it.

    Returns the fate potentials H (points x lineages), the decision intensities d (the variance of the one-step
    change of H over the moves of the walk, zero at end points and at points that cannot reach a lineage), the
    expected visits g from the start distribution ``mu`` and the mask of transient points that can reach a lineage.
    Moves into grid parts without an end point count as a move to a point whose fate potential is 0, so that
    h (1 - h) = (I - Q)^-1 d holds for every transient point, with Q the moves among the points that can arrive.
    """
    M, L = P.shape[0], int(owner.max()) + 1
    absorbing = owner >= 0
    ok = _reaching(P, absorbing) & ~absorbing
    idx = np.flatnonzero(ok)
    lu = splu((sparse.identity(len(idx), format='csc') - P[idx][:, idx]).tocsc())
    to_end = P[idx]
    B = np.column_stack([np.asarray(to_end[:, owner == i].sum(1)).ravel() for i in range(L)])
    H = np.zeros((M, L))
    H[idx] = lu.solve(B)
    for i in range(L):
        H[owner == i, i] = 1.0
    g = np.zeros(M)
    g[idx] = lu.solve(mu[idx], trans='T')
    coo = P.tocoo()
    move = ok[coo.row]
    rows, cols, w = coo.row[move], coo.col[move], coo.data[move]
    d = np.zeros((M, L))
    for i in range(L):
        d[:, i] = np.bincount(rows, weights=w * (H[cols, i] - H[rows, i]) ** 2, minlength=M)
    return H, d, g, ok


def fate_decisions(gres, ends, start, weights='cells', forward=.25, policy=None, key_add='decision'):
    """
    Where on the grid each lineage's fate is decided: the one-step variance of the fate potential, and its
    visit-weighted map from a start region.

    The walk, its end points and the start region are those of ``fate_corridors``. For lineage l with fate
    potential h_l (the probability that a walk from a grid point ends in l), the decision intensity at a transient
    grid point p is d_l(p) = sum_q P(p, q) (h_l(q) - h_l(p))^2, the variance of the one-step change of h_l (its mean
    is zero, because h_l is harmonic). It measures how much of the choice for l is made by one move from p, and is
    zero at the end points and wherever only one outcome for l remains open. Along a walk the intensities add up
    to the uncertainty: for every transient point that can reach a lineage, h_l(p) (1 - h_l(p)) = sum_q G(p, q)
    d_l(q), with G = (I - Q)^-1 the expected visits among the transient points that can reach a lineage. Walks that
    enter a part of the grid without an end point are lost; a move into such a part counts as a move to a point
    whose fate potential is 0 (the lost walks are a further outcome, not one of the lineages), and the identity then
    holds for the matrix of the walk as it is. When all walks arrive it is the variance decomposition of the
    outcome indicator of l.

    The decision map of the start region is D_l(p) = g(p) d_l(p), with g(p) the expected number of visits to p by a
    walk from the start region (as in ``fate_corridors``). It sums over the grid to the variance of the outcome
    indicator of l from the start region, sum_p mu(p) h_l(p) (1 - h_l(p)), with mu the start weights: the share of
    that variance spent at each grid point. In reinforcement-learning terms d_l is the variance of the
    temporal-difference target of the value h_l under the walk.

    Parameters
    ----------
    gres
        Grids results after ``project_cluster`` and ``align_pseudotime``.
    ends
        Dict lineage name -> end cluster(s), as in ``fate_corridors``.
    start
        The start region, as in ``fate_corridors``.
    weights
        'cells' or 'uniform' weights of the start grid points, as in ``fate_corridors``.
        (Default: 'cells')
    forward
        Forward bias of the walk, as in ``fate_probabilities``.
        (Default: 0.25)
    policy
        Move probabilities instead of the walk, as in ``fate_corridors``.
        (Default: None)
    key_add
        Prefix of the stored results.
        (Default: 'decision')

    Returns
    ----------
    DataFrame, one row per lineage: probability that a walk from the start region ends in the lineage, the
    variance of its outcome indicator from the start region (the total of the decision map) and the start
    region's size. Stored on the grid: ``<key_add>_intensity`` and ``<key_add>`` (tables of the decision
    intensity and the decision map, kept grid points x lineages), ``<key_add>_intensity_<name>`` and
    ``<key_add>_<name>`` (the same per lineage, also brought to the cells), ``<key_add>_total`` (the decision map
    summed over the lineages, also for the cells), ``<key_add>_start`` (start weights) and ``<key_add>_settings``.
    """
    from .GridCore import project_back
    from .GridNDCore import is_nd, lattice_from_scan
    if 'mapped_grids_clusters' not in gres.grids:
        raise KeyError('No clusters on the grid: call project_cluster first.')
    names, owner = _lineages(gres, ends)
    clash = [n for n in names if n in ('intensity', 'total', 'start', 'settings')]
    if clash:
        raise ValueError(f'Lineage name(s) {clash} are reserved for the stored results; choose other names.')
    if policy is None and forward != 0 and 'pseudotime' not in gres.grids:
        raise KeyError('The walk is biased toward later pseudotime: call align_pseudotime first (or forward=0).')
    lattice = gres if is_nd(gres) else lattice_from_scan(gres)
    P = _walk(gres, lattice.grids['neighbors'], forward, policy)
    mu = _start_distribution(gres, start, weights, owner)
    trapped = (mu > 0) & ~_reaching(P, owner >= 0) & (owner < 0)
    if trapped.any():
        warnings.warn(f'{trapped.sum()} start grid point(s) cannot reach any lineage and are left out.',
                      UserWarning, stacklevel=2)
        mu = np.where(trapped, 0, mu)
        if mu.sum() == 0:
            raise ValueError('No start grid point can reach a lineage.')
        mu = mu / mu.sum()
    H, d, g, ok = _decision_terms(P, owner, mu)
    D = g[:, None] * d
    mapped = np.asarray(gres.grids['mapped_grids'])
    for i, n in enumerate(names):
        gres.grids[f'{key_add}_intensity_{n}'] = d[:, i]
        project_back(gres, f'{key_add}_intensity_{n}', negative=True)
        gres.grids[f'{key_add}_{n}'] = D[:, i]
        project_back(gres, f'{key_add}_{n}', negative=True)
    gres.grids[f'{key_add}_intensity'] = pd.DataFrame(d, index=mapped, columns=names)
    gres.grids[f'{key_add}'] = pd.DataFrame(D, index=mapped, columns=names)
    gres.grids[f'{key_add}_total'] = D.sum(1)
    project_back(gres, f'{key_add}_total', negative=True)
    gres.grids[f'{key_add}_start'] = mu
    gres.grids[f'{key_add}_settings'] = dict(names=names, start=start if not isinstance(start, np.ndarray)
                                             else 'mask', weights=weights, forward=forward,
                                             policy=policy is not None)
    prob = mu @ H
    if prob.sum() < .99:
        warnings.warn(f'Walks from the start region end in a listed lineage with probability {prob.sum():.3f}; the '
                      'rest stay on grid parts without an end point.', UserWarning, stacklevel=2)
    return pd.DataFrame({'probability': prob, 'variance': mu @ (H * (1 - H)), 'start_points': int((mu > 0).sum())},
                        index=pd.Index(names, name='lineage'))


def _stored(gres, key_add):
    if f'{key_add}_settings' not in gres.grids:
        raise KeyError(f'No corridors under {key_add!r}: call fate_corridors(key_add={key_add!r}) first.')
    names = gres.grids[f'{key_add}_settings']['names']
    C = np.column_stack([np.asarray(gres.grids[f'{key_add}_{n}'], dtype=float) for n in names])
    H = gres.grids[f'{key_add}_fate'].to_numpy(dtype=float)
    return names, C, H


def _core(values, share):
    """Mask of the fewest points holding ``share`` of the total of ``values``."""
    order = np.argsort(values)[::-1]
    cum = np.cumsum(values[order])
    keep = np.zeros(len(values), bool)
    if cum[-1] <= 0:
        return keep
    n = int(np.searchsorted(cum, share * cum[-1]) + 1)
    keep[order[:n]] = True
    return keep


def corridor_profile(gres, key_add='corridor', bins=20, core=.8, shared=.5, min_mass=.02):
    """
    The corridors stage by stage along pseudotime: their width, the grid points and cells they hold, how
    committed the walks in them are, where they overlap and where they part, and the order of the fate decisions.

    Parameters
    ----------
    gres
        Grids results after ``fate_corridors``.
    key_add
        Prefix given to ``fate_corridors``.
        (Default: 'corridor')
    bins
        Number of equal-width pseudotime stages.
        (Default: 20)
    core
        Share of a stage's corridor mass that defines its core (the fewest grid points holding it).
        (Default: 0.8)
    shared
        Overlap below which two corridors count as parted (divergence pseudotime).
        (Default: 0.5)
    min_mass
        Share of a corridor's mass a stage must hold for the corridor to be measured there (stage overlap,
        divergence); stages with less hold too few walks to describe a route.
        (Default: 0.02)

    Returns
    ----------
    dict with
    'stages': one row per lineage and stage: pseudotime range, mass, width (root-mean-square distance of the corridor
    from its centre at that stage, embedding units), centre coordinates, core grid points and the cells nearest to
    them, commitment (corridor-weighted mean probability of the lineage) and decision strength (its rise per unit
    pseudotime);
    'overlap': stage overlap of every pair of corridors (pairs x stages);
    'pairs': per pair, the overlap of the whole corridors (0 disjoint, 1 the same route) and the divergence
    pseudotime (start of the first stage from which the stage overlap stays below ``shared``; NaN if it never does);
    'decisions': per lineage, the decision zone, where the corridor crosses an even chance of that lineage (fate
    potential renormalised over the lineages between 0.4 and 0.6): its share of the corridor, its mean pseudotime
    (the decision pseudotime), its centre and the cells nearest to it;
    'tree': average-linkage clustering of the lineages on 1 - overlap ('linkage', 'labels', 'order', 'newick').
    """
    from scipy.cluster.hierarchy import leaves_list, linkage, to_tree
    names, C, H = _stored(gres, key_add)
    mapped = np.asarray(gres.grids['mapped_grids'])
    t = gres.grids['pseudotime'].loc[mapped].to_numpy(dtype=float)
    X = np.asarray(gres.grids['grids'], dtype=float)[mapped]
    cells = _cell_counts(gres)
    total = H.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        R = H / total[:, None]
    edges = np.linspace(np.nanmin(t), np.nanmax(t), bins + 1)
    stage = np.clip(np.digitize(t, edges[1:-1]), 0, bins - 1)
    rows = []
    for i, n in enumerate(names):
        c = C[:, i]
        whole = c.sum()
        for b in range(bins):
            m = stage == b
            w = c[m]
            mass = w.sum()
            row = dict(lineage=n, stage=b, t_from=edges[b], t_to=edges[b + 1], mass=mass / whole if whole else 0.)
            if mass > 1e-12 * max(whole, 1e-300):
                centre = (w[:, None] * X[m]).sum(0) / mass
                row.update(width=float(np.sqrt((w * ((X[m] - centre) ** 2).sum(1)).sum() / mass)),
                           core_points=int(_core(w, core).sum()), core_cells=float(cells[m][_core(w, core)].sum()),
                           commitment=float(np.nansum(w * R[m, i]) / mass),
                           **{f'centre_{k}': float(v) for k, v in enumerate(centre)})
            rows.append(row)
    stages = pd.DataFrame(rows)
    stages['decision_strength'] = np.nan
    for n in names:
        s = stages[(stages.lineage == n) & stages.commitment.notna()]
        if len(s) >= 2:
            mid = (s.t_from + s.t_to) / 2
            stages.loc[s.index, 'decision_strength'] = np.gradient(s.commitment.to_numpy(), mid.to_numpy())
    norm = C / np.where(C.sum(0) > 0, C.sum(0), 1)
    pair_rows, ov_rows = [], {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            label = f'{names[a]}-{names[b]}'
            per = []
            for s in range(bins):
                m = stage == s
                ma, mb = C[m, a].sum(), C[m, b].sum()
                if ma < min_mass * C[:, a].sum() or mb < min_mass * C[:, b].sum():
                    per.append(np.nan)
                    continue
                per.append(float(np.minimum(C[m, a] / ma, C[m, b] / mb).sum()))
            per = np.array(per)
            ov_rows[label] = per
            defined = np.flatnonzero(np.isfinite(per))
            div = np.nan
            for k in defined:
                later = per[defined[defined >= k]]
                if (later < shared).all():
                    div = float(edges[k])
                    break
            pair_rows.append(dict(pair=label, a=names[a], b=names[b],
                                  overlap=float(np.minimum(norm[:, a], norm[:, b]).sum()), divergence=div))
    pairs = pd.DataFrame(pair_rows)
    overlap = pd.DataFrame(ov_rows, index=pd.Index(range(bins), name='stage')).T
    dec = []
    for i, n in enumerate(names):
        c = C[:, i]
        zone = (R[:, i] >= .4) & (R[:, i] <= .6) & (c > 0)
        w = c[zone]
        row = dict(lineage=n, share=float(w.sum() / c.sum()) if c.sum() else 0., points=int(zone.sum()),
                   cells=float(cells[zone].sum()))
        if w.sum() > 0:
            centre = (w[:, None] * X[zone]).sum(0) / w.sum()
            row.update(pseudotime=float((w * t[zone]).sum() / w.sum()),
                       **{f'centre_{k}': float(v) for k, v in enumerate(centre)})
        dec.append(row)
    decisions = pd.DataFrame(dec)
    D = 1 - pairs.overlap.to_numpy()
    Z = linkage(np.clip(D, 0, None), method='average')

    def newick(node):
        if node.is_leaf():
            return names[node.id]
        return f'({newick(node.get_left())},{newick(node.get_right())})'
    tree = dict(linkage=Z, labels=names, order=[names[k] for k in leaves_list(Z)], newick=newick(to_tree(Z)) + ';')
    return dict(stages=stages, overlap=overlap, pairs=pairs, decisions=decisions, tree=tree)


def plot_corridors(gres, key_add='corridor', lineages=None, core=.8, colors=None, context='grid', start=True,
                   shared=False, spine=False, dims=(0, 1, 2), view=(22, -58), ax=None, s=None, legend=True):
    """
    Draw the corridors from the start region in two or three dimensions: for each lineage, the fewest grid points
    holding ``core`` of its corridor, coloured by lineage with opacity by corridor density, so that where corridors
    run together (the shared, multipotent states) their colours mix; the start region as black rings; with
    ``shared``, small dark marks on the grid points in the cores of two or more lineages; with ``spine``, a thin
    line through the corridor's centre at each pseudotime stage that holds at least 2% of it (a guide only: a
    corridor that splits has its centre between its arms).

    Parameters
    ----------
    gres
        Grids results after ``fate_corridors``.
    key_add
        Prefix given to ``fate_corridors``.
        (Default: 'corridor')
    lineages
        Lineages to draw (default all).
    core
        Share of each corridor drawn (its densest grid points).
        (Default: 0.8)
    colors
        Colours, one per lineage drawn (default the package palette).
    context
        'grid' (kept grid points in light grey), 'cells' (cells in light grey) or None.
        (Default: 'grid')
    start, shared, spine
        Draw the start region (default True), marks on the shared grid points and the centre lines (default False).
    dims, view, ax
        As in ``plot_trajectories``.
    s
        Marker size of the corridor points (default 3 in 3-D, 9 in 2-D).
    legend
        Legend to the right of the panel.
        (Default: True)

    Returns
    ----------
    The matplotlib axes.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from .PlotCore import DARK, GREY, _finish_view, _palette, _scatter, _view_setup
    names, C, H = _stored(gres, key_add)
    show = names if lineages is None else [str(n) for n in lineages]
    unknown = [n for n in show if n not in names]
    if unknown:
        raise ValueError(f'Lineage(s) {unknown} not in the corridors; stored: {names}')
    colors = _palette(len(show)) if colors is None else list(colors)
    if len(colors) != len(show):
        raise ValueError('colors needs one colour per lineage drawn.')
    made = ax is None
    G, dims, three, ax = _view_setup(gres, dims, ax)
    if made and legend:                           # room for the legend on the right
        ax.figure.set_size_inches(7.2, 4.6)
        ax.set_position([.0, .05, .66, .9] if three else [.08, .1, .54, .82])
    if three:
        ax.computed_zorder = False
    mapped = np.asarray(gres.grids['mapped_grids'])
    Pts = G[mapped][:, dims]
    s = (3 if three else 9) if s is None else s
    kw = {'depthshade': False} if three else {}
    if context == 'grid':
        _scatter(ax, Pts, GREY, None, s * .4, None, alpha=.35, zorder=1)
    elif context == 'cells':
        _scatter(ax, np.asarray(gres.embedding['embedding'], dtype=float)[:, dims], GREY, None, s * .3, None,
                 alpha=.35, zorder=1)
    elif context is not None:
        raise ValueError("context is 'grid', 'cells' or None.")
    cores, handles = {}, []
    for n in show:
        cores[n] = _core(C[:, names.index(n)], core)
    for n, col in sorted(zip(show, colors), key=lambda x: -cores[x[0]].sum()):     # largest corridors first
        c = C[:, names.index(n)]
        keep = cores[n]
        dens = c[keep] / c[keep].max()
        rgba = np.array([to_rgba(col)] * keep.sum())
        rgba[:, 3] = .1 + .55 * np.sqrt(dens)
        ax.scatter(*Pts[keep].T, s=s, c=rgba, lw=0, zorder=2, **kw)
    handles += [Line2D([], [], ls='', marker='o', ms=4, color=col, label=n) for n, col in zip(show, colors)]
    if shared and len(cores) > 1:
        both = np.sum(list(cores.values()), axis=0) >= 2
        if both.any():
            ax.scatter(*Pts[both].T, s=s * .45, color=DARK, lw=0, alpha=.6, zorder=3, **kw)
            handles.append(Line2D([], [], ls='', marker='o', ms=2.5, color=DARK, label='Shared by two or more'))
    if start:
        mu = np.asarray(gres.grids[f'{key_add}_start'], dtype=float)
        ax.scatter(*Pts[mu > 0].T, s=s * 1.3, facecolor='none', edgecolor='#222222', lw=.3, zorder=4, **kw)
        handles.append(Line2D([], [], ls='', marker='o', ms=4, mfc='none', mec='black', label='Start region'))
    if spine:
        prof = corridor_profile(gres, key_add, core=core)['stages']
        cols = [f'centre_{k}' for k in dims]
        for n, col in zip(show, colors):
            p = prof[(prof.lineage == n) & prof.width.notna() & (prof.mass >= .02)].sort_values('stage')
            Y = p[cols].to_numpy()
            if len(Y) < 2:
                continue
            ax.plot(*Y.T, color='white', lw=2.4, solid_capstyle='round', zorder=5)
            ax.plot(*Y.T, color=col, lw=1.2, solid_capstyle='round', zorder=6)
            d = Y[-1] - Y[-2]
            if np.linalg.norm(d) > 0:
                if three:
                    ax.quiver(*Y[-1], *d, color=col, linewidth=1, arrow_length_ratio=.8, zorder=7)
                else:
                    ax.annotate('', xy=Y[-1], xytext=Y[-1] - d,
                                arrowprops=dict(arrowstyle='-|>', color=col, lw=1, mutation_scale=7))
    _finish_view(ax, three, dims, view, Pts)
    if legend:
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02 if three else 1.04, 1), frameon=False,
                  borderaxespad=0)
    return ax

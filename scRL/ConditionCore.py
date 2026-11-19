"""Compare two conditions (for example control and treated cells) on one grid."""
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage, sparse, stats
from scipy.spatial import cKDTree

from .GridNDCore import attach_lattice, is_nd, lattice_from_scan


def _lattice(gres):
    """The grid itself when it has a neighbour table, else its n-D copy (2-D scan grids)."""
    return gres if is_nd(gres) else lattice_from_scan(gres)


def _steps(gres):
    """Grid step along each axis, origin and lattice shape, read from the stored lattice coordinates."""
    G = np.asarray(gres.grids['grids'])
    shape = tuple(gres.grids.get('shape', (gres.grids['n'],) * G.shape[1]))
    strides = [int(np.prod(shape[k + 1:])) for k in range(len(shape))]
    spacing = np.array([G[s, k] - G[0, k] for k, s in enumerate(strides)])
    return spacing, G[0], shape


def _cluster_labels(gres):
    """Cluster of every kept grid point, and the cluster order; one cluster 'all' without clusters."""
    M = len(gres.grids['mapped_grids'])
    if 'mapped_grids_clusters' not in gres.grids:
        return np.array(['all'] * M, dtype=object), ['all']
    labels = np.asarray(gres.grids['mapped_grids_clusters'])
    clusters = gres.embedding.get('clusters')
    order = list(clusters.cat.categories) if hasattr(clusters, 'cat') else sorted(set(labels.tolist()))
    present = set(labels.tolist())
    return labels, [c for c in order if c in present]


def _incidence(gres, X, radius):
    """Sparse matrix (kept grid points x cells): 1 where the cell lies within ``radius`` grid steps."""
    spacing, _, _ = _steps(gres)
    points = np.asarray(gres.grids['grids'])[np.asarray(gres.grids['mapped_grids'])] / spacing
    hits = cKDTree(np.asarray(X, dtype=float) / spacing).query_ball_point(points, radius)
    rows = np.repeat(np.arange(len(hits)), [len(h) for h in hits])
    cols = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits]) if len(rows) else np.zeros(0, int)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(points), len(X)))


def project_conditions(gres, conditions, radius=1.5):
    """
    Count the cells of each condition around every kept grid point.

    A cell is counted at a grid point when it lies within ``radius`` grid steps of it (coordinates divided by
    the grid step of each axis), so a cell can be counted at several neighbouring grid points.

    Parameters
    ----------
    gres
        Grids results.
    conditions
        One condition label per cell, in the order of the embedding.
    radius
        Counting radius in grid steps.
        (Default: 1.5)

    Returns
    ----------
    DataFrame of counts (index kept grid indices, one column per condition, names sorted). Also stored as
    ``gres.grids['condition_counts']``; the labels as ``gres.embedding['conditions']`` and the radius as
    ``gres.grids['condition_radius']``.
    """
    labels = np.asarray(conditions).astype(str)
    X = np.asarray(gres.embedding['embedding'])
    if len(labels) != len(X):
        raise ValueError(f'conditions has {len(labels)} labels but there are {len(X)} cells.')
    names = sorted(set(labels.tolist()))
    if len(names) < 2:
        raise ValueError('project_conditions needs at least two conditions.')
    A = _incidence(gres, X, radius)
    counts = pd.DataFrame({n: np.asarray(A @ (labels == n).astype(float)).astype(int) for n in names},
                          index=list(gres.grids['mapped_grids']))
    gres.embedding['conditions'] = labels
    gres.grids['condition_counts'] = counts
    gres.grids['condition_radius'] = radius
    return counts


def _conditions(gres, test, reference):
    if 'conditions' not in gres.embedding:
        raise ValueError('No conditions on the grid: call project_conditions(gres, conditions) first.')
    labels = gres.embedding['conditions']
    for name in (test, reference):
        if name not in set(labels.tolist()):
            raise ValueError(f'Condition {name!r} not found. Conditions present: {sorted(set(labels.tolist()))}')
    return labels


def _classes(k_test, k_ref, test, reference, min_cells):
    present_t, present_r = k_test >= min_cells, k_ref >= min_cells
    return np.select([present_t & present_r, present_r, present_t],
                     ['shared', f'{reference} only', f'{test} only'], 'too few cells').astype(object)


def _bh(p):
    """Benjamini-Hochberg q-values."""
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def compare_abundance(gres, test, reference, min_cells=3, alpha=.05, samples=None, n_shuffles=200, seed=0):
    """
    Where the cells of two conditions lie on the grid, and where their numbers differ.

    A condition is present at a grid point from ``min_cells`` counted cells; each grid point is classed as
    shared, held by one condition only, or too sparse to tell. Where both conditions together have at least
    ``2 * min_cells`` cells the numbers are compared: without ``samples`` by a two-sided binomial test of the
    test condition's share against its share of all cells (twice the smaller tail, capped at 1); with ``samples`` by a Welch t-test across the
    samples of the two conditions. Q-values follow Benjamini-Hochberg. Because sampling alone leaves some grid
    points with cells of one condition only, the labels of the two conditions are shuffled over their cells
    ``n_shuffles`` times (sizes kept) and the number of such points expected by chance is reported per cluster.

    Parameters
    ----------
    gres
        Grids results after ``project_conditions``.
    test, reference
        The two condition names.
    min_cells
        Cells needed for a condition to count as present at a grid point.
        (Default: 3)
    alpha
        Q-value threshold of a call.
        (Default: .05)
    samples
        One sample label per cell (each sample in one condition, at least two per condition): test across
        samples instead of across cells.
        (Default: None)
    n_shuffles
        Label shufflings for the numbers expected by chance.
        (Default: 200)
    seed
        Seed of the shuffles.
        (Default: 0)

    Returns
    ----------
    DataFrame over the kept grid points (index kept grid indices): ``n_test``, ``n_reference``, ``log2fc``,
    ``pvalue``, ``qvalue``, ``class``, ``call``. Stored as ``gres.grids['abundance']``, with the arrays
    ``abundance_log2fc`` (NaN where not tested), ``abundance_qvalue``, ``abundance_class``,
    ``abundance_call``, their colour dicts ``abundance_class_colors`` and ``abundance_call_colors``, the names
    ``abundance_test`` and ``abundance_reference``, and the per-cluster summary ``abundance_clusters``.
    """
    labels = _conditions(gres, test, reference)
    if 'condition_counts' not in gres.grids:
        raise ValueError('No condition counts on the grid: call project_conditions(gres, conditions) first.')
    counts = gres.grids['condition_counts']
    k_t, k_r = counts[test].to_numpy(), counts[reference].to_numpy()
    N_t, N_r = int((labels == test).sum()), int((labels == reference).sum())
    cls = _classes(k_t, k_r, test, reference, min_cells)
    log2fc = np.log2(((k_t + .5) / N_t) / ((k_r + .5) / N_r))
    tested = k_t + k_r >= 2 * min_cells
    pvalue = np.full(len(k_t), np.nan)
    if samples is None:
        n = (k_t + k_r)[tested]
        share = N_t / (N_t + N_r)
        lower = stats.binom.cdf(k_t[tested], n, share)
        upper = stats.binom.sf(k_t[tested] - 1, n, share)
        pvalue[tested] = np.minimum(1, 2 * np.minimum(lower, upper))
    else:
        pvalue[tested] = _sample_test(gres, labels, np.asarray(samples).astype(str), test, reference)[tested]
    qvalue = np.full(len(k_t), np.nan)
    qvalue[tested] = _bh(pvalue[tested])
    call = np.where(~tested, 'too few cells',
                    np.where(qvalue < alpha, np.where(log2fc < 0, 'depleted', 'enriched'), 'no difference'))
    call = call.astype(object)
    table = pd.DataFrame({'n_test': k_t, 'n_reference': k_r, 'log2fc': log2fc, 'pvalue': pvalue,
                          'qvalue': qvalue, 'class': cls, 'call': call}, index=counts.index)

    # what sampling alone gives: shuffle the two conditions' labels over their cells
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero((labels == test) | (labels == reference))
    A = _incidence(gres, np.asarray(gres.embedding['embedding'])[pool], gres.grids['condition_radius'])
    is_test = (labels[pool] == test).astype(float)
    clusters, order = _cluster_labels(gres)
    chance = {c: np.zeros(2) for c in order}
    for _ in range(n_shuffles):
        perm = rng.permutation(is_test)
        st = np.asarray(A @ perm)
        sr = np.asarray(A @ (1 - perm))
        shuffled = _classes(st, sr, test, reference, min_cells)
        for c in order:
            here = clusters == c
            chance[c] += [(shuffled[here] == f'{reference} only').sum(), (shuffled[here] == f'{test} only').sum()]
    rows = []
    for c in order:
        here = clusters == c
        rows.append({'points': int(here.sum()),
                     f'{reference} only': int((cls[here] == f'{reference} only').sum()),
                     f'{reference} only by chance': chance[c][0] / max(n_shuffles, 1),
                     f'{test} only': int((cls[here] == f'{test} only').sum()),
                     f'{test} only by chance': chance[c][1] / max(n_shuffles, 1),
                     'depleted': int((call[here] == 'depleted').sum()),
                     'enriched': int((call[here] == 'enriched').sum()),
                     'median_log2fc': float(np.median(log2fc[here & tested])) if (here & tested).any() else np.nan})
    gres.grids['abundance'] = table
    gres.grids['abundance_log2fc'] = np.where(tested, log2fc, np.nan)
    gres.grids['abundance_qvalue'] = qvalue
    gres.grids['abundance_class'] = cls
    gres.grids['abundance_call'] = call
    gres.grids['abundance_class_colors'] = {'shared': '#8c8c8c', f'{reference} only': '#2166ac',
                                            f'{test} only': '#b2182b', 'too few cells': '#e0e0e0'}
    gres.grids['abundance_call_colors'] = {'depleted': '#2166ac', 'enriched': '#b2182b',
                                           'no difference': '#8c8c8c', 'too few cells': '#e0e0e0'}
    gres.grids['abundance_test'], gres.grids['abundance_reference'] = test, reference
    gres.grids['abundance_clusters'] = pd.DataFrame(rows, index=order)
    return table


def _sample_test(gres, labels, samples, test, reference):
    """Welch t-test across samples of log2((count + .5) / (cells + 1)) at every kept grid point."""
    if len(samples) != len(labels):
        raise ValueError(f'samples has {len(samples)} labels but there are {len(labels)} cells.')
    groups = {}
    for name in (test, reference):
        ids = sorted(set(samples[labels == name].tolist()))
        for s in ids:
            if set(labels[samples == s].tolist()) != {name}:
                raise ValueError(f'Sample {s!r} holds cells of more than one condition.')
        if len(ids) < 2:
            raise ValueError(f'Condition {name!r} has {len(ids)} sample(s); the test across samples needs at '
                             f'least two per condition.')
        groups[name] = ids
    A = _incidence(gres, np.asarray(gres.embedding['embedding']), gres.grids['condition_radius'])
    y = {}
    for name, ids in groups.items():
        y[name] = np.column_stack([np.log2((np.asarray(A @ (samples == s).astype(float)) + .5)
                                           / ((samples == s).sum() + 1)) for s in ids])
    with np.errstate(invalid='ignore', divide='ignore'):
        p = stats.ttest_ind(y[test], y[reference], axis=1, equal_var=False).pvalue
    return np.where(np.isnan(p), 1.0, p)


def _grid_for_cells(gres, mask, radius=None):
    """The map of the cells in ``mask`` on the lattice of ``gres`` (see ``condition_grid``)."""
    from .GridCore import Grids_Results
    lattice = _lattice(gres)
    spacing, origin, shape = _steps(gres)
    d = len(shape)
    mask = np.asarray(mask, bool)
    X = np.asarray(gres.embedding['embedding'])[mask]
    if len(X) == 0:
        raise ValueError('There are no cells to build a grid from.')
    r = gres.grids.get('radius', 1.0) if radius is None else float(radius)
    r = max(r, np.sqrt(d) / 2)
    parent = np.asarray(gres.grids['mapped_grids'])
    coords = np.array(np.unravel_index(parent, shape)).T
    dist, _ = cKDTree((X - origin) / spacing).query(coords, k=1, distance_upper_bound=r + 1e-9)
    core = np.zeros(int(np.prod(shape)), bool)
    core[parent[dist <= r]] = True
    structure = ndimage.generate_binary_structure(d, d if d <= 4 else 1)
    kept_mask = ndimage.binary_dilation(core.reshape(shape), structure).ravel()
    in_parent = np.zeros(len(kept_mask), bool)
    in_parent[parent] = True
    kept_mask &= in_parent
    keep = kept_mask[parent]
    positions = np.flatnonzero(keep)

    new = Grids_Results()
    new.embedding['embedding'] = X
    for key in ('clusters', 'cluster_colors', 'conditions', 'X_latent'):
        if key in gres.embedding:
            value = gres.embedding[key]
            new.embedding[key] = (value[mask].reset_index(drop=True) if isinstance(value, pd.Series)
                                  else np.asarray(value)[mask])
    if 'clusters' in new.embedding and hasattr(gres.embedding['clusters'], 'cat'):
        new.embedding['clusters'] = pd.Series(pd.Categorical(
            np.asarray(new.embedding['clusters']), categories=gres.embedding['clusters'].cat.categories))
    new.grids['n'] = gres.grids['n']
    new.grids['grids'] = gres.grids['grids']
    new.grids['spacing'] = spacing
    new.grids['origin'] = origin
    new.grids['method'] = 'occupancy'
    new.grids['radius'] = r
    new.grids['mapped_grids'] = parent[positions]
    new.grids['masked_grids'] = np.flatnonzero(~kept_mask)
    new.grids['parent_positions'] = positions
    attach_lattice(new, shape, lattice.grids['moves'])
    if 'mapped_grids_clusters' in gres.grids and 'clusters' in new.embedding:
        own = np.asarray(new.embedding['clusters'])
        _, nn = cKDTree(X).query(np.asarray(gres.grids['grids'])[new.grids['mapped_grids']], k=1)
        new.grids['mapped_grids_clusters'] = own[nn]
        if 'cluster_colors' in gres.embedding:
            lut = dict(zip(np.asarray(gres.embedding['clusters']).tolist(), gres.embedding['cluster_colors']))
            new.grids['mapped_grids_colors'] = np.array([lut[c] for c in own[nn]])
    if 'pseudotime' in gres.grids:
        new.grids['pseudotime'] = pd.Series(gres.grids['pseudotime'].loc[parent[positions]].to_numpy(),
                                            index=parent[positions])
        if 'pseudotime_span' in gres.grids:
            new.grids['pseudotime_span'] = gres.grids['pseudotime_span']
    if 'proj' in gres.grids:
        new.grids['proj'] = gres.grids['proj'].iloc[positions].reset_index(drop=True)
    return new


def condition_grid(gres, condition, radius=None):
    """
    The map of one condition on the same lattice as the pooled grid.

    Keeps the lattice points within ``radius`` grid steps of that condition's cells plus one rim layer (the
    rule of the occupancy grid), restricted to the kept points of ``gres``. The lattice, grid step, moves and
    grid indices are those of ``gres``; boundary and travel points are rebuilt. Each kept grid point takes the
    cluster of its nearest cell of that condition (the rule of ``project_cluster``), so the map shows which
    lineages the condition's own cells occupy. Pseudotime and projected values are inherited, so that every
    map runs on one clock. No reward tables, start or end points are carried over: call ``d_rewards`` or
    ``c_rewards`` on the returned grid to train a learner there (``compare_values`` builds its own rewards, so
    that both maps are paid by the same rule).

    Parameters
    ----------
    gres
        Grids results after ``project_conditions``.
    condition
        The condition whose map is built.
    radius
        Distance in grid steps within which a cell keeps a grid point.
        (Default: None, the radius of ``gres``, 1.0 for a 2-D scan grid)

    Returns
    ----------
    New grids results; ``grids['parent_positions']`` gives the position of each kept point among the kept
    points of ``gres``.
    """
    labels = gres.embedding.get('conditions')
    if labels is None or condition not in set(np.asarray(labels).tolist()):
        present = [] if labels is None else sorted(set(np.asarray(labels).tolist()))
        raise ValueError(f'Condition {condition!r} not found. Conditions present: {present}')
    return _grid_for_cells(gres, np.asarray(labels) == condition, radius)


def _lineage_grid(gres, mask, ends, starts, mode, beta, lo, hi, radius=None):
    """
    The map of the cells in ``mask`` with the lineage reward built from its own labels: end points are the grid
    points labelled with ``ends``, paid by the time weight of their pseudotime scaled to [lo, hi].
    """
    from .EnvironmentNDCore import _table, _time_weight
    g = _grid_for_cells(gres, mask, radius)
    own = np.asarray(g.grids['mapped_grids_clusters'])
    t = g.grids['pseudotime'].loc[g.grids['mapped_grids']].to_numpy(dtype=float)
    s = np.clip((t - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros(len(t))
    bonus = np.where(np.isin(own, ends), _time_weight(s, beta, mode), 0.0)
    g.qlearning[f'd_{mode}_rewards'] = _table(g, bonus)
    g.qlearning['reward_key'] = '.'.join(map(str, ends))
    mapped = np.asarray(g.grids['mapped_grids'])
    g.grids['starts_cluster_grids'] = list(mapped[np.isin(own, starts)]) if starts else None
    g.grids['ends_cluster_grids'] = list(mapped[np.isin(own, ends)])
    return g


def _difference(gres, mask_test, mask_ref, rule, gamma, max_step, radius=None):
    """Exact values on two maps and their difference at the shared travel points, over the parent kept points."""
    from .ExactCore import exact_values
    M = len(gres.grids['mapped_grids'])
    values, travel = [], []
    for mask in (mask_test, mask_ref):
        g = _lineage_grid(gres, mask, *rule, radius=radius)
        V = exact_values(g, 'd', rule[2], gamma, max_step, key_add='_values')
        full = np.full(M, np.nan)
        full[g.grids['parent_positions']] = V
        values.append(full)
        is_travel = np.zeros(M, bool)
        is_travel[g.grids['parent_positions'][np.isin(g.grids['mapped_grids'], g.grids['mapped_travel'])]] = True
        travel.append(is_travel)
    shared = travel[0] & travel[1]
    return values[0], values[1], np.where(shared, values[0] - values[1], np.nan), travel[1]


def _z(effect, sd):
    """effect / sd; where sd is 0: 0 for a zero effect, NaN otherwise."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(sd > 0, effect / np.where(sd > 0, sd, 1), np.where((sd == 0) & (effect == 0), 0.0, np.nan))


def compare_values(gres, test, reference, ends, starts=None, mode='Decision', beta=1, gamma=.9, max_step=50,
                   n_shuffles=200, seed=0, key_add=None):
    """
    A lineage reward solved exactly on each condition's own map, compared where both maps have travel points.

    Each condition's map is labelled from its own cells (see ``condition_grid``). On every map the grid points
    labelled with ``ends`` are the end points; an end point pays the time weight of ``d_rewards`` ('Decision'
    or 'Contribution', ``beta``) of its pseudotime, scaled over the pseudotime range of the end points of the
    pooled grid, so that the reward of a grid point depends on its pseudotime in the same way on every map.
    The observed difference (test minus reference) is set against the differences obtained after shuffling
    the two conditions' labels over their cells (sizes kept): per grid point as an effect (observed minus the
    mean of the shuffles) and a z-score, per cluster as the difference of medians with a shuffle p-value.

    How to read it: the exact value of a grid point is the discounted distance to the best-paying part of the
    lineage (see ``exact_values``). 'Decision' asks whether the condition's cells still reach the entry of
    their own territory of the lineage, 'Contribution' whether they reach its far end. Where a condition has no
    cells of the lineage there is nothing to reach, and its value toward the lineage falls to zero.

    Parameters
    ----------
    gres
        Grids results after ``project_cluster``, ``align_pseudotime`` and ``project_conditions``.
    test, reference
        The two condition names.
    ends
        Cluster name(s) of the lineage's end.
    starts
        Cluster name(s) of the start: grid points labelled with them do not stop an episode.
        (Default: None)
    mode
        'Decision' or 'Contribution'.
        (Default: 'Decision')
    beta
        Decay of the time weight.
        (Default: 1)
    gamma, max_step
        As in ``exact_values``.
    n_shuffles
        Label shufflings for the null.
        (Default: 200)
    seed
        Seed of the shuffles.
        (Default: 0)
    key_add
        Prefix of the stored arrays.
        (Default: None, 'cond_' and the end names joined by '.')

    Returns
    ----------
    DataFrame per cluster of the pooled grid (one row 'all' without clusters): ``shared_points``, ``effect``,
    ``relative`` (the effect divided by the spread of the reference map's values, 99th minus 1st percentile
    over its travel points), ``null_sd``, ``z`` (0 where the effect and the null spread are both 0, NaN where
    only the null spread is 0), ``pvalue`` (from the shuffles in which the cluster has shared points) and
    ``shuffles`` (their number; effect, z and p-value are NaN when fewer than half of the shuffles entered).
    Stored on the grid (over the kept points, NaN where undefined):
    the values on each map ``<key>_<test>`` and ``<key>_<reference>``, ``<key>_effect`` and ``<key>_z``.
    """
    from .EnvironmentNDCore import _as_list, _time_weight
    from .GridNDCore import check_labels
    labels = _conditions(gres, test, reference)
    if 'mapped_grids_clusters' not in gres.grids:
        raise ValueError('compare_values needs cluster labels: call project_cluster first.')
    ends = _as_list(ends)
    starts = _as_list(starts) if starts else None
    _time_weight(0.0, beta, mode)
    check_labels(gres, ends + (starts or []))
    pooled_ends = np.isin(np.asarray(gres.grids['mapped_grids_clusters']), ends)
    t = gres.grids['pseudotime'].loc[np.asarray(gres.grids['mapped_grids'])].to_numpy(dtype=float)[pooled_ends]
    rule = (ends, starts, mode, beta, t.min(), t.max())
    mask_t, mask_r = labels == test, labels == reference
    V_t, V_r, d, ref_travel = _difference(gres, mask_t, mask_r, rule, gamma, max_step)
    spread = np.subtract(*np.percentile(V_r[ref_travel], [99, 1]))
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(mask_t | mask_r)
    n_test = int(mask_t.sum())
    null = np.full((n_shuffles, len(d)), np.nan)
    for i in range(n_shuffles):
        perm = rng.permutation(pool)
        st, sr = np.zeros(len(labels), bool), np.zeros(len(labels), bool)
        st[perm[:n_test]], sr[perm[n_test:]] = True, True
        null[i] = _difference(gres, st, sr, rule, gamma, max_step)[2]
    enough = (~np.isnan(null)).sum(0) >= n_shuffles / 2
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu = np.where(enough, np.nanmean(null, axis=0), np.nan)
        sd = np.where(enough, np.nanstd(null, axis=0), np.nan)
    effect = d - mu
    z = _z(effect, sd)
    clusters, order = _cluster_labels(gres)
    table = _cluster_table(d, null, clusters, order, spread)
    key = key_add or 'cond_' + '.'.join(map(str, ends))
    gres.grids[f'{key}_{test}'], gres.grids[f'{key}_{reference}'] = V_t, V_r
    gres.grids[f'{key}_effect'], gres.grids[f'{key}_z'] = effect, z
    return table


def _cluster_table(d, null, clusters, order, spread):
    """
    Per cluster: the median observed difference against the medians of the shuffles in which the cluster has
    shared points; NaN effect, z and p-value when fewer than half of the shuffles entered.
    """
    n_shuffles = len(null)
    rows = []
    for c in order:
        here = clusters == c
        observed = d[here & ~np.isnan(d)]
        medians = np.array([np.median(row[here & ~np.isnan(row)]) for row in null
                            if (here & ~np.isnan(row)).any()])
        row = {'shared_points': len(observed), 'effect': np.nan, 'relative': np.nan, 'null_sd': np.nan,
               'z': np.nan, 'pvalue': np.nan, 'shuffles': len(medians)}
        if len(observed) and len(medians) and len(medians) >= n_shuffles / 2:
            eff = np.median(observed) - medians.mean()
            null_sd = medians.std()
            count = (np.abs(medians - medians.mean()) >= abs(eff)).sum()
            row.update(effect=eff, relative=eff / spread if spread > 0 else np.nan, null_sd=null_sd,
                       z=float(_z(np.array(eff), np.array(null_sd))), pvalue=(1 + count) / (len(medians) + 1))
        rows.append(row)
    return pd.DataFrame(rows, index=order)


def compare_fates(gres, test, reference, key='gate_top', n_permutations=2000, n_boot=500, seed=0):
    """
    Share of each condition's cells in each category of a cell label (for example the heaviest lineage).

    Parameters
    ----------
    gres
        Grids results after ``project_conditions`` holding the cell label ``gres.embedding[key]``.
    test, reference
        The two condition names.
    key
        Cell label to count.
        (Default: 'gate_top', from ``fate_gate``)
    n_permutations
        Label permutations for the p-values.
        (Default: 2000)
    n_boot
        Bootstrap resamples of the cells for the 95% intervals.
        (Default: 500)
    seed
        Seed.
        (Default: 0)

    Returns
    ----------
    DataFrame indexed by category: ``reference``, ``reference_low``, ``reference_high``, ``test``,
    ``test_low``, ``test_high`` (shares with 95% bootstrap intervals), ``difference`` (test minus reference)
    and ``pvalue`` (two-sided, from permuting the condition labels).
    """
    labels = _conditions(gres, test, reference)
    if key not in gres.embedding:
        raise KeyError(f'{key!r} is not stored for the cells; available keys: {sorted(map(str, gres.embedding))}')
    cell_key = np.asarray(gres.embedding[key]).astype(str)
    pool = np.flatnonzero((labels == test) | (labels == reference))
    cats = sorted(set(cell_key[pool].tolist()))
    C = (cell_key[pool][:, None] == np.array(cats)[None, :]).astype(float)
    is_test = labels[pool] == test
    rng = np.random.default_rng(seed)

    def shares(t_mask):
        return C[t_mask].mean(0), C[~t_mask].mean(0)

    st, sr = shares(is_test)
    diff = st - sr
    boot_t = np.array([C[is_test][rng.integers(0, is_test.sum(), is_test.sum())].mean(0) for _ in range(n_boot)])
    boot_r = np.array([C[~is_test][rng.integers(0, (~is_test).sum(), (~is_test).sum())].mean(0)
                       for _ in range(n_boot)])
    count = np.zeros(len(cats))
    for _ in range(n_permutations):
        pt, pr = shares(rng.permutation(is_test))
        count += np.abs(pt - pr) >= np.abs(diff)
    return pd.DataFrame({'reference': sr, 'reference_low': np.percentile(boot_r, 2.5, axis=0),
                         'reference_high': np.percentile(boot_r, 97.5, axis=0),
                         'test': st, 'test_low': np.percentile(boot_t, 2.5, axis=0),
                         'test_high': np.percentile(boot_t, 97.5, axis=0),
                         'difference': diff, 'pvalue': (1 + count) / (n_permutations + 1)}, index=cats)

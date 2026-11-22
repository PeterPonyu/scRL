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


def compare_abundance(gres, test, reference, min_cells=3, alpha=.05, samples=None, n_shuffles=200, seed=0,
                      scale=None):
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
        samples instead of across cells, and, when clusters are projected, add to the per-cluster table the
        share of each sample's cells per cluster with a test across samples. Without samples every test is across cells and cannot tell a
        difference between conditions from a difference between two libraries; with one library per condition
        read the calls as a description of these two libraries. With samples the grid-point test is valid but
        has little power with a handful of libraries (three or four per condition can leave almost no grid
        point called after correction); the per-cluster shares, where each library contributes one number,
        are where differences show.
        (Default: None)
    n_shuffles
        Label shufflings for the numbers expected by chance.
        (Default: 200)
    seed
        Seed of the shuffles.
        (Default: 0)
    scale
        Scale of the per-cluster test across samples: 'log' compares log2((count + 0.5) / (cells + 1)) of each
        sample (count: its cells in the cluster; cells: all its cells), 'share' compares the shares. A
        difference of shares is dominated by the large clusters and by the samples with the most extreme
        composition; a ratio is not, so the log scale can find a change in one cluster that the share scale
        misses.
        None means 'log' here, as the category is always the cluster label.
        (Default: None)

    Returns
    ----------
    DataFrame over the kept grid points (index kept grid indices): ``n_test``, ``n_reference``, ``log2fc``,
    ``pvalue``, ``qvalue``, ``class``, ``call``. Stored as ``gres.grids['abundance']``, with the arrays
    ``abundance_log2fc`` (NaN where not tested), ``abundance_qvalue``, ``abundance_class``,
    ``abundance_call``, their colour dicts ``abundance_class_colors`` and ``abundance_call_colors``, the names
    ``abundance_test`` and ``abundance_reference``, and the per-cluster summary ``abundance_clusters`` (with
    ``samples`` and projected clusters also ``share_reference``, ``share_test``: the mean over samples of the
    share of a sample's cells in the cluster, where cells without a cluster label count in no cluster so a
    sample's shares can sum to less than 1; ``pvalue_samples`` from a Welch t-test across samples on the
    chosen ``scale``, 1 or 0 when the values vary in neither condition; ``qvalue_samples``, Benjamini-Hochberg
    over clusters; and with ``scale='log'`` ``log2fc_samples``, the mean over test samples minus the mean over
    reference samples of the log2 pseudo-counted shares).
    """
    labels = _conditions(gres, test, reference)
    scale = _resolve_scale('clusters', scale)
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
    summary = pd.DataFrame(rows, index=order)
    if samples is not None and 'clusters' in gres.embedding and 'mapped_grids_clusters' in gres.grids:
        per_sample, of = _sample_conditions(gres, samples, (test, reference))
        cell_clusters = _cell_clusters(gres, 'the per-sample cluster test')
        shares = {s: [(cell_clusters[per_sample == s] == str(c)).mean() for c in order] for s in of}
        a = np.array([shares[s] for s in of if of[s] == test])
        b = np.array([shares[s] for s in of if of[s] == reference])
        summary['share_reference'], summary['share_test'] = b.mean(0), a.mean(0)
        if scale == 'log':
            y = {s: _log_share(np.array([(cell_clusters[per_sample == s] == str(c)).sum() for c in order], float),
                               (per_sample == s).sum()) for s in of}
            a = np.array([y[s] for s in of if of[s] == test])
            b = np.array([y[s] for s in of if of[s] == reference])
            summary['log2fc_samples'] = a.mean(0) - b.mean(0)
        _, _, p = _welch(a, b)
        summary['pvalue_samples'] = p
        q = np.full(len(p), np.nan)
        ok = ~np.isnan(p)
        q[ok] = _bh(p[ok])
        summary['qvalue_samples'] = q
    gres.grids['abundance_clusters'] = summary
    return table


def _log_share(count, cells):
    """log2((count + 0.5) / (cells + 1)): a sample's share of cells on a log scale, with a fixed pseudo-count."""
    return np.log2((count + .5) / (cells + 1))


SCALE_RULE = ("scale='log' compares log2((count + 0.5) / (cells + 1)) per sample, 'share' compares the shares; "
              "None means 'log' for the cluster label (key 'clusters') and 'share' for any other key.")


def _resolve_scale(key, scale, stat='top', standardise=False):
    """The scale of the per-sample tests: an explicit ``scale`` wins, otherwise 'log' for the cluster label."""
    if scale is None:
        return 'log' if key == 'clusters' and stat == 'top' and not standardise else 'share'
    if scale not in ('log', 'share'):
        raise ValueError(f"scale must be 'log', 'share' or None, got {scale!r}. {SCALE_RULE}")
    if scale == 'log' and stat != 'top':
        raise ValueError("scale='log' needs stat='top' (counts of cells per category); use scale='share'.")
    if scale == 'log' and standardise:
        raise ValueError("scale='log' cannot be combined with standardise=True: standardising reweights shares to "
                         "one cluster composition, which has no log-scale form here, and for the cluster label it "
                         "leaves every unit with the pooled composition, so nothing remains to compare.")
    return scale


def _warn_zero_counts(cats, counts, cells, condition):
    """
    Warn when, on the log scale, a category has no cells in some samples while the samples of the conditions
    differ in size (median cells per sample more than 20% apart): its values then follow the size alone.
    """
    counts, cells, condition = np.asarray(counts, float), np.asarray(cells, float), np.asarray(condition)
    medians = [np.median(cells[condition == c]) for c in sorted(set(condition.tolist()))]
    if len(medians) < 2 or min(medians) <= 0 or max(medians) / min(medians) <= 1.25:
        return
    for i, c in enumerate(cats):
        if (counts[:, i] == 0).any():
            warnings.warn(f'category {c!r} has no cells in some samples while the samples of the two conditions '
                          f'differ in size; on the log scale its values then change with the sample size alone. '
                          f"Use scale='share'.", UserWarning, stacklevel=3)
            return


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
        y[name] = np.column_stack([_log_share(np.asarray(A @ (samples == s).astype(float)), (samples == s).sum())
                                   for s in ids])
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
    shared points; effect, relative effect, null spread, z and p-value are NaN when fewer than half of the
    shuffles entered.
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


def _sample_conditions(gres, samples, names=None):
    """The condition of each sample (a sample must hold the cells of one condition only)."""
    labels = np.asarray(gres.embedding['conditions'])
    samples = np.asarray(samples).astype(str)
    if len(samples) != len(labels):
        raise ValueError(f'samples has {len(samples)} labels but there are {len(labels)} cells.')
    of = {}
    for s in sorted(set(samples.tolist())):
        held = sorted(set(labels[samples == s].tolist()))
        if len(held) > 1:
            raise ValueError(f'Sample {s!r} holds cells of more than one condition ({held}).')
        of[s] = held[0]
    if names is not None:
        for name in names:
            n = sum(c == name for c in of.values())
            if n < 2:
                raise ValueError(f'Condition {name!r} has {n} sample(s); a test across samples needs at least two '
                                 f'per condition.')
    return samples, of


def _welch(a, b):
    """
    Welch t statistic, degrees of freedom and two-sided p-value of mean(a) - mean(b), per column. When neither
    group varies the p-value is 1 for equal means and 0 for different ones (t 0 or plus or minus infinity, df NaN).
    """
    ma, mb = a.mean(0), b.mean(0)
    va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
    se2 = va + vb
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (ma - mb) / np.sqrt(se2)
        df = se2 ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        p = 2 * stats.t.sf(np.abs(t), df)
    still = se2 == 0
    same = ma == mb
    t = np.where(still, np.where(same, 0.0, np.sign(ma - mb) * np.inf), t)
    p = np.where(still, np.where(same, 1.0, 0.0), p)
    return t, df, p


def _selection(gres, within):
    """Boolean mask over cells: all cells, a given mask, or the cells of the given clusters."""
    n = len(gres.embedding['embedding'])
    if within is None:
        return np.ones(n, bool)
    if isinstance(within, (str, list, tuple)) or (isinstance(within, np.ndarray) and within.dtype.kind in 'OUS'):
        names = [within] if isinstance(within, str) else list(within)
        clusters = _cell_clusters(gres, 'within= by cluster names')
        unknown = [c for c in names if str(c) not in set(clusters.tolist())]
        if unknown:
            raise ValueError(f'Cluster(s) {unknown} not found. Clusters present: {sorted(set(clusters.tolist()))}')
        return np.isin(clusters, [str(c) for c in names])
    mask = np.asarray(within)
    if mask.dtype != bool or len(mask) != n:
        raise ValueError(f'within must be None, cluster names, or a boolean mask over the {n} cells.')
    return mask


def _missing_key(gres, key):
    hint = ''
    if key == 'fate_top':
        hint = " Run fate_probabilities first, or pass key='gate_top' to use the gate's ranking."
    return f'{key!r} is not stored for the cells.{hint} Available keys: {sorted(map(str, gres.embedding))}'


def _statistic(gres, key, stat, cells):
    """
    Per-cell values whose mean over a group of cells is the statistic, for the given cells, and the categories.
    'top': one-hot of the categories of ``gres.embedding[key]``; 'mean': the columns ``key + '_' + name``.
    """
    if stat == 'top':
        if key not in gres.embedding:
            raise KeyError(_missing_key(gres, key))
        cell_key = np.asarray(gres.embedding[key]).astype(str)[cells]
        cats = sorted(set(cell_key.tolist()))
        return (cell_key[:, None] == np.array(cats)[None, :]).astype(float), cats
    if stat != 'mean':
        raise ValueError(f"stat must be 'top' or 'mean', got {stat!r}")
    if str(key).endswith('_top'):
        raise ValueError(f"stat='mean' needs the walk shares, for example key='fate' (key_add of "
                         f"fate_probabilities), not a heaviest-lineage label such as {key!r}.")
    table = gres.grids.get(key)
    names = [str(c) for c in table.columns] if isinstance(table, pd.DataFrame) else []
    names = [n for n in names if f'{key}_{n}' in gres.embedding]
    if not names:
        raise ValueError(f"stat='mean' needs the per-lineage columns '{key}_<name>' written by fate_probabilities or "
                         f"fate_gate (key_add={key!r}); none were found.")
    V = np.column_stack([np.asarray(gres.embedding[f'{key}_{n}'], dtype=float)[cells] for n in names])
    return V, names


def _group_mean(V):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(V, axis=0)


def _composition(V, clusters, units, kept):
    """
    Per unit, over the kept clusters (present in every unit): cluster weights (shares of the unit's cells in the
    kept clusters), within-cluster statistics and the statistic standardised to the pooled composition.
    """
    pooled = np.array([(clusters == c).mean() for c in kept])
    pooled = pooled / pooled.sum()
    out = {}
    for u, cells in units.items():
        in_kept = cells & np.isin(clusters, kept)
        w = np.array([(clusters[in_kept] == c).mean() for c in kept])
        stat_c = np.array([_group_mean(V[cells & (clusters == c)]) for c in kept])
        out[u] = (w, stat_c, pooled @ stat_c)
    return out


def _kept_clusters(clusters, units):
    """Clusters present in every unit, and the dropped ones."""
    present = [set(clusters[cells].tolist()) for cells in units.values()]
    kept = sorted(set.intersection(*present)) if present else []
    dropped = sorted(set(clusters.tolist()) - set(kept))
    if not kept:
        raise ValueError('No cluster holds cells in every unit; standardise=True needs at least one such cluster.')
    return kept, dropped


def _check_selected(units, what):
    for u, cells in units.items():
        if not cells.any():
            raise ValueError(f'The selection holds no cells of {what} {u!r}.')


def shares_by_sample(gres, samples, key='fate_top', stat='top', within=None, standardise=False, scale=None):
    """
    Share of each sample's cells in each category of a cell label (for example the heaviest lineage), or the
    mean of per-lineage values over each sample's cells.

    With replicate samples (libraries) the sample is the unit of a comparison: each sample contributes one
    number per category, so a difference between two samples is not mistaken for a difference between
    conditions.

    Parameters
    ----------
    gres
        Grids results after ``project_conditions``.
    samples
        One sample label per cell; a sample holds the cells of one condition only.
    key
        With ``stat='top'`` the cell label to count; with ``stat='mean'`` the prefix of the per-lineage columns
        written by ``fate_probabilities`` or ``fate_gate`` (for example 'fate').
        (Default: 'fate_top', the heaviest lineage from ``fate_probabilities``; ``'gate_top'`` uses the ranking
        of ``fate_gate``, which is not a share)
    stat
        'top': share of cells in each category; 'mean': mean of each per-lineage column over the cells (cells
        with a missing value are left out of that column's mean).
        (Default: 'top')
    within
        Restrict to these cells: a boolean mask over cells or a list of cluster names.
        (Default: None, all cells)
    standardise
        Also give, per category, the statistic reweighted to the pooled cluster composition of the selected
        cells (columns ``<category>_std``), over the clusters present in every sample; the clusters left out
        are listed in ``table.attrs['dropped_clusters']``. A cluster whose statistic is missing for every cell
        of a sample makes that sample's standardised value NaN.
        (Default: False)
    scale
        Scale of the tests across samples for ``stat='top'``: 'log' compares log2((count + 0.5) / (cells + 1))
        of each sample (count: its cells in the category; cells: its cells in the comparison), 'share' compares
        the shares. None means 'log' when ``key`` is the cluster label ('clusters') and 'share' for any other
        key; an explicit value always wins. Only the entry named 'clusters' counts as the cluster label (the
        name is compared, not the values); for cluster labels stored under another name pass ``scale='log'``.
        A difference of cluster shares is dominated by the large clusters and by the samples with the most
        extreme composition; a ratio is not, so the log scale can find a change in one cluster that the share
        scale misses. The log scale is unsafe for a category with no cells in some samples when the samples
        of the conditions differ in size (as ``within=`` can make them): its pseudo-counted log share then
        changes with the sample size alone. That case gives a warning, also under the default for
        ``key='clusters'``; lineage labels keep the share scale for this reason. Differences and means stay in
        share units on either scale. With ``standardise=True`` None means 'share'; an explicit 'log' is
        rejected there, as standardising reweights shares and has no log-scale form. 'log' needs
        ``stat='top'``.
        (Default: None)

    Returns
    ----------
    DataFrame, one row per sample: ``condition`` and the statistic for each category (for ``stat='top'`` the
    shares of a row sum to 1); with ``scale='log'`` also ``<category>_log2``, the values the log-scale tests
    compare.
    """
    if 'conditions' not in gres.embedding:
        raise ValueError('No conditions on the grid: call project_conditions(gres, conditions) first.')
    samples, of = _sample_conditions(gres, samples)
    scale = _resolve_scale(key, scale, stat, standardise)
    selected = _selection(gres, within)
    cells = np.flatnonzero(selected)
    V, cats = _statistic(gres, key, stat, cells)
    if 'condition' in cats:
        raise ValueError(f"gres.embedding[{key!r}] has a category named 'condition', which is the name of the "
                         f"column holding each sample's condition; rename that category.")
    added = ['_std'] if standardise else ['_log2'] if scale == 'log' else []
    clash = [f'{c}{x}' for c in cats for x in added if f'{c}{x}' in cats]
    if clash:
        raise ValueError(f'gres.embedding[{key!r}] has categories {clash}, which are also the names of added '
                         f'columns; rename those categories.')
    units = {u: samples[cells] == u for u in of}
    _check_selected(units, 'sample')
    rows = {u: _group_mean(V[m]) for u, m in units.items()}
    table = pd.DataFrame.from_dict(rows, orient='index', columns=cats)
    if scale == 'log':
        _warn_zero_counts(cats, [V[units[u]].sum(0) for u in table.index], [units[u].sum() for u in table.index],
                          [of[u] for u in table.index])
        for i, c in enumerate(cats):
            table[f'{c}_log2'] = [_log_share(V[units[u], i].sum(), units[u].sum()) for u in table.index]
    if standardise:
        clusters = _cell_clusters(gres, 'standardise=True')[cells]
        kept, dropped = _kept_clusters(clusters, units)
        parts = _composition(V, clusters, units, kept)
        for i, c in enumerate(cats):
            table[f'{c}_std'] = [parts[u][2][i] for u in table.index]
        table.attrs['dropped_clusters'] = dropped
    table.insert(0, 'condition', [of[u] for u in table.index])
    return table


def _cell_clusters(gres, what):
    """Cluster label of every cell as text; a missing label is an error rather than a cluster named 'nan'."""
    if 'clusters' not in gres.embedding:
        raise ValueError(f'{what} needs projected clusters (project_cluster).')
    raw = gres.embedding['clusters']
    missing = np.asarray(pd.isna(raw))
    if missing.any():
        raise ValueError(f'{int(missing.sum())} cells have no cluster label; give every cell a cluster for {what}.')
    return np.asarray(raw).astype(str)


def _blocks(gres, samples, of, pairs, test, reference):
    """Block of each sample (from a label per cell or a mapping sample -> block); one sample per condition each."""
    if isinstance(pairs, dict):
        block = {str(k): v for k, v in pairs.items()}
    else:
        pairs = np.asarray(pairs).astype(str)
        if len(pairs) != len(samples):
            raise ValueError(f'pairs has {len(pairs)} labels but there are {len(samples)} cells.')
        block = {}
        for u in of:
            held = sorted(set(pairs[samples == u].tolist()))
            if len(held) != 1:
                raise ValueError(f'Sample {u!r} lies in more than one block ({held}).')
            block[u] = held[0]
    members = {}
    for u, c in of.items():
        if c not in (test, reference):
            continue
        if u not in block:
            raise ValueError(f'Sample {u!r} has no block.')
        members.setdefault(block[u], {}).setdefault(c, []).append(u)
    for b, by in members.items():
        if sorted(by) != sorted([test, reference]) or any(len(v) != 1 for v in by.values()):
            raise ValueError(f'Block {b!r} must hold exactly one sample of each condition; it holds {by}.')
    return [(by[test][0], by[reference][0]) for _, by in sorted(members.items())]


def compare_fates(gres, test, reference, key='fate_top', samples=None, n_permutations=2000, n_boot=500, seed=0,
                  stat='top', within=None, standardise=False, pairs=None, scale=None):
    """
    Compare a fate statistic between two conditions: the share of cells in each category of a cell label (for
    example the heaviest lineage), or the mean of per-lineage values (for example the lineage shares of
    ``fate_probabilities``).

    Without ``samples`` the comparison is across cells: it cannot tell a difference between conditions from a
    difference between two libraries, so with one library per condition read it as a description of these two
    libraries. With ``samples`` (at least two per condition) each sample contributes one number per category and
    the numbers are also compared across samples (Welch t-test); with a handful of libraries this is where a
    difference between conditions can be shown. ``pairs`` adds a paired test when the samples form blocks.

    A difference in lineage shares between conditions is first of all a difference in which cell states are
    present. ``standardise=True`` reweights each unit to one cluster composition and splits the difference into
    a composition part and a within-cluster part; ``within=`` restricts the comparison to chosen cells (for
    example the uncommitted clusters).

    Parameters
    ----------
    gres
        Grids results after ``project_conditions``.
    test, reference
        The two condition names.
    key
        With ``stat='top'`` the cell label to count; with ``stat='mean'`` the prefix of the per-lineage columns
        written by ``fate_probabilities`` or ``fate_gate``.
        (Default: 'fate_top', the heaviest lineage from ``fate_probabilities``; ``'gate_top'`` uses the ranking
        of ``fate_gate``, which is not a share)
    samples
        One sample label per cell (each sample in one condition, at least two per condition).
        (Default: None)
    n_permutations
        Label permutations for the p-values.
        (Default: 2000)
    n_boot
        Bootstrap resamples of the cells for the 95% intervals.
        (Default: 500)
    seed
        Seed.
        (Default: 0)
    stat
        'top': share of cells in each category; 'mean': mean of each per-lineage column over the cells (cells
        with a missing value are left out of that column's mean). ``stat='mean'`` needs the walk shares, so
        pass ``key='fate'`` (the ``key_add`` of ``fate_probabilities``), not a ``_top`` label. The permutations
        and the bootstrap act on the statistic.
        (Default: 'top')
    within
        Compare only these cells: a boolean mask over cells or a list of cluster names.
        (Default: None, all cells)
    standardise
        Add the statistic standardised to the pooled cluster composition of the selected cells and split the
        difference over the same clusters into a composition part and a within-cluster part (needs projected
        clusters).
        (Default: False)
    pairs
        One block label per cell, or a dict sample -> block (needs ``samples``); each block holds exactly one
        sample of each condition. Adds a paired t-test across blocks.
        (Default: None)
    scale
        Scale of the tests across samples for ``stat='top'``: 'log' compares log2((count + 0.5) / (cells + 1))
        of each sample (count: its cells in the category; cells: its cells in the comparison), 'share' compares
        the shares. None means 'log' when ``key`` is the cluster label ('clusters') and 'share' for any other
        key; an explicit value always wins. Only the entry named 'clusters' counts as the cluster label (the
        name is compared, not the values); for cluster labels stored under another name pass ``scale='log'``.
        A difference of cluster shares is dominated by the large clusters and by the samples with the most
        extreme composition; a ratio is not, so the log scale can find a change in one cluster that the share
        scale misses. The log scale is unsafe for a category with no cells in some samples when the samples
        of the conditions differ in size (as ``within=`` can make them): its pseudo-counted log share then
        changes with the sample size alone. That case gives a warning, also under the default for
        ``key='clusters'``; lineage labels keep the share scale for this reason. Differences and means stay in
        share units on either scale. With ``standardise=True`` None means 'share'; an explicit 'log' is
        rejected there, as standardising reweights shares and has no log-scale form. 'log' needs
        ``stat='top'``.
        (Default: None)

    Returns
    ----------
    DataFrame indexed by category: ``reference``, ``reference_low``, ``reference_high``, ``test``,
    ``test_low``, ``test_high`` (statistic with 95% bootstrap intervals), ``difference`` (test minus reference)
    and ``pvalue`` (two-sided, from permuting the condition labels). With ``samples`` also
    ``reference_mean`` and ``test_mean`` (means over samples of the shares), ``difference_samples`` (in share
    units), ``t``, ``df`` and ``pvalue_samples`` (Welch t-test across samples on the chosen ``scale``; when the
    values vary in neither condition the p-value is 1 for equal and 0 for different means), ``n_reference``,
    ``n_test`` (numbers of samples) and, with ``scale='log'``, ``log2fc_samples`` (mean over test samples minus
    mean over reference samples of the log2 pseudo-counted shares). With
    ``standardise``: ``reference_std``, ``test_std``, ``difference_std`` (each unit, a sample when ``samples``
    is given and otherwise a condition, reweighted to the pooled cluster composition over the kept clusters,
    those present in every unit; the others are listed in ``table.attrs['dropped_clusters']``),
    ``difference_kept``, ``composition_part`` and ``within_part``. All of these use the kept clusters only.
    For each condition, w are the shares of its cells in the kept clusters (with ``samples``, the mean over its
    samples of each sample's shares) and s the within-cluster statistics (with ``samples``, the weight-weighted
    mean over its samples, sum of w s over sum of w, not the plain mean of per-sample values).
    ``difference_kept`` is the test minus reference difference of sum w s, the statistic over the cells of the
    kept clusters (with ``samples``, the mean over samples). ``composition_part`` is the sum of the weight
    difference times the mean statistic and ``within_part`` the sum of the mean weight times the statistic
    difference; they add up exactly to ``difference_kept``, which equals ``difference`` (or with ``samples``
    ``difference_samples``) when no cluster is dropped and the statistic has no missing values. A cluster whose
    statistic is missing for every cell of a unit makes the standardised values and the parts NaN. With
    ``samples`` also ``pvalue_samples_std``. With ``pairs``: ``pvalue_paired`` (paired t-test across blocks on
    the chosen ``scale``) and, with ``standardise``, ``pvalue_paired_std``.
    """
    labels = _conditions(gres, test, reference)
    if stat == 'top' and key not in gres.embedding:
        raise KeyError(_missing_key(gres, key))
    if pairs is not None and samples is None:
        raise ValueError('pairs needs samples.')
    scale = _resolve_scale(key, scale, stat, standardise)
    selected = _selection(gres, within)
    pool = np.flatnonzero(((labels == test) | (labels == reference)) & selected)
    C, cats = _statistic(gres, key, stat, pool)
    is_test = labels[pool] == test
    _check_selected({test: is_test, reference: ~is_test}, 'condition')
    rng = np.random.default_rng(seed)
    mean = (lambda M: M.mean(0)) if stat == 'top' else _group_mean

    def shares(t_mask):
        return mean(C[t_mask]), mean(C[~t_mask])

    st, sr = shares(is_test)
    diff = st - sr
    boot_t = np.array([mean(C[is_test][rng.integers(0, is_test.sum(), is_test.sum())]) for _ in range(n_boot)])
    boot_r = np.array([mean(C[~is_test][rng.integers(0, (~is_test).sum(), (~is_test).sum())])
                       for _ in range(n_boot)])
    count = np.zeros(len(cats))
    for _ in range(n_permutations):
        pt, pr = shares(rng.permutation(is_test))
        count += np.abs(pt - pr) >= np.abs(diff)
    table = pd.DataFrame({'reference': sr, 'reference_low': np.percentile(boot_r, 2.5, axis=0),
                          'reference_high': np.percentile(boot_r, 97.5, axis=0),
                          'test': st, 'test_low': np.percentile(boot_t, 2.5, axis=0),
                          'test_high': np.percentile(boot_t, 97.5, axis=0),
                          'difference': diff, 'pvalue': (1 + count) / (n_permutations + 1)}, index=cats)
    per_sample = None
    if samples is not None:
        sample_of, of = _sample_conditions(gres, samples, (test, reference))
        units = {u: sample_of[pool] == u for u, c in of.items() if c in (test, reference)}
        _check_selected(units, 'sample')
        per_sample = {u: mean(C[m]) for u, m in units.items()}
        a = np.array([per_sample[u] for u in units if of[u] == test])
        b = np.array([per_sample[u] for u in units if of[u] == reference])
        tested = per_sample
        if scale == 'log':
            _warn_zero_counts(cats, [C[m].sum(0) for m in units.values()], [m.sum() for m in units.values()],
                              [of[u] for u in units])
            tested = {u: _log_share(C[m].sum(0), m.sum()) for u, m in units.items()}
        ya = np.array([tested[u] for u in units if of[u] == test])
        yb = np.array([tested[u] for u in units if of[u] == reference])
        t, df, p = _welch(ya, yb)
        table['reference_mean'], table['test_mean'] = b.mean(0), a.mean(0)
        table['difference_samples'], table['t'], table['df'], table['pvalue_samples'] = a.mean(0) - b.mean(0), t, df, p
        table['n_reference'], table['n_test'] = len(b), len(a)
        if scale == 'log':
            table['log2fc_samples'] = ya.mean(0) - yb.mean(0)
        if pairs is not None:
            matched = _blocks(gres, sample_of, of, pairs, test, reference)
            table['pvalue_paired'] = _paired(np.array([tested[x] for x, _ in matched]),
                                             np.array([tested[y] for _, y in matched]))
    if standardise:
        clusters = _cell_clusters(gres, 'standardise=True')[pool]
        if samples is None:
            units = {test: is_test, reference: ~is_test}
        kept, dropped = _kept_clusters(clusters, units)
        parts = _composition(C, clusters, units, kept)
        side = {u: (test if (of[u] if samples is not None else u) == test else reference) for u in units}

        def condition_terms(name):
            members = [u for u in units if side[u] == name]
            w = np.mean([parts[u][0] for u in members], axis=0)
            weighted = np.sum([parts[u][0][:, None] * parts[u][1] for u in members], axis=0)
            total = np.sum([parts[u][0] for u in members], axis=0)[:, None]
            s_mean = np.where(total > 0, weighted / np.where(total > 0, total, 1), 0.0)
            return w, s_mean, np.mean([parts[u][2] for u in members], axis=0)

        w_t, s_t, std_t = condition_terms(test)
        w_r, s_r, std_r = condition_terms(reference)
        table['reference_std'], table['test_std'], table['difference_std'] = std_r, std_t, std_t - std_r
        table['difference_kept'] = w_t @ s_t - w_r @ s_r
        table['composition_part'] = (w_t - w_r) @ ((s_t + s_r) / 2)
        table['within_part'] = ((w_t + w_r) / 2) @ (s_t - s_r)
        if samples is not None:
            a = np.array([parts[u][2] for u in units if side[u] == test])
            b = np.array([parts[u][2] for u in units if side[u] == reference])
            table['pvalue_samples_std'] = _welch(a, b)[2]
            if pairs is not None:
                table['pvalue_paired_std'] = _paired(np.array([parts[x][2] for x, _ in matched]),
                                                     np.array([parts[y][2] for _, y in matched]))
        table.attrs['dropped_clusters'] = dropped
    return table


def _paired(a, b):
    """Two-sided paired t-test of a - b across blocks, per column."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return stats.ttest_rel(a, b, axis=0).pvalue


def design_effect(values, samples, alpha=0.05, scale='rank'):
    """
    How far a test that takes cells as units over-calls, given how cells cluster within samples.

    Cells from one animal (donor, library) resemble each other more than they resemble cells from another, so a
    test that counts every cell as independent rejects too often when groups of animals are compared. The
    intraclass correlation measures that resemblance (one-way analysis of variance, unequal sample sizes); the
    design effect ``1 + (m - 1) * ICC`` is the factor by which the variance of a mean over m cells per sample is
    inflated, and the predicted false-call rate is the share of rejections expected from a two-sided cell-level
    test at level ``alpha`` when the conditions do not differ, if the variance of its statistic is inflated by
    the design effect and whole samples are assigned to the conditions at random. It is an estimate under that
    approximation, not a guarantee: it is less reliable with few samples or very unequal sample sizes. The
    sample-level tests of this module (``compare_fates(..., samples=...)``) are the remedy: they take the
    sample as the unit.

    Parameters
    ----------
    values
        One number per cell (for example a fate probability or a pseudotime); array-like or pandas Series.
        Cells with a missing value are dropped together with their sample label. With ``scale='raw'`` the
        remaining values must be finite.
    samples
        One label per cell naming its replicate (animal, donor, library). Cells with a missing label are
        dropped.
    alpha
        Level of the cell-level test whose false-call rate is predicted; between 0 and 1, exclusive.
        (Default: 0.05)
    scale
        'rank': replace the values by their average ranks over all kept cells before anything else; 'raw':
        use the values as they are. Ranks make the result independent of any monotone transform of the values.
        (Default: 'rank')

    Returns
    ----------
    A plain ``dict`` (not a table, unlike the comparison functions of this module) with ``n_cells`` and
    ``n_samples`` kept, ``cells_per_sample`` (m, the mean number of kept cells per sample), ``icc`` (as
    estimated, possibly negative), ``design_effect`` (1 when the estimate is not positive),
    ``predicted_false_call_rate``, ``effective_cells_per_sample`` (m divided by the design effect), ``alpha``,
    ``scale`` and ``n_dropped`` (cells left out for a missing value or a missing sample label).
    """
    if scale not in ('rank', 'raw'):
        raise ValueError(f"scale must be 'rank' or 'raw', not {scale!r}.")
    if not 0 < alpha < 1:
        raise ValueError(f'alpha must lie between 0 and 1 (exclusive), not {alpha!r}.')
    x = np.asarray(values, dtype=float).ravel()
    labels = np.asarray(samples, dtype=object).ravel()
    if len(x) != len(labels):
        raise ValueError(f'values has {len(x)} entries but samples has {len(labels)} labels; one label per cell is needed.')
    keep = ~np.isnan(x) & ~np.asarray(pd.isna(labels), dtype=bool)
    n_dropped = int((~keep).sum())
    x, labels = x[keep], labels[keep]
    if scale == 'raw' and not np.isfinite(x).all():
        raise ValueError("values must be finite or missing when scale='raw'; use scale='rank' for unbounded values.")
    N = len(x)
    if N < 2:
        raise ValueError(f'design_effect needs at least two cells with a value; {N} left after dropping missing values.')
    names, codes = np.unique(labels.astype(str), return_inverse=True)
    k = len(names)
    if k < 2:
        raise ValueError('design_effect needs cells from at least two samples.')
    if N - k == 0:
        raise ValueError('design_effect needs more than one cell in at least one sample; every sample has one cell.')
    if scale == 'rank':
        x = stats.rankdata(x, method='average')
    sizes = np.bincount(codes).astype(float)
    means = np.bincount(codes, weights=x) / sizes
    msb = float((sizes * (means - x.mean()) ** 2).sum() / (k - 1))
    msw = float(((x - means[codes]) ** 2).sum() / (N - k))
    n0 = (N - (sizes ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    icc = float((msb - msw) / denom) if denom > 0 else 0.0
    m = N / k
    deff = 1 + (m - 1) * max(icc, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    return {'n_cells': int(N), 'n_samples': int(k), 'cells_per_sample': float(m), 'icc': icc,
            'design_effect': float(deff),
            'predicted_false_call_rate': float(2 * (1 - stats.norm.cdf(z / np.sqrt(deff)))),
            'effective_cells_per_sample': float(m / deff), 'alpha': float(alpha), 'scale': scale,
            'n_dropped': n_dropped}

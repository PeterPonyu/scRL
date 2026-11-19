"""Regular grids in two to five dimensions.

A grid is a regular lattice described by a neighbour table: for every kept grid point and every move, the
position of the grid point reached by that move, or -1 when the move leaves the kept grid. Grid points are
indexed in C order (``np.ravel_multi_index``), which in 2-D equals the index ``i + j * n`` of the 2-D scan grid.
"""
import copy
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

ACTIONS_2D = ['R', 'RT', 'T', 'LT', 'L', 'LB', 'B', 'RB']
OFFSETS_2D = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
DEFAULT_N = {2: 50, 3: 30, 4: 16, 5: 12}
DEFAULT_RADIUS = {2: 1.0, 3: 1.0, 4: 1.5, 5: 2.0}
MAX_LATTICE_POINTS = 5_000_000


def make_offsets(ndim, moves):
    """
    Lattice offsets of the moves an agent can make.

    Parameters
    ----------
    ndim
        Number of grid dimensions.
    moves
        'axis' (one step along a single axis, 2 * ndim moves) or
        'full' (every combination of -1, 0, +1 except standing still; the 8 moves of the 2-D scan grid in 2-D).

    Returns
    ----------
    Offsets (moves x ndim, int64) and the move names.
    """
    if moves == 'full':
        if ndim == 2:
            return np.array(OFFSETS_2D, dtype=np.int64), list(ACTIONS_2D)
        offs = [o for o in itertools.product((-1, 0, 1), repeat=ndim) if any(o)]
        names = [''.join('-0+'[v + 1] for v in o) for o in offs]
        return np.array(offs, dtype=np.int64), names
    if moves == 'axis':
        axis_names = ['x', 'y', 'z'] if ndim <= 3 else [f'd{k + 1}' for k in range(ndim)]
        offs, names = [], []
        for k in range(ndim):
            for sign, s in ((1, '+'), (-1, '-')):
                o = [0] * ndim
                o[k] = sign
                offs.append(o)
                names.append(axis_names[k] + s)
        return np.array(offs, dtype=np.int64), names
    raise ValueError(f'moves must be "axis" or "full", got {moves!r}')


def attach_lattice(gres, shape, moves):
    """
    Build the neighbour table, boundary and travel points and adjacency for the kept grid points.

    Reads ``gres.grids['mapped_grids']`` (flat lattice indices, any order). A kept grid point with fewer
    valid neighbours than the number of moves is a boundary point, otherwise a travel point.

    Parameters
    ----------
    gres
        Grids results holding 'mapped_grids'.
    shape
        Lattice shape.
    moves
        'axis' or 'full', see ``make_offsets``.

    Returns
    ----------
    None
    """
    shape = tuple(int(s) for s in shape)
    mapped = np.asarray(gres.grids['mapped_grids'], dtype=np.int64)
    offsets, names = make_offsets(len(shape), moves)
    if moves == 'full' and len(shape) >= 4:
        warnings.warn(f'moves="full" gives {len(offsets)} moves per grid point in {len(shape)}-D; the neighbour '
                      f'table and reward tables may be large.', UserWarning)
    M = len(mapped)
    pos = np.full(int(np.prod(shape)), -1, dtype=np.int64)
    pos[mapped] = np.arange(M)
    coords = np.array(np.unravel_index(mapped, shape)).T
    neighbors = np.full((M, len(offsets)), -1, dtype=np.int64)
    for a, off in enumerate(offsets):
        nc = coords + off
        inside = ((nc >= 0) & (nc < np.array(shape))).all(1)
        neighbors[inside, a] = pos[np.ravel_multi_index(nc[inside].T, shape)]
    valid = neighbors >= 0
    is_boundary = valid.sum(1) < len(offsets)
    rows, cols = np.nonzero(valid)
    gres.grids['ndim'] = len(shape)
    gres.grids['shape'] = shape
    gres.grids['moves'] = moves
    gres.grids['offsets'] = offsets
    gres.grids['actions'] = names
    gres.grids['neighbors'] = neighbors
    gres.grids['mapped_boundary'] = mapped[is_boundary]
    gres.grids['mapped_travel'] = mapped[~is_boundary]
    gres.grids['mapped_adj'] = sparse.csr_matrix((np.ones(len(rows)), (rows, neighbors[rows, cols])),
                                                 shape=(M, M))


def lattice_from_scan(gres):
    """
    Copy a 2-D scan grid result and attach the n-D lattice to the copy.

    The kept grid points and their stored order are unchanged, so the n-D functions can be run on a 2-D scan grid.

    Parameters
    ----------
    gres
        Grids results from the 2-D scan.

    Returns
    ----------
    New grids results.
    """
    from .GridCore import Grids_Results
    new = Grids_Results()
    for name in ('embedding', 'grids', 'qlearning'):
        getattr(new, name).update(copy.deepcopy(getattr(gres, name)))
    n = gres.grids['n']
    attach_lattice(new, (n, n), 'full')
    new.grids['method'] = 'scan'
    return new


def is_nd(gres):
    """True when the grid carries a neighbour table (built by the n-D code)."""
    return 'neighbors' in gres.grids


def check_embedding(X):
    """Return the embedding as a float array, or raise ValueError when it cannot carry a grid."""
    X = np.asarray(X)
    if X.ndim != 2 or not 2 <= X.shape[1] <= 5:
        raise ValueError(f'The embedding must be a cells x dimensions array with 2 to 5 dimensions, '
                         f'got shape {X.shape}.')
    if X.shape[0] < 2:
        raise ValueError('The embedding must hold at least 2 cells.')
    if not np.isfinite(X).all():
        raise ValueError('The embedding contains NaN or infinite values.')
    if np.ptp(X, axis=0).max() == 0:
        raise ValueError('All cells have the same coordinates; a grid cannot be laid over them.')
    return X


def grids_nd(X, n=None, radius=None, moves='auto'):
    """
    Regular grid over a 2-D to 5-D embedding, kept where the cells are.

    The lattice has equal spacing on every axis. A lattice point is kept when a cell lies within ``radius``
    grid steps of it; one further layer of points around them forms the rim. Kept points with fewer valid
    neighbours than the number of moves are boundary points, the others are travel points.

    Parameters
    ----------
    X
        Embedding, cells x dimensions (2 to 5).
    n
        Number of grid points across the widest axis of the data.
        (Default: 50 in 2-D, 30 in 3-D, 16 in 4-D, 12 in 5-D)
    radius
        Distance in grid steps within which a cell keeps a grid point; never less than half the diagonal
        of one grid cell.
        (Default: 1.0 in 2-D and 3-D, 1.5 in 4-D, 2.0 in 5-D)
    moves
        'axis' (one step along one axis) or 'full' (diagonal steps as well); 'auto' is 'full' in 2-D and
        'axis' in 3-D and above.
        (Default: 'auto')

    Returns
    ----------
    Grids results with mapped information
    """
    from .GridCore import Grids_Results
    X = check_embedding(X)
    d = X.shape[1]
    n = DEFAULT_N[d] if n is None else int(n)
    if n < 2:
        raise ValueError('n must be at least 2.')
    radius = DEFAULT_RADIUS[d] if radius is None else float(radius)
    if moves == 'auto':
        moves = 'full' if d == 2 else 'axis'
    lo, hi = X.min(0).astype(float), X.max(0).astype(float)
    span = hi - lo
    h = span.max() / (n - 1)
    r = max(radius, np.sqrt(d) / 2)
    pad = int(np.ceil(r)) + 1
    inner = np.ceil(span / h - 1e-9).astype(int) + 1
    shape = tuple(int(k) for k in inner + 2 * pad)
    size = int(np.prod(shape, dtype=np.float64))
    if size > MAX_LATTICE_POINTS:
        raise ValueError(f'A grid with n={n} would have {size:,} lattice points (limit {MAX_LATTICE_POINTS:,}); '
                         f'use a smaller n.')
    origin = lo - ((inner - 1) * h - span) / 2 - pad * h
    coords = np.indices(shape).reshape(d, -1).T
    dist, _ = cKDTree((X - origin) / h).query(coords, k=1, distance_upper_bound=r + 1e-9)
    core = (dist <= r).reshape(shape)
    structure = ndimage.generate_binary_structure(d, d if d <= 4 else 1)
    kept = ndimage.binary_dilation(core, structure)

    gres = Grids_Results()
    gres.embedding['embedding'] = X
    gres.grids['n'] = n
    gres.grids['grids'] = origin + coords * h
    gres.grids['spacing'] = np.full(d, h)
    gres.grids['origin'] = origin
    gres.grids['method'] = 'occupancy'
    gres.grids['radius'] = r
    gres.grids['mapped_grids'] = np.flatnonzero(kept.ravel())
    gres.grids['masked_grids'] = np.flatnonzero(~kept.ravel())
    gres.grids['core'] = core.ravel()[gres.grids['mapped_grids']]
    attach_lattice(gres, shape, moves)

    M = len(gres.grids['mapped_grids'])
    parts = connected_components(gres.grids['mapped_adj'], directed=False)[0]
    print(f'Grid {"x".join(map(str, shape))}: {M} grid points kept, '
          f'{len(gres.grids["mapped_boundary"]) / M:.0%} boundary, {parts} connected part(s), '
          f'{len(gres.grids["actions"])} moves')
    if parts > 1:
        print(f'Warning: There are {parts} components in the graph, please consider to reduce the parameter "n" '
              f'or increase "radius".')
    return gres


def _positions(gres, grid_index):
    """Positions in 'mapped_grids' of the given flat grid indices."""
    mapped = np.asarray(gres.grids['mapped_grids'])
    order = np.argsort(mapped)
    return order[np.searchsorted(mapped, np.asarray(grid_index), sorter=order)]


def check_labels(gres, names):
    """Raise ValueError when a cluster name labels no grid point."""
    present = list(dict.fromkeys(np.asarray(gres.grids['mapped_grids_clusters']).tolist()))
    missing = [c for c in names if c not in present]
    if missing:
        raise ValueError(f'Cluster(s) {missing} label no grid point. Labels present on the grid: {present}')


def project_cluster_nd(gres, clusters=None, cluster_colors=None):
    """
    n-D version of ``project_cluster``: each kept grid point takes the label of its nearest cell.

    Without ``cluster_colors`` the clusters get the qualitative matplotlib ``tab20`` colours (up to 20
    clusters) or seaborn ``husl`` colours (more than 20), so that neighbouring clusters differ in hue.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    if clusters is None:
        print('Warning: No annotation is provided!')
        return
    X = gres.embedding['embedding']
    points = gres.grids['grids'][gres.grids['mapped_grids']]
    gres.embedding['clusters'] = clusters
    _, nn = cKDTree(X).query(points, k=1)
    mapped_grids_clusters = np.asarray(list(clusters))[nn]
    if cluster_colors is None:
        n = len(clusters.cat.categories)
        cluster_colors = list(plt.get_cmap('tab20').colors[:n]) if n <= 20 else sns.husl_palette(n)
    color_lut = dict(zip(clusters.cat.categories, cluster_colors))
    gres.embedding['cluster_colors'] = np.array([color_lut[i] for i in clusters])
    gres.grids['mapped_grids_clusters'] = mapped_grids_clusters
    gres.grids['mapped_grids_colors'] = np.array([color_lut[i] for i in mapped_grids_clusters])


def _closest_pair(gres, tree, in_main, in_other):
    """
    Positions (main, other) of the closest pair of grid points between two parts of the grid.

    Candidate pairs near the minimum distance are found with the tree; their distances are then computed on
    the stored grid coordinates and the exact minimum is kept. Ties are broken by the smallest grid index of
    the main-part point, then of its partner.
    """
    G = gres.grids['grids']
    mapped = np.asarray(gres.grids['mapped_grids'])
    points = G[mapped]
    dist, _ = tree.query(points[in_other], k=1)
    reach = dist.min() * (1 + 1e-7) + 1e-12
    others = in_other[dist <= reach]
    mains = np.unique(np.concatenate([in_main[np.asarray(i, dtype=np.int64)]
                                      for i in tree.query_ball_point(points[others], reach)]))
    D = np.vstack([np.linalg.norm(G[mapped[m]] - G[mapped[others]], axis=1) for m in mains])
    rows, cols = np.nonzero(D == D.min())
    k = np.lexsort((mapped[others[cols]], mapped[mains[rows]]))[0]
    return mains[rows[k]], others[cols[k]]


def align_pseudotime_nd(gres, early_cluster, n_sample_cells=10, key_add='pseudotime', boundary=True,
                        early_cell=None):
    """
    n-D version of ``align_pseudotime``.

    Pseudotime is the mean number of grid steps from start points sampled in the early cluster, min-max
    scaled; the unscaled range in grid steps is stored as ``gres.grids[key_add + '_span']``. When the grid has several connected parts, the part holding most sampled start points is the
    main part (on equal counts, the part holding the start point with the smallest grid index), and each other part is attached at its grid point nearest to the main part, continuing the
    pseudotime of the main-part grid point it is nearest to. Ties between equally near pairs are broken by
    the smallest grid index (of the main-part point, then of its partner).
    """
    mapped_grids = np.asarray(gres.grids['mapped_grids'])
    mapped_grids_clusters = np.asarray(gres.grids['mapped_grids_clusters'])
    points = gres.grids['grids'][mapped_grids]
    if early_cell:
        _, p = cKDTree(points).query(gres.embedding['embedding'][early_cell], k=1)
        start_point = [mapped_grids[p]]
    else:
        check_labels(gres, [early_cluster])
        candidates = mapped_grids[np.where(mapped_grids_clusters == early_cluster)[0]]
        if boundary:
            on_boundary = list(set(candidates) & set(gres.grids['mapped_boundary']))
            candidates = on_boundary if len(on_boundary) else candidates
        start_point = np.random.choice(candidates, n_sample_cells)
    starts = _positions(gres, start_point)
    adj = gres.grids['mapped_adj']
    n_parts, part = connected_components(adj, directed=False)
    if n_parts == 1:
        uniq, inv = np.unique(starts, return_inverse=True)
        t = dijkstra(adj, directed=False, indices=uniq)[inv].mean(0)
    else:
        print(f'Warning: The grid graph has {n_parts} components; the component holding most start points is '
              f'used for pseudo-time aligning and the others are attached at their nearest grid point')
        counts = np.bincount(part[starts], minlength=n_parts)
        tied = np.isin(part[starts], np.flatnonzero(counts == counts.max()))
        main = part[starts[tied][np.argmin(mapped_grids[starts[tied]])]]
        t = dijkstra(adj, directed=False, indices=np.unique(starts[part[starts] == main])).mean(0)
        in_main = np.flatnonzero(part == main)
        tree = cKDTree(points[in_main])
        for c in range(n_parts):
            if c == main:
                continue
            in_c = np.flatnonzero(part == c)
            m, o = _closest_pair(gres, tree, in_main, in_c)
            t[in_c] = dijkstra(adj, directed=False, indices=o)[in_c] + t[m]
    gres.grids[key_add + '_span'] = t.max() - t.min()
    t = (t - t.min()) / (t.max() - t.min())
    gres.grids[key_add] = pd.Series(t, index=mapped_grids)


def project_back_nd(gres, key, neighbors=15, w=None, negative=False):
    """
    n-D version of ``project_back``: Gaussian kernel over the nearest kept grid points of each cell. Where the
    distances to those grid points are (almost) equal, so that the kernel has no width, they get equal weight.
    """
    X = gres.embedding['embedding']
    points = gres.grids['grids'][gres.grids['mapped_grids']]
    k = min(neighbors, len(points))
    dist, nn = cKDTree(points).query(X, k=k)
    dist, nn = dist.reshape(len(X), k), nn.reshape(len(X), k)
    sigma = dist.std(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
        total = weight.sum(1, keepdims=True)
        weight = weight / total
    flat = (sigma == 0) | ~(total > 0)
    weight = np.where(flat, 1 / k, weight)
    val = (weight * np.asarray(gres.grids[key])[nn]).sum(1)
    if w:
        val = val * np.log(w)
    if negative:
        gres.embedding[key] = val
    else:
        gres.embedding[key] = (val - val.min()) / (val.max() - val.min())


def project_nd(gres, data, neighbors=15):
    """n-D version of ``project``: Gaussian kernel over the nearest cells of each kept grid point."""
    gres.embedding['data'] = data
    X = gres.embedding['embedding']
    points = gres.grids['grids'][gres.grids['mapped_grids']]
    data = data.reset_index(drop=True)
    k = min(neighbors, len(X))
    dist, nn = cKDTree(X).query(points, k=k)
    dist, nn = dist.reshape(len(points), k), nn.reshape(len(points), k)
    sigma = dist.std(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
        total = weight.sum(1, keepdims=True)
        weight = np.where(total == 0, 0.0, weight / total)
    rows = np.repeat(np.arange(len(points)), k)
    W = sparse.csr_matrix((weight.ravel(), (rows, nn.ravel())), shape=(len(points), len(X)))
    gres.grids['proj'] = pd.DataFrame(W @ data.to_numpy(dtype=float), columns=data.columns)


def get_state_nd(gres, X, KNN=10):
    """Latent vectors of the KNN cells nearest to each kept grid point, (grid points, KNN, latent), mapped order."""
    points = gres.grids['grids'][gres.grids['mapped_grids']]
    k = min(KNN, len(gres.embedding['embedding']))
    _, nn = cKDTree(gres.embedding['embedding']).query(points, k=k)
    return np.asarray(X)[nn.reshape(len(points), k)]

"""Regular grids in two to five dimensions.

A grid is a regular lattice described by a neighbour table: for every kept grid point and every move, the
position of the grid point reached by that move, or -1 when the move leaves the kept grid. Grid points are
indexed in C order (``np.ravel_multi_index``), which in 2-D equals the v1 index ``i + j * n``.
"""
import copy
import itertools

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

V1_ACTIONS_2D = ['R', 'RT', 'T', 'LT', 'L', 'LB', 'B', 'RB']
V1_OFFSETS_2D = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
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
        'full' (every combination of -1, 0, +1 except standing still; the 8 v1 moves in 2-D).

    Returns
    ----------
    Offsets (moves x ndim, int64) and the move names.
    """
    if moves == 'full':
        if ndim == 2:
            return np.array(V1_OFFSETS_2D, dtype=np.int64), list(V1_ACTIONS_2D)
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
    Copy a v1 2-D scan result and attach the n-D lattice to the copy.

    The kept grid points and their stored order are unchanged, so the n-D functions can be run on a v1 grid.

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

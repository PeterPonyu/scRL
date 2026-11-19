"""Regular grids in two to five dimensions.

A grid is a regular lattice described by a neighbour table: for every kept grid point and every move, the
position of the grid point reached by that move, or -1 when the move leaves the kept grid. Grid points are
indexed in C order (``np.ravel_multi_index``), which in 2-D equals the v1 index ``i + j * n``.
"""
import copy
import itertools

import numpy as np
from scipy import sparse

V1_ACTIONS_2D = ['R', 'RT', 'T', 'LT', 'L', 'LB', 'B', 'RB']
V1_OFFSETS_2D = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


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

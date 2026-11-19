import numpy as np
from scRL.GridCore import Grids_Results
from scRL.GridNDCore import make_offsets, attach_lattice, lattice_from_scan, is_nd


def test_offsets_2d_full_match_2d_code_order():
    offs, names = make_offsets(2, 'full')
    assert names == ['R', 'RT', 'T', 'LT', 'L', 'LB', 'B', 'RB']
    assert offs.tolist() == [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]


def test_offsets_counts_and_axis_names():
    for d in (2, 3, 4, 5):
        assert len(make_offsets(d, 'axis')[0]) == 2 * d
    assert len(make_offsets(3, 'full')[0]) == 26
    assert make_offsets(3, 'axis')[1] == ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']
    assert make_offsets(4, 'axis')[1][:2] == ['d1+', 'd1-']


def _block(shape, lo, hi):
    g = Grids_Results()
    mask = np.zeros(shape, bool)
    mask[tuple(slice(lo, hi) for _ in shape)] = True
    g.grids['mapped_grids'] = np.flatnonzero(mask.ravel())
    return g


def test_neighbor_table_points_to_expected_flat_index():
    shape = (4, 5, 6)
    g = _block(shape, 0, 6)
    attach_lattice(g, shape, 'axis')
    mapped, nb, offs = g.grids['mapped_grids'], g.grids['neighbors'], g.grids['offsets']
    coords = np.array(np.unravel_index(mapped, shape)).T
    for p in range(0, len(mapped), 7):
        for a, off in enumerate(offs):
            c = coords[p] + off
            if (c >= 0).all() and (c < shape).all():
                assert mapped[nb[p, a]] == np.ravel_multi_index(c, shape)
            else:
                assert nb[p, a] == -1


def test_only_the_centre_of_a_3x3x3_block_is_travel():
    for moves in ('axis', 'full'):
        g = _block((5, 5, 5), 1, 4)
        attach_lattice(g, (5, 5, 5), moves)
        assert len(g.grids['mapped_travel']) == 1 and len(g.grids['mapped_boundary']) == 26
        assert g.grids['mapped_travel'][0] == np.ravel_multi_index((2, 2, 2), (5, 5, 5))
        assert g.grids['mapped_adj'].shape == (27, 27) and is_nd(g)


def test_lattice_from_scan_matches_2d_code(scan2d):
    nd = lattice_from_scan(scan2d)
    assert np.array_equal(nd.grids['mapped_grids'], scan2d.grids['mapped_grids'])
    assert set(nd.grids['mapped_boundary']) == set(scan2d.grids['mapped_boundary'])
    assert set(nd.grids['mapped_travel']) == set(scan2d.grids['mapped_travel'])
    assert nd.grids['actions'] == ['R', 'RT', 'T', 'LT', 'L', 'LB', 'B', 'RB']
    assert not is_nd(scan2d)

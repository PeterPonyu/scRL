import copy

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

import scRL
from conftest import make_tree, seed_all

ENDS = {'A': 'tipA', 'B': 'tipB'}


def _labelled(X, labels):
    labels = labels.astype(object)
    far = np.linalg.norm(X - np.array([6, 0, 0])[:X.shape[1]], axis=1) > 3.5
    labels[(labels == 'branchA') & far] = 'tipA'
    labels[(labels == 'branchB') & far] = 'tipB'
    return pd.Series(pd.Categorical(labels.astype(str)))


def _grid(X, labels, seed=0, **kwargs):
    gres = scRL.grids_from_embedding(X, **kwargs)
    scRL.project_cluster(gres, _labelled(X, labels))
    scRL.align_pseudotime(gres, 'root', boundary=False, seed=seed)
    scRL.project_back(gres, 'pseudotime')
    scRL.fate_probabilities(gres, ENDS, forward=3)
    return gres


@pytest.fixture(scope='module')
def _g3():
    X, labels, Z, genes = make_tree(d=3)
    seed_all(0)
    return _grid(X, labels)


@pytest.fixture(scope='module')
def _g2():
    X, labels, Z, genes = make_tree(d=2)
    seed_all(0)
    return _grid(X.astype(float), labels, n=20, j=3, n_jobs=1)     # float64: the scan grid keeps the cells' precision


@pytest.fixture
def g3(_g3):
    return copy.deepcopy(_g3)


@pytest.fixture
def g2(_g2):
    return copy.deepcopy(_g2)


def _own_shares(gres):
    """The shares the package gives its own cells, divided by the share of arriving walks."""
    S = np.column_stack([gres.embedding[f'fate_{n}'] for n in ENDS])
    return S / gres.embedding['fate_reached'][:, None]


def _reference_shares(gres, Yq, k=15):
    """Independent re-implementation of the validated arithmetic."""
    pts = gres.grids['grids'][gres.grids['mapped_grids']]
    H = gres.grids['fate'].to_numpy(float)
    dist, nn = cKDTree(pts).query(Yq, k=k)
    S = np.zeros((len(Yq), H.shape[1]))
    for i in range(len(Yq)):
        sigma = dist[i].std()
        w = np.full(k, 1 / k) if sigma == 0 else np.exp(-dist[i] ** 2 / (2 * sigma ** 2))
        w = w / w.sum()
        S[i] = w @ H[nn[i]]
    S = np.clip(S, 0, 1)
    over = S.sum(1)
    S[over > 1] /= over[over > 1, None]
    return S / S.sum(1, keepdims=True)


# ------------------------------------------------------------------ place_cells
def test_place_cells_single_neighbour_returns_the_embedding_rows():
    rng = np.random.default_rng(0)
    rep = rng.normal(size=(50, 6))
    emb = rng.normal(size=(50, 3))
    rows = [3, 17, 40]
    out = scRL.place_cells(rep, emb, rep[rows], neighbors=1)
    assert out.shape == (3, 3)
    assert np.array_equal(out, emb[rows])


def test_place_cells_mean_of_three_nearest_by_hand():
    rep = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    emb = np.array([[0.0, 0.0], [3.0, 6.0], [6.0, 3.0], [100.0, 100.0], [200.0, 200.0]])
    out = scRL.place_cells(rep, emb, np.array([[1.2]]), neighbors=3)
    assert np.allclose(out, [[3.0, 3.0]])                 # rows 0, 1, 2
    out = scRL.place_cells(rep, emb, np.array([[10.4]]), neighbors=2)
    assert np.allclose(out, [[150.0, 150.0]])             # rows 3, 4


def test_place_cells_caps_neighbors_at_the_number_of_cells():
    rep = np.array([[0.0], [1.0], [2.0]])
    emb = np.array([[0.0, 0.0], [3.0, 3.0], [6.0, 6.0]])
    assert np.allclose(scRL.place_cells(rep, emb, np.array([[0.5]]), neighbors=50), [[3.0, 3.0]])


def test_place_cells_errors():
    rep, emb, q = np.zeros((10, 4)), np.zeros((10, 3)), np.zeros((2, 4))
    with pytest.raises(ValueError, match='row'):
        scRL.place_cells(rep, emb[:9], q)
    with pytest.raises(ValueError, match='feature'):
        scRL.place_cells(rep, emb, np.zeros((2, 5)))
    with pytest.raises(ValueError, match='neighbors'):
        scRL.place_cells(rep, emb, q, neighbors=0)
    for bad in (np.nan, np.inf):
        for which in range(3):
            args = [rep.copy(), emb.copy(), q.copy()]
            args[which][0, 0] = bad
            with pytest.raises(ValueError, match='finite|NaN'):
                scRL.place_cells(*args)


# ------------------------------------------------------------------ project_cells
def test_vector_key_reproduces_project_back_3d(g3):
    pos, _ = scRL.project_cells(g3, g3.embedding['embedding'], key='pseudotime')
    assert isinstance(pos, np.ndarray) and pos.ndim == 1
    scRL.project_back(g3, 'pseudotime', negative=True)
    assert np.abs(pos - g3.embedding['pseudotime']).max() < 1e-10


def test_vector_key_reproduces_project_back_2d(g2):
    pos, _ = scRL.project_cells(g2, g2.embedding['embedding'], key='pseudotime')
    scRL.project_back(g2, 'pseudotime', negative=True)
    assert np.abs(pos - g2.embedding['pseudotime']).max() < 1e-10


@pytest.mark.parametrize('name', ['g3', 'g2'])
def test_shares_reproduce_the_cells_of_the_grid(name, request):
    gres = request.getfixturevalue(name)
    X = gres.embedding['embedding']
    S, outside = scRL.project_cells(gres, X, key='fate')
    assert isinstance(S, pd.DataFrame)
    assert np.abs(S.to_numpy() - _own_shares(gres)).max() < 1e-8
    assert np.abs(S.to_numpy() - _reference_shares(gres, X)).max() < 1e-8


def test_shares_are_a_valid_table(g3):
    rng = np.random.default_rng(1)
    X = g3.embedding['embedding']
    Yq = X[rng.integers(0, len(X), 200)] + rng.normal(0, 0.5, (200, 3))
    S, _ = scRL.project_cells(g3, Yq)
    assert list(S.columns) == list(g3.grids['fate'].columns)
    assert len(S) == 200
    v = S.to_numpy()
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1 + 1e-12).all()
    assert np.allclose(v.sum(1), 1)


@pytest.mark.parametrize('name', ['g3', 'g2'])
def test_outside_flags(name, request):
    gres = request.getfixturevalue(name)
    X = gres.embedding['embedding']
    d = X.shape[1]
    pts = gres.grids['grids']
    step = np.ptp(pts, axis=0) / (gres.grids['n'] - 1) if name == 'g2' else gres.grids['spacing']
    far = X.max(0) + 10 * step
    S, out = scRL.project_cells(gres, far[None, :], key='fate')
    assert out.dtype == bool and out.all()
    assert np.isfinite(S.to_numpy()).all()
    v, out = scRL.project_cells(gres, far[None, :], key='pseudotime')
    assert out.all() and np.isfinite(v).all()
    _, own = scRL.project_cells(gres, X)
    assert own.mean() < 0.01


def test_max_distance_is_monotone(g3):
    rng = np.random.default_rng(2)
    X = g3.embedding['embedding']
    Yq = X[rng.integers(0, len(X), 300)] + rng.normal(0, 1.5, (300, 3))
    flags = [scRL.project_cells(g3, Yq, max_distance=r)[1] for r in (0.5, 1.0, 2.0, 4.0)]
    for small, large in zip(flags, flags[1:]):
        assert (large <= small).all()
    assert flags[0].sum() > flags[-1].sum()


def test_project_cells_errors(g3):
    X = g3.embedding['embedding']
    with pytest.raises(ValueError, match='column'):
        scRL.project_cells(g3, X[:, :2])
    with pytest.raises(ValueError, match='fate'):
        scRL.project_cells(g3, X, key='nope')
    bad = X[:3].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite|NaN'):
        scRL.project_cells(g3, bad)
    with pytest.raises(ValueError, match='neighbors'):
        scRL.project_cells(g3, X, neighbors=0)
    for r in (0, -1):
        with pytest.raises(ValueError, match='max_distance'):
            scRL.project_cells(g3, X, max_distance=r)


def test_importable_from_the_package():
    import scRL as pkg
    assert callable(pkg.place_cells) and callable(pkg.project_cells)
    assert {'place_cells', 'project_cells'} <= set(pkg.__all__)


# ------------------------------------------------------------------ held-out cells
def _agreement(seed):
    X, labels, Z, genes = make_tree(d=3, seed=seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    ref, qry = np.sort(perm[:int(.7 * len(X))]), np.sort(perm[int(.7 * len(X)):])
    lab = _labelled(X, labels)
    seed_all(seed)
    full = scRL.grids_from_embedding(X)
    scRL.project_cluster(full, lab)
    scRL.align_pseudotime(full, 'root', boundary=False, seed=seed)
    scRL.fate_probabilities(full, ENDS, forward=3)
    part = scRL.grids_from_embedding(X[ref])
    scRL.project_cluster(part, lab.iloc[ref].reset_index(drop=True))
    scRL.align_pseudotime(part, 'root', boundary=False, seed=seed)
    scRL.fate_probabilities(part, ENDS, forward=3)
    Yq = scRL.place_cells(Z[ref], X[ref], Z[qry], neighbors=10)
    S, _ = scRL.project_cells(part, Yq)
    want = _own_shares(full)[qry]
    return float((S.to_numpy().argmax(1) == want.argmax(1)).mean())


def test_held_out_cells_keep_their_top_lineage():
    agree = [_agreement(s) for s in range(5)]
    print('held-out agreement', agree)
    assert min(agree) >= 0.65

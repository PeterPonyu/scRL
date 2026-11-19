import copy
import random

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
import torch

import scRL


def make_tree(n_cells=600, d=3, d_latent=8, seed=0):
    """Y-shaped differentiation tree in d dimensions: root -> trunk -> branchA / branchB."""
    rng = np.random.default_rng(seed)
    dirs = np.zeros((3, d))
    dirs[0, 0] = 1.0
    dirs[1, 0], dirs[1, 1] = 0.6, 0.8
    if d == 2:
        dirs[2, 0], dirs[2, 1] = 0.6, -0.8
    else:
        dirs[2, 0], dirs[2, 2] = 0.6, 0.8
    seg = rng.choice(3, n_cells, p=[0.4, 0.3, 0.3])
    t = rng.uniform(0, 1, n_cells)
    base = np.where(seg[:, None] == 0, t[:, None] * dirs[0] * 6, dirs[0] * 6 + t[:, None] * dirs[seg] * 6)
    X = base + rng.normal(0, 0.35, (n_cells, d))
    labels = np.where(seg == 0, np.where(t < 0.3, 'root', 'trunk'), np.where(seg == 1, 'branchA', 'branchB'))
    W = rng.normal(0, 1, (d, d_latent))
    Z = (X @ W + rng.normal(0, 0.1, (n_cells, d_latent))).astype(np.float32)
    genes = pd.DataFrame({
        'geneA': np.clip((seg == 1) * t * 3 + rng.normal(0, .1, n_cells), 0, None),
        'geneB': np.clip((seg == 2) * t * 3 + rng.normal(0, .1, n_cells), 0, None),
        'geneRoot': np.clip((seg == 0) * (1 - t) * 3 + rng.normal(0, .1, n_cells), 0, None),
    })
    return X.astype(np.float32), labels, Z, genes


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _scan(n):
    X, labels, Z, genes = make_tree(d=2)
    seed_all(0)
    gres = scRL.grids_from_embedding(X, n=n, j=3, n_jobs=1)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    np.random.seed(1)
    scRL.align_pseudotime(gres, 'root', boundary=False)
    scRL.project(gres, genes)
    return gres


@pytest.fixture(scope='session')
def _scan20():
    return _scan(20)


@pytest.fixture(scope='session')
def _scan50():
    return _scan(50)


@pytest.fixture
def scan2d(_scan20):
    """2-D scan grid result (n=20) with clusters, pseudotime and projected genes. Fresh copy per test."""
    return copy.deepcopy(_scan20)


@pytest.fixture
def scan2d_n50(_scan50):
    """Same with n=50."""
    return copy.deepcopy(_scan50)


@pytest.fixture(scope='session')
def tree2d():
    return make_tree(d=2)


@pytest.fixture(scope='session')
def tree3d():
    return make_tree(d=3)

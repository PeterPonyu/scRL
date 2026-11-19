"""Combine value maps of several lineages into lineage weights."""
import numpy as np
import pandas as pd
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
    """Heaviest lineage and normalised entropy of each row of a weight matrix."""
    top = np.asarray(names, dtype=object)[W.argmax(1)].astype(str)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.where(W > 0, W * np.log(W), 0).sum(1) / np.log(W.shape[1])
    return top, entropy


def fate_gate(gres, keys, names=None, tau=.1, key_add='gate'):
    """
    Combine one value map per lineage into lineage weights.

    Each map is rank-scaled to [0, 1] over the kept grid points, and the weights are the softmax of the
    scaled values divided by ``tau`` across lineages: a grid point gets most of its weight from the lineage in
    whose map it ranks highest. A small ``tau`` gives nearly one-hot weights.

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
    values = np.column_stack([_kept_values(gres, k) for k in keys])
    M = len(values)
    scaled = np.column_stack([(rankdata(v) - 1) / (M - 1) for v in values.T])
    z = scaled / tau
    W = np.exp(z - z.max(1, keepdims=True))
    W /= W.sum(1, keepdims=True)
    colors = {n: to_hex(plt.get_cmap('tab10')(i % 10)) for i, n in enumerate(names)}
    gate = pd.DataFrame(W, index=list(gres.grids['mapped_grids']), columns=names)
    top, entropy = _summaries(W, names)
    gres.grids[key_add] = gate
    for i, n in enumerate(names):
        gres.grids[f'{key_add}_{n}'] = W[:, i]
    gres.grids[f'{key_add}_top'], gres.grids[f'{key_add}_entropy'] = top, entropy
    gres.grids[f'{key_add}_top_colors'] = colors
    for n in names:
        project_back(gres, f'{key_add}_{n}', negative=True)
    Wc = np.clip(np.column_stack([gres.embedding[f'{key_add}_{n}'] for n in names]), 0, None)
    Wc /= Wc.sum(1, keepdims=True)
    for i, n in enumerate(names):
        gres.embedding[f'{key_add}_{n}'] = Wc[:, i]
    gres.embedding[f'{key_add}_top'], gres.embedding[f'{key_add}_entropy'] = _summaries(Wc, names)
    gres.embedding[f'{key_add}_top_colors'] = dict(colors)
    return gate

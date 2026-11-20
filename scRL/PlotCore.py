"""Views of a grid and of the values stored on it."""
import contextlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .GridNDCore import _positions

GREY, DARK = '#d9d9d9', '#4d4d4d'


def _raw_grid_values(gres, key, travel):
    """Values of ``gres.grids[key]`` at the travel points, in the order of ``travel``, in their own type."""
    if key not in gres.grids:
        raise KeyError(f'{key!r} is not stored on the grid; available keys: {sorted(map(str, gres.grids))}')
    values = gres.grids[key]
    mapped = np.asarray(gres.grids['mapped_grids'])
    if isinstance(values, pd.Series) and len(values) == len(mapped) and set(values.index) == set(mapped):
        return np.asarray(values.loc[travel])
    values = np.asarray(values).ravel()
    if len(values) != len(mapped):
        raise ValueError(f'gres.grids[{key!r}] has {len(values)} values but the grid has {len(mapped)} points.')
    return values[_positions(gres, travel)]


def _grid_values(gres, key, travel):
    """Values of ``gres.grids[key]`` at the travel points, in the order of ``travel``, as numbers."""
    return _raw_grid_values(gres, key, travel).astype(float)


def _cluster_order(gres):
    """Cluster names in their categorical order."""
    clusters = gres.embedding['clusters']
    if hasattr(clusters, 'cat'):
        return list(clusters.cat.categories)
    return list(dict.fromkeys(np.asarray(clusters).tolist()))


def _is_categorical(values):
    """True for values to draw as categories (strings or a categorical)."""
    if isinstance(getattr(values, 'dtype', None), pd.CategoricalDtype):
        return True
    return np.asarray(values).dtype.kind in 'OUS'


class _Layer(tuple):
    """
    A point set to draw: (coordinates, colour, values, labels). ``colour`` is one colour or one per point;
    ``values`` are numbers for a colour map; ``labels`` is a categorical of category names (its categories give
    the legend order). ``size`` scales the marker area; ``names`` holds the cluster of each point, if known.
    """
    size = 1.0
    names = None


def _layer(coords, colour, values, labels, size=1.0, names=None):
    layer = _Layer((coords, colour, values, labels))
    layer.size = size
    layer.names = names
    return layer


def _missing_cell_key(gres, key):
    """Message for a value asked for on the cells that is not stored there."""
    hint = (f' It is stored on the grid: project it onto the cells with scRL.project_back(gres, {key!r}) first, or '
            f'draw the grid points (cells=False).' if key in gres.grids else '')
    return f'{key!r} is not stored for the cells.{hint} Keys stored for the cells: {sorted(map(str, gres.embedding))}'


def _sub(names, mask):
    return None if names is None else np.asarray(names)[mask]


def _category_layer(gres, coords, labels, key, cells, order=None, names=None):
    """
    Layers coloured by category: the stored colour dict ``<key>_colors`` when present, else tab10/tab20/husl.
    Missing labels (NaN, None) go to a small grey layer drawn first. ``order`` is the category order of a
    categorical; plain strings are sorted.
    """
    raw = np.asarray(labels, dtype=object)
    missing = pd.isna(raw)
    if missing.all():
        raise ValueError(f'{key!r} has no values to draw: every value is missing.')
    labels = raw[~missing].astype(str)
    present = set(labels.tolist())
    base = [str(c) for c in order if str(c) in present] if order is not None else sorted(present)
    stored = (gres.embedding if cells else gres.grids).get(key + '_colors')
    if isinstance(stored, dict) and stored:
        stored = {str(k): v for k, v in stored.items()}
        first = base if order is not None else [c for c in stored if c in present]
        rest = [c for c in base if c not in first]
        order = first + rest
        missing_colours = [c for c in order if c not in stored]
        lut = {**stored, **dict(zip(missing_colours, _palette(len(missing_colours))))}
    else:
        order = base
        lut = dict(zip(order, _palette(len(order))))
    from matplotlib.colors import to_rgba_array
    layers = [_layer(coords[missing], GREY, None, None, .5, _sub(names, missing))] if missing.any() else []
    return layers + [_layer(coords[~missing], to_rgba_array([lut[c] for c in labels]), None,
                            pd.Categorical(labels, categories=order), names=_sub(names, ~missing))]


def _category_order(values):
    """The categories of a categorical (Series or array), else None."""
    if isinstance(getattr(values, 'dtype', None), pd.CategoricalDtype):
        return list(values.cat.categories) if isinstance(values, pd.Series) else list(values.categories)
    return None


def _palette(n):
    """Qualitative colours: tab10 up to 10, tab20 up to 20, husl beyond."""
    if n <= 10:
        return list(plt.get_cmap('tab10').colors[:n])
    if n <= 20:
        return list(plt.get_cmap('tab20').colors[:n])
    import seaborn as sns
    return list(sns.husl_palette(n))


def _value_layers(coords, values, key, names=None):
    """Layers for numeric values: the points without a value (NaN) first, small and grey, then the others."""
    missing = np.isnan(values)
    if missing.all():
        raise ValueError(f'{key!r} has no values to draw: every value is NaN.')
    layers = [_layer(coords[missing], GREY, None, None, .5, _sub(names, missing))] if missing.any() else []
    return layers + [_layer(coords[~missing], None, values[~missing], None, names=_sub(names, ~missing))]


def _layers(gres, key, cells, dims, rim):
    """Point sets to draw (see ``_Layer``)."""
    if cells:
        X = np.asarray(gres.embedding['embedding'])[:, dims]
        names = np.asarray(gres.embedding['clusters']).astype(str) if 'clusters' in gres.embedding else None
        if key is None:
            return [_layer(X, GREY, None, None, names=names)]
        if key == 'clusters':
            return [_layer(X, np.asarray(gres.embedding['cluster_colors']), None,
                           pd.Categorical(np.asarray(gres.embedding['clusters']), categories=_cluster_order(gres)),
                           names=names)]
        if key not in gres.embedding:
            raise KeyError(_missing_cell_key(gres, key))
        values = gres.embedding[key]
        if _is_categorical(values):
            return _category_layer(gres, X, values, key, True, _category_order(values), names)
        return _value_layers(X, np.asarray(values, dtype=float), key, names)
    coords = np.asarray(gres.grids['grids'])
    travel = np.asarray(gres.grids['mapped_travel'])
    boundary = np.asarray(gres.grids['mapped_boundary'])
    labelled = 'mapped_grids_clusters' in gres.grids
    names = np.asarray(gres.grids['mapped_grids_clusters'])[_positions(gres, travel)].astype(str) if labelled else None
    if key is None:
        return [_layer(coords[boundary][:, dims], GREY, None, None), _layer(coords[travel][:, dims], DARK, None, None,
                                                                            names=names)]
    layers = [_layer(coords[boundary][:, dims], GREY, None, None)] if rim else []
    if key == 'clusters':
        p = _positions(gres, travel)
        return layers + [_layer(coords[travel][:, dims], np.asarray(gres.grids['mapped_grids_colors'])[p], None,
                                pd.Categorical(np.asarray(gres.grids['mapped_grids_clusters'])[p],
                                               categories=_cluster_order(gres)), names=names)]
    values = _raw_grid_values(gres, key, travel)
    if _is_categorical(gres.grids[key]) or _is_categorical(values):
        return layers + _category_layer(gres, coords[travel][:, dims], values, key, False,
                                        _category_order(gres.grids[key]), names)
    return layers + _value_layers(coords[travel][:, dims], values.astype(float), key, names)


def _legend_handles(gres, layers):
    """One legend entry per category drawn, in category order, or None when no layer is drawn by category."""
    drawn = [(c, l) for _, c, _, l in layers if l is not None]
    if not drawn:
        return None
    lut = {}
    for colours, labels in drawn:
        for colour, label in zip(colours, np.asarray(labels)):
            lut.setdefault(label, colour)
    return [Line2D([], [], ls='', marker='o', color=lut[c], label=str(c)) for c in drawn[0][1].categories if c in lut]


def _finite(layers):
    values = [v for _, _, v, _ in layers if v is not None]
    if not values:
        return np.zeros(0)
    finite = np.concatenate(values)
    return finite[np.isfinite(finite)]


def _value_limits(layers, vmin=None, vmax=None, vcenter=None, robust=True):
    """
    Colour limits from the finite values drawn. ``robust``: the 1st and 99th percentiles, except on a side where
    no value lies more than 5% of that range beyond the percentile (there the limit stays at the extreme
    value, so only real outliers are cut off); the full range when the two percentiles are equal (most values
    equal). Otherwise the range. ``vcenter`` makes the limits symmetric
    about it; explicit limits win.
    """
    finite = _finite(layers)
    if not len(finite):
        return vmin, vmax
    lo, hi = finite.min(), finite.max()
    if robust:
        p1, p99 = np.percentile(finite, [1, 99])
        if p99 > p1:                              # else most values are equal: keep the full range
            margin = .05 * (p99 - p1)
            lo = p1 if lo < p1 - margin else lo
            hi = p99 if hi > p99 + margin else hi
    if vcenter is not None:
        half = max(abs(lo - vcenter), abs(hi - vcenter))
        lo, hi = vcenter - half, vcenter + half
    return (lo if vmin is None else vmin), (hi if vmax is None else vmax)


def _extend(layers, lo, hi):
    """Which ends of the colour bar need an arrow: values fall outside the limits there."""
    finite = _finite(layers)
    if not len(finite) or lo is None or hi is None:
        return 'neither'
    low, high = (finite < lo).any(), (finite > hi).any()
    return 'both' if low and high else 'min' if low else 'max' if high else 'neither'


def _scatter(ax, coords, colour, values, s, cmap, vmin=None, vmax=None, **kw):
    if ax.name == '3d':
        kw.setdefault('depthshade', False)
    if values is None:
        return ax.scatter(*coords.T, s=s, lw=0, color=colour, **kw)
    return ax.scatter(*coords.T, s=s, lw=0, c=values, cmap=cmap, vmin=vmin, vmax=vmax, **kw)


def _style_3d(ax, dims, view, coords):
    """View angle, equal data aspect in a slightly shrunk box, unlabelled ticks and axis names."""
    ax.view_init(*view)
    span = np.ptp(coords, axis=0)
    ax.set_box_aspect(np.maximum(span, span.max() * 1e-3), zoom=.8)
    for axis, d in zip((ax.xaxis, ax.yaxis, ax.zaxis), dims):
        axis.set_ticklabels([])
        axis.set_label_text(f'Dim {d + 1}')
        axis.labelpad = -8
        axis.set_pane_color((1, 1, 1, 1))
        axis._axinfo['grid'].update(color='#e3e3e3', linewidth=.4)
    if abs(view[0]) >= 85:                    # looking along the third axis: its name has no place
        ax.zaxis.label.set_visible(False)
        ax.set_zticks([])


def plot_grids(gres, key=None, cells=False, dims=(0, 1, 2), view=(22, -58), ax=None, cmap='plasma',
               s=None, rim=False, interactive=False, title=None, colorbar=True, vmin=None, vmax=None, vcenter=None,
               robust=True, label=None, alpha=1):
    """
    Draw the grid, or the cells, coloured by a stored value.

    2-D grids are drawn on ordinary axes; 3-D grids on 3-D axes; 4-D and 5-D grids on the three axes
    chosen with ``dims``.

    Parameters
    ----------
    gres
        Grids results.
    key
        Value to colour by: a key of ``gres.grids`` (grid points) or of ``gres.embedding`` (cells).
        ``'clusters'`` colours by cluster, with a legend; any other key holding strings (or a categorical)
        is drawn the same way, with the colours of ``<key>_colors`` when stored. Points whose value is NaN are
        drawn small and grey. None shows the grid structure: travel points dark grey, boundary points light
        grey. With ``interactive=True`` a list of keys gives one trace per key and a menu to switch between
        them.
        (Default: None)
    cells
        Draw the cells instead of the grid points.
        (Default: False)
    dims
        Embedding axes to draw for 3-D and higher grids.
        (Default: (0, 1, 2))
    view
        Elevation and azimuth of the 3-D view.
        (Default: (22, -58))
    ax
        Axes to draw into; a 3-D axes for 3-D and higher grids. A new figure is made when None.
        (Default: None)
    cmap
        Colour map for the values.
        (Default: 'plasma')
    s
        Marker size.
        (Default: None, chosen from the number of dimensions)
    rim
        Also draw the boundary points in light grey when colouring grid points by a value.
        (Default: False)
    interactive
        Return a rotatable plotly figure instead of drawing with matplotlib (needs plotly).
        (Default: False)
    title
        Title of the plot.
        (Default: None)
    colorbar
        Show a colour bar when a value is drawn (a legend for categories).
        (Default: True)
    vmin, vmax
        Limits of the colour scale.
        (Default: None, from the finite values drawn)
    vcenter
        Value at the centre of the colour scale: the limits are made symmetric about it.
        (Default: None)
    robust
        Colour limits at the 1st and 99th percentile of the values drawn, so that a few outliers do not take
        the colour range; the colour bar shows arrows where values fall outside. False: minimum and maximum.
        (Default: True)
    label
        Colour bar label.
        (Default: None, the key)
    alpha
        Opacity of the points.
        (Default: 1)

    Returns
    ----------
    The matplotlib axes, or a plotly figure when ``interactive`` is True.
    """
    ndim = np.asarray(gres.embedding['embedding']).shape[1]
    three = ndim > 2
    dims = list(dims) if three else [0, 1]
    if interactive:
        keys = key if isinstance(key, (list, tuple)) else [key]
        return _plotly(gres, keys, cells, rim, three, cmap, s, title, colorbar, dims, vmin, vmax, vcenter, robust,
                       label)
    if isinstance(key, (list, tuple)):
        raise ValueError('A list of keys needs interactive=True; draw one key at a time otherwise.')
    layers = _layers(gres, key, cells, dims, rim)
    handles = _legend_handles(gres, layers)
    own = ax is None
    if own:
        if three:
            strip = _legend_width(handles) if (colorbar and handles) else 1.0
            fig = plt.figure(figsize=(4.2 + strip, 4.6))
            right = 4.2 / fig.get_figwidth()
            ax = fig.add_axes([0, 0, right, .95], projection='3d')
        else:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot()
    elif three and ax.name != '3d':
        raise ValueError(f'A {ndim}-D grid needs a 3-D axes, for example fig.add_subplot(projection=\'3d\').')
    if s is None:
        s = 4 if cells else (8 if three else 12)
    lo, hi = _value_limits(layers, vmin, vmax, vcenter, robust)
    mappable = None
    for layer in layers:
        coords, colour, values, _ = layer
        artist = _scatter(ax, coords, colour, values, s * layer.size, cmap, lo, hi, alpha=alpha)
        if values is not None:
            mappable = artist
    if three:
        _style_3d(ax, dims, view, np.vstack([c for c, _, _, _ in layers]))
    else:
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(f'Dim {dims[0] + 1}')
        ax.set_ylabel(f'Dim {dims[1] + 1}')
    fig = ax.figure
    if colorbar and handles:
        if own and three:
            fig.legend(handles=handles, loc='center left', bbox_to_anchor=(right, .5), frameon=False)
        else:
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    elif colorbar and mappable is not None:
        kw = dict(label=key if label is None else label, extend=_extend(layers, lo, hi))
        if own and three:
            fig.colorbar(mappable, cax=fig.add_axes([right + .15 / fig.get_figwidth(), .225, .12 / fig.get_figwidth(),
                                                     .55], label='<colorbar>'), **kw)
        else:
            fig.colorbar(mappable, ax=ax, shrink=.6, pad=.02 if three else .05, **kw)
    if title is not None:
        ax.set_title(title)
    return ax


def _plotly(gres, keys, cells, rim, three, cmap, s, title, colorbar, dims, vmin=None, vmax=None, vcenter=None,
            robust=True, label=None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('plot_grids(interactive=True) needs plotly: pip install plotly') from None
    from matplotlib.colors import to_hex

    def trace(layer, key, visible=True, limits=(None, None)):
        coords, colour, values, labels = layer
        name = str(key)
        if values is not None:
            marker = dict(color=values, colorscale=cmap, showscale=colorbar, cmin=limits[0], cmax=limits[1],
                          colorbar=dict(title=name if label is None else label))
            shown = [f'{name}: {v:.3g}' for v in values]
        elif np.ndim(colour) > 1:
            marker = dict(color=[to_hex(c) for c in colour])
            shown = [f'{name}: {v}' for v in np.asarray(labels)] if labels is not None else [''] * len(coords)
        else:
            marker = dict(color=to_hex(colour))
            shown = [f'{name}: missing' if layer.size < 1 else ''] * len(coords)
        if layer.names is not None:
            shown = [f'{t}<br>cluster: {c}' if t else f'cluster: {c}' for t, c in zip(shown, layer.names)]
        marker['size'] = (np.sqrt(s) if s else 3) * np.sqrt(layer.size)
        xyz = dict(zip('xyz', coords.T))
        kw = dict(mode='markers', marker=marker, visible=visible, text=shown, hovertemplate='%{text}<extra></extra>')
        return go.Scatter3d(**kw, **xyz) if three else go.Scatter(**kw, **xyz)

    traces, groups = [], []
    for i, k in enumerate(keys):
        layers = _layers(gres, k, cells, dims, rim if len(keys) == 1 else False)
        limits = _value_limits(layers, vmin, vmax, vcenter, robust)
        traces += [trace(layer, k, visible=i == 0, limits=limits) for layer in layers]
        groups += [i] * len(layers)
    fig = go.Figure(traces)
    if len(keys) == 1:
        fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=40))
    else:
        buttons = [dict(label=str(k), method='update', args=[{'visible': [g == i for g in groups]}])
                   for i, k in enumerate(keys)]
        fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, y=1, xanchor='left', yanchor='bottom')],
                          margin=dict(l=0, r=0, b=0, t=40))
    labels = {f'{a}axis_title': f'Dim {d + 1}' for a, d in zip('xyz', dims)}
    if three:
        fig.update_layout(scene=dict(aspectmode='data', **labels))
    else:
        fig.update_layout(**labels)
        fig.update_yaxes(scaleanchor='x')
    fig.update_layout(showlegend=False, template='plotly_white')
    return fig


def _require_3d(gres, name):
    if np.asarray(gres.embedding['embedding']).shape[1] < 3:
        raise ValueError(f'{name} needs a grid with 3 or more dimensions; use plot_grids for a 2-D grid.')


def _lattice_spacing(gres):
    """Grid step along each embedding axis, read from the stored lattice coordinates."""
    G = np.asarray(gres.grids['grids'])
    shape = gres.grids.get('shape', (gres.grids['n'],) * G.shape[1])
    strides = [int(np.prod(shape[k + 1:])) for k in range(len(shape))]
    return np.array([G[s, k] - G[0, k] for k, s in enumerate(strides)]), tuple(shape)


def _policy_moves(gres, trainer, travel):
    """
    The move the agent tends to make from each given travel point (flat indices), in embedding units.

    ActorCritic and SAC: the expected move, the moves weighted by the actor's probabilities at the grid point's
    state. DDQN and TabularQ: the move with the largest Q-value. An integer array gives one move index per
    kept grid point; a 2-D array (kept grid points x moves) gives move weights, booleans or probabilities, and
    the weighted mean move is used (for example the equally good moves ``exact_<key>_best``).
    """
    import torch
    from .GridNDCore import OFFSETS_2D
    offsets = np.asarray(gres.grids['offsets'] if 'offsets' in gres.grids else OFFSETS_2D, dtype=float)
    if isinstance(trainer, np.ndarray):
        if len(trainer) != len(gres.grids['mapped_grids']):
            raise ValueError('A move array needs one row per kept grid point.')
        rows = trainer[_positions(gres, travel)]
        if rows.ndim == 2:
            weights = rows.astype(float)
            total = weights.sum(1, keepdims=True)
            move = np.where(total > 0, weights @ offsets / np.where(total > 0, total, 1), 0)
        else:
            move = offsets[rows.astype(int)]
        return move * _lattice_spacing(gres)[0]
    agent = trainer.agent
    if hasattr(agent, 'Q'):
        move = offsets[np.argmax(agent.Q.loc[travel].to_numpy(dtype=float), axis=1)]
    else:
        net = agent.actor if hasattr(agent, 'actor') else agent.policy_net
        device = next(iter(net.parameters())).device if hasattr(net, 'parameters') else None
        states = torch.tensor(trainer.env.state_space.mean(axis=1)[_positions(gres, travel)], dtype=torch.float,
                              device=device)
        out = net(states).detach().cpu().numpy()
        move = out @ offsets if hasattr(agent, 'actor') else offsets[np.argmax(out, axis=1)]
    return move * _lattice_spacing(gres)[0]


def _block_field(gres, policy, block_dims, k, groups=None):
    """
    The policy averaged over blocks of k grid steps along ``block_dims`` (every other axis collapsed), and over
    ``groups`` (for example the slab) when given. Blocks need at least 3 travel points (fewer when a block holds
    fewer lattice positions). Returns the mean position (embedding units), the mean move (grid steps, all axes)
    and the group of every block.
    """
    travel = np.asarray(gres.grids['mapped_travel'])
    spacing = _lattice_spacing(gres)[0]
    moves = _policy_moves(gres, policy, travel) / spacing
    c = _lattice_coords(gres, travel)
    cells = c[:, list(block_dims)] // k
    keys = cells if groups is None else np.column_stack([groups, cells])
    _, block, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    block = block.ravel()
    enough = counts >= min(3, k ** len(block_dims))
    n = len(counts)
    tails = np.asarray(gres.grids['grids'])[travel]
    pos = np.zeros((n, tails.shape[1]))
    mean = np.zeros((n, moves.shape[1]))
    np.add.at(pos, block, tails)
    np.add.at(mean, block, moves)
    pos, mean = pos / counts[:, None], mean / counts[:, None]
    group = np.zeros(n) if groups is None else np.zeros(n)
    if groups is not None:
        group[block] = groups
    return pos[enough], mean[enough], group[enough]


def _lattice_coords(gres, flat):
    """Integer lattice coordinates (points x axes) of flat grid indices."""
    return np.array(np.unravel_index(np.asarray(flat), _lattice_spacing(gres)[1])).T


def _auto_step(gres, dims, target=15):
    """Sub-lattice step so that about ``target`` arrows span the longest of the given lattice axes."""
    kept = _lattice_coords(gres, gres.grids['mapped_grids'])[:, dims]
    return max(1, int(np.ceil((np.ptp(kept, axis=0).max() + 1) / target)))


def _thinned_travel(gres, dims, step):
    """Travel points whose lattice coordinates along ``dims`` are multiples of ``step``."""
    travel = np.asarray(gres.grids['mapped_travel'])
    keep = (_lattice_coords(gres, travel)[:, dims] % step == 0).all(1)
    return travel[keep]


ARROW = '#333333'


def _across_cmap():
    """Blue (toward lower values) through dark grey (within the plane) to red (toward higher values)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('across', ['#2166ac', '#3a3a3a', '#b2182b'])


def _sections(gres, axis, n_slices, dims):
    """How a grid is cut into slabs: the cut axis, the two shown axes and a function giving each point's slab."""
    dims = list(dims)
    cut, shown = dims[axis], [d for i, d in enumerate(dims) if i != axis]
    spacing, shape = _lattice_spacing(gres)
    origin = np.asarray(gres.grids['grids'])[0]
    kept = _lattice_coords(gres, gres.grids['mapped_grids'])[:, cut]
    lo, width = kept.min() - .5, (kept.max() - kept.min() + 1) / n_slices

    def slab_of(coords):
        u = np.rint((coords[:, cut] - origin[cut]) / spacing[cut])
        return np.clip(np.floor((u - lo) / width), 0, n_slices - 1)

    def bounds(k):
        return origin[cut] + (lo + k * width) * spacing[cut], origin[cut] + (lo + (k + 1) * width) * spacing[cut]

    return cut, shown, slab_of, bounds


def _draw_section(ax, k, layers, cut, shown, slab_of, bounds, limits, cmap, vmin, vmax, s, arrows=None,
                  title_size='medium'):
    """Draw slab ``k`` of a sectioned grid into a 2-D axes; returns the colour-mapped artist, if any."""
    mappable = None
    for layer in layers:
        coords, colour, vals, _ = layer
        slab = slab_of(coords) == k
        colour_k = colour[slab] if np.ndim(colour) > 1 else colour
        artist = _scatter(ax, coords[slab][:, shown], colour_k, None if vals is None else vals[slab],
                          s * layer.size, cmap, vmin, vmax)
        if vals is not None:
            mappable = artist
    if arrows is not None:
        pos, mean, group, step, spacing = arrows
        here = group == k
        inplane, across = mean[here][:, shown], mean[here][:, cut]
        size = np.linalg.norm(inplane, axis=1)
        draw = size >= .05
        if draw.any():
            uv = inplane[draw] * .8 * step * spacing[shown]
            ax.quiver(*pos[here][draw][:, shown].T, *uv.T, np.clip(across[draw], -1, 1), cmap=_across_cmap(),
                      clim=(-1, 1), angles='xy', scale_units='xy', scale=1, width=.006, headwidth=3.5,
                      headlength=4, headaxislength=3.5, zorder=3)
    a, b = bounds(k)
    ax.set_title(f'Dim {cut + 1}: {a:.2f} to {b:.2f}', fontsize=title_size)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f'Dim {shown[0] + 1}')
    ax.set_ylabel(f'Dim {shown[1] + 1}')
    return mappable


def _legend_width(handles):
    """Width in inches of a one-column legend of these entries at the default font size, with a margin."""
    size = plt.rcParams['legend.fontsize']
    size = plt.rcParams['font.size'] if isinstance(size, str) else size
    return .75 + .62 * size / 72 * max(len(h.get_label()) for h in handles)


def _limits(points, pad=.03):
    span = np.ptp(points, axis=0)
    return list(zip(points.min(0) - pad * span, points.max(0) + pad * span))


def _panel_grid(fig_w_extra, ncols, nrows, aspect, panel_w):
    """Figure and axes laid out in inches: panels of width ``panel_w`` and height ``aspect * panel_w``."""
    panel_h = float(np.clip(aspect, .35, 2.5)) * panel_w
    left, gap_x, top, gap_y, bottom = .45, .5, .35, .75, .45
    W = left + ncols * panel_w + (ncols - 1) * gap_x + fig_w_extra
    H = top + nrows * panel_h + (nrows - 1) * gap_y + bottom
    fig = plt.figure(figsize=(W, H))
    axes = [fig.add_axes([(left + c * (panel_w + gap_x)) / W, (H - top - (r + 1) * panel_h - r * gap_y) / H,
                          panel_w / W, panel_h / H]) for r in range(nrows) for c in range(ncols)]
    return fig, axes, (left + ncols * panel_w + (ncols - 1) * gap_x) / W


def plot_slices(gres, key=None, axis=2, n_slices=6, cells=False, dims=(0, 1, 2), cmap='plasma', s=None,
                ncols=None, figsize=None, policy=None, policy_every=None, vmin=None, vmax=None, vcenter=None,
                robust=True, label=None):
    """
    Cut a 3-D (or higher) grid into slabs along one axis and draw each slab as a 2-D section.

    The slabs have equal width in grid steps along ``dims[axis]``, so every kept grid point (or every cell,
    with ``cells=True``) falls in exactly one slab. All panels share the axis limits and one colour scale;
    each panel title gives the slab's coordinate range. The figure size follows the data aspect.

    Parameters
    ----------
    gres
        Grids results of a 3-D or higher grid.
    key
        Value to colour by (see ``plot_grids``; ``'clusters'`` colours by cluster with one legend). None
        shows travel points dark and boundary points light. With a key and ``cells=False`` only travel
        points are drawn.
        (Default: None)
    axis
        Position in ``dims`` of the axis to cut along.
        (Default: 2)
    n_slices
        Number of slabs.
        (Default: 6)
    cells
        Draw the cells instead of the grid points.
        (Default: False)
    dims
        Embedding axes used.
        (Default: (0, 1, 2))
    cmap
        Colour map for the values.
        (Default: 'plasma')
    s
        Marker size.
        (Default: None)
    ncols
        Panels per row.
        (Default: None, at most 3)
    figsize
        Figure size; the panels are then placed by matplotlib.
        (Default: None, from the number of panels and the data aspect)
    policy
        A trained ``scRL.trainer``, or an array of moves (see ``plot_policy``): draw the direction
        the agent tends to move in, averaged over blocks of ``policy_every`` x ``policy_every`` lattice
        positions of the slab (blocks with at least 3 travel points). Each arrow starts at the mean position
        of its block and is the mean move in the plane, 0.8 block widths long when all moves agree on one
        axis move (diagonal moves give proportionally longer arrows) and shorter when they do not (not drawn
        below 5% of that). Its colour gives the mean move across the slab:
        blue toward lower, red toward higher values of the slicing axis, dark grey within the plane.
        (Default: None)
    policy_every
        Block width, in grid steps, of the policy arrows.
        (Default: None, so that about 15 blocks span the longer side of a panel)
    vmin, vmax
        Limits of the colour scale.
        (Default: None, from the finite values drawn)
    vcenter
        Value at the centre of the colour scale: the limits are made symmetric about it.
        (Default: None)
    robust
        Colour limits that leave out a few outlying values (see ``plot_grids``).
        (Default: True)
    label
        Colour bar label.
        (Default: None, the key)

    Returns
    ----------
    The matplotlib figure.
    """
    _require_3d(gres, 'plot_slices')
    cut, shown, slab_of, bounds = _sections(gres, axis, n_slices, dims)
    layers = _layers(gres, key, cells, list(range(np.asarray(gres.grids['grids']).shape[1])), rim=False)
    vmin, vmax = _value_limits(layers, vmin, vmax, vcenter, robust)
    allxy = np.vstack([c[:, shown] for c, _, _, _ in layers])
    limits = _limits(allxy)
    arrows = None
    if policy is not None:
        step = _auto_step(gres, shown) if policy_every is None else int(policy_every)
        tails = np.asarray(gres.grids['grids'])[np.asarray(gres.grids['mapped_travel'])]
        arrows = _block_field(gres, policy, shown, step, slab_of(tails)) + (step, _lattice_spacing(gres)[0])
    handles = _legend_handles(gres, layers)
    ncols = ncols or min(n_slices, 3)
    nrows = int(np.ceil(n_slices / ncols))
    if figsize is None:
        aspect = np.ptp(allxy[:, 1]) / np.ptp(allxy[:, 0])
        fig, axes, right = _panel_grid(_legend_width(handles) if handles else (1.6 if arrows is not None else 1.1),
                                       ncols, nrows, aspect, 2.8)
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes, right = list(axes.ravel()), .88
    if s is None:
        s = 4 if cells else 10
    mappable = None
    for k, ax in enumerate(axes):
        if k >= n_slices:
            ax.set_visible(False)
            continue
        m = _draw_section(ax, k, layers, cut, shown, slab_of, bounds, limits, cmap, vmin, vmax, s, arrows)
        mappable = m or mappable
    W = fig.get_figwidth()
    if handles:
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(right + .1 / W, .5), frameon=False)
    elif mappable is not None:
        top = .62 if arrows is not None else .2
        fig.colorbar(mappable, cax=fig.add_axes([right + .25 / W, top, .14 / W, .8 - top], label='<colorbar>'),
                     label=key if label is None else label, extend=_extend(layers, vmin, vmax))
    if arrows is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        cax = fig.add_axes([right + .35 / W, .4, .8 / W, .025], label='<colorbar>')
        bar = fig.colorbar(ScalarMappable(Normalize(-1, 1), _across_cmap()), cax=cax, orientation='horizontal')
        bar.set_label(f'move along Dim {cut + 1}', fontsize='small')
        bar.set_ticks([-1, 0, 1])
        bar.ax.tick_params(labelsize='small')
    return fig


def plot_views(gres, key=None, views=((20, -60), (20, 30), (90, -90)), cells=False, dims=(0, 1, 2),
               cmap='plasma', s=None, figsize=None, vmin=None, vmax=None, vcenter=None, robust=True, label=None,
               titles=None):
    """
    The 3-D view of ``plot_grids`` from several angles in one row, with one shared colour bar (or one
    cluster legend).

    Parameters
    ----------
    gres
        Grids results of a 3-D or higher grid.
    key
        Value to colour by (see ``plot_grids``).
        (Default: None)
    views
        (elevation, azimuth) of each panel.
        (Default: ((20, -60), (20, 30), (90, -90)))
    cells
        Draw the cells instead of the grid points.
        (Default: False)
    dims
        Embedding axes to draw.
        (Default: (0, 1, 2))
    cmap
        Colour map for the values.
        (Default: 'plasma')
    s
        Marker size.
        (Default: None)
    figsize
        Figure size.
        (Default: None, 4 inches per view)
    vmin, vmax
        Limits of the colour scale.
        (Default: None, from the finite values drawn)
    vcenter
        Value at the centre of the colour scale: the limits are made symmetric about it.
        (Default: None)
    robust
        Colour limits that leave out a few outlying values (see ``plot_grids``).
        (Default: True)
    label
        Colour bar label.
        (Default: None, the key)
    titles
        Panel titles: None gives the view angles in small grey type, a sequence gives one title per view,
        False gives none.
        (Default: None)

    Returns
    ----------
    The matplotlib figure.
    """
    _require_3d(gres, 'plot_views')
    n = len(views)
    layers = _layers(gres, key, cells, list(dims), False) if key is not None else []
    handles = _legend_handles(gres, layers) if key is not None else None
    vmin, vmax = _value_limits(layers, vmin, vmax, vcenter, robust)
    extra = _legend_width(handles) if handles else 1.2
    panel, gap = 3.8, .1
    fig = plt.figure(figsize=figsize or (n * panel + (n - 1) * gap + extra, 3.9))
    W = fig.get_figwidth()
    width = (W - extra - (n - 1) * gap) / n
    axes = [fig.add_axes([i * (width + gap) / W, .01, width / W, .9], projection='3d') for i in range(n)]
    if titles is None:
        names = [f'elev {e}\u00b0, azim {a}\u00b0'.replace('-', '\u2212') for e, a in views]
    elif titles is False:
        names = [None] * n
    else:
        names = list(titles)
    for ax, view, name in zip(axes, views, names):
        plot_grids(gres, key=key, cells=cells, dims=dims, view=view, ax=ax, cmap=cmap, s=s, colorbar=False,
                   vmin=vmin, vmax=vmax)
        if name:
            ax.set_title(name, fontsize='small', color='#666666' if titles is None else None)
    right = (W - extra) / W
    mappables = [c for c in axes[-1].collections if c.get_array() is not None]
    if handles:
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(right, .5), frameon=False)
    elif key is not None and mappables:
        fig.colorbar(mappables[-1], cax=fig.add_axes([right + .2 / W, .225, .12 / W, .55], label='<colorbar>'),
                     label=key if label is None else label, extend=_extend(layers, vmin, vmax))
    return fig


def _projection(gres, key, collapse, dims, how):
    """
    One orthogonal projection: (values or RGBA image, x edges, y edges, shown axes). Values are the max or mean
    over the travel points stacked along ``collapse`` (their number when ``key`` is None); positions without a
    travel point are masked.
    """
    shown = [d for d in dims if d != collapse]
    travel = np.asarray(gres.grids['mapped_travel'])
    c = _lattice_coords(gres, travel)[:, shown]
    lo, hi = c.min(0), c.max(0)
    nx, ny = hi - lo + 1
    flat = (c[:, 1] - lo[1]) * nx + (c[:, 0] - lo[0])
    spacing = _lattice_spacing(gres)[0]
    origin = np.asarray(gres.grids['grids'])[0]
    xe = origin[shown[0]] + (np.arange(lo[0], hi[0] + 2) - .5) * spacing[shown[0]]
    ye = origin[shown[1]] + (np.arange(lo[1], hi[1] + 2) - .5) * spacing[shown[1]]
    layers = _layers(gres, key, False, dims, False) if key is not None else []
    named = [layer for layer in layers if layer[3] is not None]
    if named:
        lut = dict(zip(np.asarray(named[0][3]).tolist(), named[0][1]))
        order = {name: i for i, name in enumerate(named[0][3].categories)}
        raw = (np.asarray(gres.grids['mapped_grids_clusters'])[_positions(gres, travel)] if key == 'clusters'
               else _raw_grid_values(gres, key, travel))
        present = ~pd.isna(np.asarray(raw, dtype=object))
        image = np.zeros((ny * nx, 4))
        frame = pd.DataFrame({'pix': flat[present], 'label': np.asarray(raw, dtype=object)[present].astype(str)})
        for pix, group in frame.groupby('pix'):
            counts = group['label'].value_counts()
            best = sorted(counts.index[counts == counts.max()], key=lambda n: order.get(n, len(order)))[0]
            image[pix, :3], image[pix, 3] = np.asarray(lut[best])[:3], 1
        return image.reshape(ny, nx, 4), xe, ye, shown
    values = np.ones(len(travel)) if key is None else _grid_values(gres, key, travel)
    if key is not None and np.isnan(values).all():
        raise ValueError(f'{key!r} has no values to draw: every value is NaN.')
    frame = pd.Series(values).groupby(flat)
    agg = frame.size() if key is None else (frame.max() if how == 'max' else frame.mean())
    grid = np.full(ny * nx, np.nan)
    grid[agg.index.to_numpy()] = agg.to_numpy(dtype=float)
    return np.ma.masked_invalid(grid.reshape(ny, nx)), xe, ye, shown


def _draw_projection(ax, data, collapse, how, key, cmap, vmin, vmax, title_size='medium', n_axes=3):
    image, xe, ye, shown = data
    if image.ndim == 3:
        ax.imshow(image, origin='lower', extent=(xe[0], xe[-1], ye[0], ye[-1]), interpolation='nearest')
        mappable = None
    else:
        mappable = ax.pcolormesh(xe, ye, image, cmap=cmap, vmin=vmin, vmax=vmax)
    if key is None:
        what = 'number of grid points'
    elif image.ndim == 3:
        what = 'most frequent cluster' if key == 'clusters' else 'most frequent category'
    else:
        what = how
    along = f'Dim {collapse + 1}' if n_axes <= 3 else 'the other axes'
    ax.set_title(f'Dim {shown[0] + 1}-Dim {shown[1] + 1} plane, {what} along {along}', fontsize=title_size)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f'Dim {shown[0] + 1}')
    ax.set_ylabel(f'Dim {shown[1] + 1}')
    return mappable


def plot_projections(gres, key=None, how='max', dims=(0, 1, 2), cmap='plasma', figsize=None, vmin=None, vmax=None,
                     vcenter=None, robust=True, label=None):
    """
    The grid collapsed along each of three axes onto the plane of the other two.

    Each panel is a raster with one pixel per lattice position of the plane. A pixel shows the maximum
    (``how='max'``) or mean (``how='mean'``) of the value over the travel points stacked along the collapsed
    axis; positions without a travel point are blank. All panels share one colour scale.

    Parameters
    ----------
    gres
        Grids results of a 3-D or higher grid.
    key
        Value to show (a key of ``gres.grids``). None shows the number of travel points along the collapsed
        axis, that is how thick the grid is there. ``'clusters'`` (or any key holding categories) shows the
        most frequent category, with a legend. NaN values are ignored; a pixel is blank when no value along
        the collapsed axis is a number.
        (Default: None)
    how
        'max' or 'mean'.
        (Default: 'max')
    dims
        The three embedding axes.
        (Default: (0, 1, 2))
    cmap
        Colour map for the values.
        (Default: 'plasma')
    figsize
        Figure size.
        (Default: None, from the data aspect)
    vmin, vmax
        Limits of the colour scale.
        (Default: None, from the values drawn)
    vcenter
        Value at the centre of the colour scale: the limits are made symmetric about it.
        (Default: None)
    robust
        Colour limits that leave out a few outlying values (see ``plot_grids``).
        (Default: True)
    label
        Colour bar label.
        (Default: None, the key)

    Returns
    ----------
    The matplotlib figure.
    """
    _require_3d(gres, 'plot_projections')
    if how not in ('max', 'mean'):
        raise ValueError(f'how must be "max" or "mean", got {how!r}')
    dims = list(dims)
    data = [_projection(gres, key, d, dims, how) for d in dims]
    categorical = data[0][0].ndim == 3
    handles = _legend_handles(gres, _layers(gres, key, False, dims, False)) if categorical else None
    if not categorical:
        drawn = [(None, None, np.concatenate([np.ma.compressed(p[0]) for p in data]), None)]
        vmin, vmax = _value_limits(drawn, vmin, vmax, vcenter, robust and key is not None)
    widths = [p[1][-1] - p[1][0] for p in data]
    heights = [p[2][-1] - p[2][0] for p in data]
    panel_h = 2.6
    panel_w = [panel_h * w / h for w, h in zip(widths, heights)]
    left, gap, extra, top, bottom = .45, .5, _legend_width(handles) if handles else 1.1, .55, .45
    W = left + sum(panel_w) + gap * 2 + extra
    H = top + panel_h + bottom
    fig = plt.figure(figsize=figsize or (W, H))
    W, H = fig.get_figwidth(), fig.get_figheight()
    x, mappable = left, None
    for d, p, w in zip(dims, data, panel_w):
        ax = fig.add_axes([x / W, bottom / H, w / W, panel_h / H])
        mappable = _draw_projection(ax, p, d, how, key, cmap, vmin, vmax, title_size='small',
                                    n_axes=np.asarray(gres.grids['grids']).shape[1]) or mappable
        x += w + gap
    right = (x - gap) / W
    if handles:
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(right + .1 / W, .5), frameon=False)
    else:
        fig.colorbar(mappable, cax=fig.add_axes([right + .25 / W, .2, .14 / W, .6], label='<colorbar>'),
                     label=label if label is not None else 'grid points' if key is None else key,
                     extend=_extend(drawn, vmin, vmax))
    return fig


def _values(gres, key, cells):
    """Cell values, or grid values at the travel points."""
    if cells:
        if key not in gres.embedding:
            raise KeyError(_missing_cell_key(gres, key))
        return np.asarray(gres.embedding[key], dtype=float).ravel()
    return _grid_values(gres, key, np.asarray(gres.grids['mapped_travel']))


def _clusters(gres, cells):
    """Cluster label of every cell, or of every travel point."""
    if cells:
        return np.asarray(gres.embedding['clusters'])
    travel = np.asarray(gres.grids['mapped_travel'])
    return np.asarray(gres.grids['mapped_grids_clusters'])[_positions(gres, travel)]


def plot_trend(gres, keys, x='pseudotime', bins=20, cells=True, labels=None, colors=None, ax=None, band=True,
               clusters=None, xlabel=None, ylabel=None):
    """
    Values along pseudotime (or another value): the median in equal-width bins, with the inter-quartile range.

    Parameters
    ----------
    gres
        Grids results.
    keys
        One key or a list of keys to draw.
    x
        Key of the value on the horizontal axis.
        (Default: 'pseudotime')
    bins
        Number of equal-width bins over the range of ``x``; empty bins are skipped.
        (Default: 20)
    cells
        Use cell values (``gres.embedding``); otherwise grid values at the travel points (``gres.grids``).
        (Default: True)
    labels
        Legend labels.
        (Default: None, the keys)
    colors
        Line colours.
        (Default: None)
    ax
        Axes to draw into.
        (Default: None, a new figure)
    band
        Shade the inter-quartile range.
        (Default: True)
    clusters
        Use only the cells (or grid points) of these clusters, to follow one lineage.
        (Default: None, all)
    xlabel, ylabel
        Axis labels.
        (Default: None, 'Pseudotime' (or the name of ``x``) and 'Value')

    Returns
    ----------
    The matplotlib axes.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    labels = keys if labels is None else list(labels)
    colors = [None] * len(keys) if colors is None else list(colors)
    xv = _values(gres, x, cells)
    keep = np.ones(len(xv), bool)
    if clusters is not None:
        clusters = [clusters] if np.isscalar(clusters) else list(clusters)
        names = _clusters(gres, cells)
        present = list(dict.fromkeys(names.tolist()))
        missing = [c for c in clusters if c not in present]
        if missing:
            raise ValueError(f'Cluster(s) {missing} not found. Clusters present: {present}')
        keep = np.isin(names, clusters)
    keep &= ~np.isnan(xv)
    xv = xv[keep]
    if ax is None:
        ax = plt.figure(figsize=(6.2, 4)).add_axes([.11, .12, .6, .8])
    edges = np.linspace(xv.min(), xv.max(), bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    k = np.minimum(((xv - xv.min()) / (xv.max() - xv.min()) * bins).astype(int), bins - 1)
    used = [b for b in range(bins) if (k == b).any()]
    for key, label, colour in zip(keys, labels, colors):
        y = _values(gres, key, cells)[keep]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            q = np.array([np.nanpercentile(y[k == b], [25, 50, 75]) for b in used])
        line, = ax.plot(centres[used], q[:, 1], label=label, color=colour)
        if band:
            ax.fill_between(centres[used], q[:, 0], q[:, 2], color=line.get_color(), alpha=.25, lw=0)
    ax.set_xlabel(xlabel if xlabel is not None else 'Pseudotime' if x == 'pseudotime' else x)
    ax.set_ylabel('Value' if ylabel is None else ylabel)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, borderaxespad=0)
    return ax


def plot_clusters(gres, key, cells=True, ax=None):
    """
    Box plot of a value per cluster, clusters ordered by median from highest to lowest.

    Parameters
    ----------
    gres
        Grids results after ``project_cluster``.
    key
        Value to summarise.
    cells
        Use cell values and cell clusters; otherwise grid values and grid clusters at the travel points.
        (Default: True)
    ax
        Axes to draw into.
        (Default: None, a new figure)

    Returns
    ----------
    The matplotlib axes.
    """
    values = _values(gres, key, cells)
    clusters = _clusters(gres, cells)
    order = pd.Series(values).groupby(clusters).median().dropna().sort_values(ascending=False).index
    if ax is None:
        names = [str(c) for c in order]
        W = max(4, .45 * len(order) + 1)
        bottom = .3 + .055 * max(len(n) for n in names) * np.sin(np.pi / 4) + .15
        H = 3 + bottom + .3
        ax = plt.figure(figsize=(W, H)).add_axes([.8 / W, bottom / H, (W - 1) / W, 3 / H])
    boxes = ax.boxplot([values[(clusters == c) & ~np.isnan(values)] for c in order], patch_artist=True,
                       showfliers=False)
    if 'cluster_colors' in gres.embedding:
        lut = dict(zip(np.asarray(gres.embedding['clusters']), gres.embedding['cluster_colors']))
        for patch, c in zip(boxes['boxes'], order):
            patch.set_facecolor(lut[c])
    ax.set_xticks(range(1, len(order) + 1), [str(c) for c in order], rotation=45, ha='right',
                  rotation_mode='anchor')
    ax.set_ylabel(key)
    return ax


def plot_policy(gres, trainer, dims=(0, 1, 2), view=(22, -58), ax=None, every=None, color=ARROW, key=None,
                cmap='plasma', interactive=False):
    """
    Arrows showing where the agent tends to move.

    ActorCritic and SAC: the expected move, the moves weighted by the actor's probabilities at the grid
    point's state. DDQN and TabularQ: the move with the largest Q-value. On 2-D grids an arrow is drawn at the
    travel points of a regular sub-lattice, as the move in embedding units. On 3-D and higher grids the moves
    are averaged over blocks of ``every`` grid steps along each drawn axis (blocks with at least 3 travel
    points; the other axes are collapsed): each arrow starts at the mean position of its block and is the mean
    move, 0.8 block widths long when all moves agree on one axis move (diagonal moves give proportionally
    longer arrows) and shorter when they do not (not drawn below 5% of that), over small faint grid points coloured by ``key`` (no points when ``key`` is None).

    Parameters
    ----------
    gres
        Grids results.
    trainer
        A trained ``scRL.trainer``; an integer array with one move index per kept grid point; or a 2-D array
        of move weights (kept grid points x moves), for example ``gres.grids['exact_<key>_best']``.
    dims
        Embedding axes to draw for 3-D and higher grids.
        (Default: (0, 1, 2))
    view
        Elevation and azimuth of the 3-D view.
        (Default: (22, -58))
    ax
        Axes to draw into; a 3-D axes for 3-D and higher grids.
        (Default: None, a new figure)
    every
        Sub-lattice step (2-D) or block width (3-D and higher), in grid steps.
        (Default: None, so that about 15 arrows span the longest side of the grid)
    color
        Arrow colour.
        (Default: dark grey)
    key
        Colour the grid points underneath by this stored value.
        (Default: None)
    cmap
        Colour map for ``key``.
        (Default: 'plasma')
    interactive
        Return a rotatable plotly figure with one cone per block (3-D and higher grids, needs plotly).
        (Default: False)

    Returns
    ----------
    The matplotlib axes, or a plotly figure when ``interactive`` is True.
    """
    three = np.asarray(gres.embedding['embedding']).shape[1] > 2
    dims = list(dims) if three else [0, 1]
    if interactive and not three:
        raise ValueError('interactive policy view needs a 3-D grid')
    step = _auto_step(gres, dims) if every is None else int(every)
    if not three:
        travel = _thinned_travel(gres, dims, step)
        arrows = _policy_moves(gres, trainer, travel)[:, dims]
        tails = np.asarray(gres.grids['grids'])[travel][:, dims]
        if key is not None:
            ax = plot_grids(gres, key=key, dims=dims, view=view, ax=ax, cmap=cmap)
        else:
            ax = plot_grids(gres, dims=dims, view=view, ax=ax)
            for c in ax.collections:
                c.set_color(GREY)
        ax.quiver(*tails.T, *arrows.T, color=color, angles='xy', scale_units='xy', scale=1, width=.003)
        return ax
    pos, mean, _ = _block_field(gres, trainer, dims, step)
    spacing = _lattice_spacing(gres)[0]
    moving = np.linalg.norm(mean[:, dims], axis=1) >= .05
    vectors = mean[moving][:, dims] * .8 * step * spacing[dims]
    pos = pos[moving][:, dims]
    if interactive:
        return _plotly_policy(gres, pos, vectors, key, dims, cmap)
    if ax is None:
        fig = plt.figure(figsize=(5.4, 4.6))
        ax = fig.add_axes([0, 0, 4.2 / 5.4, .95], projection='3d')
    if key is not None:
        layers = _layers(gres, key, False, dims, False)
        lo, hi = _value_limits(layers)
        mappable = None
        for layer in layers:
            artist = _scatter(ax, layer[0], layer[1], layer[2], 3 * layer.size, cmap, lo, hi, alpha=.35)
            mappable = artist if layer[2] is not None else mappable
        handles = _legend_handles(gres, layers)
        if handles:
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
        elif mappable is not None:
            ax.figure.colorbar(mappable, ax=ax, shrink=.55, pad=.02, label=key, extend=_extend(layers, lo, hi))
    ax.quiver(*pos.T, *vectors.T, color=color, lw=1.3, arrow_length_ratio=.45)
    _style_3d(ax, dims, view, np.asarray(gres.grids['grids'])[np.asarray(gres.grids['mapped_grids'])][:, dims])
    return ax


def _plotly_policy(gres, pos, vectors, key, dims, cmap):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('plot_policy(interactive=True) needs plotly: pip install plotly') from None
    from matplotlib.colors import to_hex
    traces = []
    for layer in (_layers(gres, key, False, dims, False) if key is not None else
                  [_layer(np.asarray(gres.grids['grids'])[np.asarray(gres.grids['mapped_travel'])][:, dims], GREY,
                          None, None)]):
        coords, colour, values, _ = layer
        if values is not None:
            marker = dict(size=3, color=values, colorscale=cmap, showscale=True, colorbar=dict(title=key))
        elif np.ndim(colour) > 1:
            marker = dict(size=3, color=[to_hex(c) for c in colour])
        else:
            marker = dict(size=3 * np.sqrt(layer.size), color=to_hex(colour))
        traces.append(go.Scatter3d(mode='markers', marker=marker, opacity=.35, hoverinfo='skip',
                                   **dict(zip('xyz', coords.T))))
    traces.append(go.Cone(**dict(zip('xyz', pos.T)), **dict(zip('uvw', vectors.T)), sizemode='raw', sizeref=1,
                          anchor='tail', colorscale=[[0, ARROW], [1, ARROW]], showscale=False))
    fig = go.Figure(traces)
    labels = {f'{a}axis_title': f'Dim {d + 1}' for a, d in zip('xyz', dims)}
    fig.update_layout(scene=dict(aspectmode='data', **labels), showlegend=False, template='plotly_white',
                      margin=dict(l=0, r=0, b=0, t=40))
    return fig


def save_rotation(gres, path, key=None, cells=False, frames=72, elevation=20, fps=12, dpi=100, **kwargs):
    """
    Save a turntable animation of the 3-D view of ``plot_grids``: the azimuth turns once round in ``frames``
    steps.

    Parameters
    ----------
    gres
        Grids results of a 3-D or higher grid.
    path
        Output file, '.gif' (written with Pillow) or '.mp4' (needs ffmpeg).
    key
        Value to colour by (see ``plot_grids``).
        (Default: None)
    cells
        Draw the cells instead of the grid points.
        (Default: False)
    frames
        Number of frames in one turn.
        (Default: 72)
    elevation
        Elevation of the view.
        (Default: 20)
    fps
        Frames per second.
        (Default: 12)
    dpi
        Resolution of the frames.
        (Default: 100)
    **kwargs
        Further arguments of ``plot_grids``.

    Returns
    ----------
    The path written.
    """
    from matplotlib import animation
    _require_3d(gres, 'save_rotation')
    suffix = str(path).lower().rsplit('.', 1)[-1]
    if suffix == 'gif':
        writer = animation.PillowWriter(fps=fps)
    elif suffix == 'mp4':
        if not animation.writers.is_available('ffmpeg'):
            raise RuntimeError('Writing .mp4 needs ffmpeg; install it or save a .gif.')
        writer = animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError(f'path must end in .gif or .mp4, got {path!r}')
    ax = _rotation_axes(gres, key, cells, elevation, **kwargs)
    try:
        def turn(i):
            ax.view_init(elevation, 360 * i / frames)
            return []

        anim = animation.FuncAnimation(ax.figure, turn, frames=frames, blit=False)
        anim.save(str(path), writer=writer, dpi=dpi)
    finally:
        plt.close(ax.figure)
    return path


def _rotation_axes(gres, key, cells, elevation, **kwargs):
    """The view turned by ``save_rotation``: fixed limits and box, no axis text besides the colour bar's."""
    for name in ('view', 'ax'):
        if name in kwargs:
            raise TypeError(f'save_rotation sets {name!r} itself; it cannot be passed on to plot_grids.')
    ax = plot_grids(gres, key=key, cells=cells, view=(elevation, 0), **kwargs)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.set_zlim(ax.get_zlim())
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_label_text('')
        axis.set_ticklabels([])
    return ax


def plot_report(gres, key, clusters=None, title=None, cmap='plasma', vmin=None, vmax=None, vcenter=None, robust=True,
                label=None):
    """
    One page of results for a value on a 3-D grid.

    Row 1: the 3-D view and the three projections (maximum along each axis). Row 2: six sections along the
    third axis. Rows 1 and 2 share one colour scale. Row 3: the value per cluster (grid points) and the value
    along pseudotime (grid points, restricted to ``clusters`` when given).

    Parameters
    ----------
    gres
        Grids results of a 3-D or higher grid.
    key
        Value to show (a key of ``gres.grids``).
    clusters
        Clusters for the trend panel, to follow one lineage.
        (Default: None, all)
    title
        Title of the page.
        (Default: None)
    cmap
        Colour map.
        (Default: 'plasma')
    vmin, vmax
        Limits of the colour scale of rows 1 and 2.
        (Default: None, from the finite values)
    vcenter
        Value at the centre of the colour scale: the limits are made symmetric about it.
        (Default: None)
    robust
        Colour limits that leave out a few outlying values (see ``plot_grids``).
        (Default: True)
    label
        Colour bar label.
        (Default: None, the key)

    Returns
    ----------
    The matplotlib figure.
    """
    _require_3d(gres, 'plot_report')
    dims = [0, 1, 2]
    if key == 'clusters' or key not in gres.grids or _is_categorical(gres.grids[key]):
        raise ValueError('plot_report needs a numeric key stored on the grid.')
    layers = _layers(gres, key, False, dims, False)
    vmin, vmax = _value_limits(layers, vmin, vmax, vcenter, robust)
    data = [_projection(gres, key, d, dims, 'max') for d in dims]
    cut, shown, slab_of, bounds = _sections(gres, 2, 6, dims)
    limits = _limits(np.vstack([c[:, shown] for c, _, _, _ in layers]))

    W, left, right_strip, gap = 12.0, .45, 1.2, .45
    proj_w = [p[1][-1] - p[1][0] for p in data]
    proj_h = [p[2][-1] - p[2][0] for p in data]
    width3d = 3.2
    avail = W - left - right_strip - width3d - 3 * gap
    scale = min(avail / sum(proj_w), 2.8 / max(proj_h))
    pw, ph = [w * scale for w in proj_w], [h * scale for h in proj_h]
    row1 = max(max(ph), 2.4)
    sw = (W - left - right_strip - 5 * .35) / 6
    sh = float(np.clip(sw * (limits[1][1] - limits[1][0]) / (limits[0][1] - limits[0][0]), .5, 2.5))
    row3 = 3.0
    longest = max(len(str(c)) for c in _cluster_order(gres)) if 'clusters' in gres.embedding else 3
    top = .55 + (.35 if title is not None else 0)
    bottom = .35 + .055 * longest
    H = top + row1 + .6 + sh + .6 + row3 + bottom
    fig = plt.figure(figsize=(W, H))
    if title is not None:
        fig.suptitle(title, y=1 - .3 / H)

    def rect(x, y_top, w, h):
        return [x / W, (H - y_top - h) / H, w / W, h / H]

    ax3 = fig.add_axes(rect(0, top - .2, width3d + .2, row1 + .3), projection='3d')
    plot_grids(gres, key=key, ax=ax3, cmap=cmap, colorbar=False, vmin=vmin, vmax=vmax)
    mappable, x = None, width3d + gap + left * .5
    for d, p, w, h in zip(dims, data, pw, ph):
        ax = fig.add_axes(rect(x, top + row1 - h, w, h))
        mappable = _draw_projection(ax, p, d, 'max', key, cmap, vmin, vmax, title_size='small') or mappable
        x += w + gap

    y2 = top + row1 + .6
    for k in range(6):
        ax = fig.add_axes(rect(left + k * (sw + .35), y2, sw, sh))
        _draw_section(ax, k, layers, cut, shown, slab_of, bounds, limits, cmap, vmin, vmax, 6, title_size='small')
    fig.colorbar(mappable, cax=fig.add_axes(rect(W - right_strip + .3, top + .2, .14, row1 + .6 + sh - .4),
                                            label='<colorbar>'),
                 label=key if label is None else label, extend=_extend(layers, vmin, vmax))

    y3 = y2 + sh + .6
    plot_clusters(gres, key, cells=False, ax=fig.add_axes(rect(.75, y3, 4.6, row3)))
    plot_trend(gres, key, cells=False, clusters=clusters, ax=fig.add_axes(rect(6.6, y3, 4.6, row3)))
    return fig


@contextlib.contextmanager
def plot_style(font='Arial', base_size=8):
    """
    Publication settings for figures drawn inside a ``with`` block; the previous settings are restored after it.

    Sets the font (falling back to DejaVu Sans), the base font size, embedded TrueType fonts in PDF and
    PostScript, text kept as text in SVG, thin axes lines and 300 dpi for saved figures.

    Parameters
    ----------
    font
        Font family.
        (Default: 'Arial')
    base_size
        Base font size in points.
        (Default: 8)
    """
    with plt.rc_context({'font.family': [font, 'DejaVu Sans'],
                         'font.size': base_size, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.linewidth': .6, 'savefig.dpi': 300}):
        yield


def plot_training(returns, values=None, window=100, ax=None):
    """
    Training curve: moving average of the episode returns, and optionally the value estimate.

    The trainers record one return per episode and one value estimate per step. When both have the same
    length the value estimate is drawn on a second y axis as a moving average over the same window, with each
    axis label in the colour of its line; otherwise it is drawn in a second panel against the step number, and
    the figure is returned.

    Parameters
    ----------
    returns
        Episode returns, as returned by ``trainer.train()``.
    values
        Value estimates recorded during training (second output of ``trainer.train()``).
        (Default: None)
    window
        Moving-average window in episodes.
        (Default: 100)
    ax
        Axes to draw into; not possible together with a value estimate of another length.
        (Default: None, a new figure)

    Returns
    ----------
    The matplotlib axes of the returns, or the figure when two panels are drawn.
    """
    from .utils import moving_average

    def smooth(a, w):
        if w % 2 == 1 and w < len(a):
            return moving_average(a, w), np.arange(1, len(a) + 1)
        w = max(1, min(w, len(a)))
        return np.convolve(a, np.ones(w) / w, mode='valid'), np.arange(w, len(a) + 1)

    r = np.asarray(returns, dtype=float)
    y, x = smooth(r, window)
    two_panels = values is not None and len(values) != len(r)
    if two_panels and ax is not None:
        raise ValueError('values has one entry per step, not per episode; it is drawn in a second panel, '
                         'so ax cannot be given.')
    if two_panels:
        fig, (ax, ax_v) = plt.subplots(1, 2, figsize=(9, 3.5))
        fig.subplots_adjust(wspace=.35, bottom=.17)
    elif ax is None:
        ax = plt.figure(figsize=(5, 3.5)).add_subplot()
    ax.plot(x, y, color='C0')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    if values is None:
        return ax
    v = np.asarray(values, dtype=float)
    if two_panels:
        ax_v.plot(np.arange(1, len(v) + 1), v, color='C1', lw=1)
        ax_v.set_xlabel('Step')
        ax_v.set_ylabel('Value estimate')
        return fig
    vs, xs = smooth(v, window)
    twin = ax.twinx()
    twin.plot(xs, vs, color='C1')
    twin.set_ylabel('Value estimate', color='C1')
    ax.set_ylabel('Return', color='C0')
    return ax


def _map_axes(fig, rect, three):
    return fig.add_axes(rect, projection='3d' if three else None)


def _draw_map(ax, gres, key, cells, dims, view, three, cmap='plasma', vmin=None, vmax=None, vcenter=None):
    """One map panel (3-D or flat) drawn through ``plot_grids`` without its own legend or colour bar."""
    plot_grids(gres, key=key, cells=cells, dims=dims, view=view, ax=ax, cmap=cmap, colorbar=False, vmin=vmin,
               vmax=vmax, vcenter=vcenter)
    mappables = [c for c in ax.collections if c.get_array() is not None]
    return mappables[-1] if mappables else None


def _weights(gres, key_add, cells):
    """Lineage weights (rows: kept grid points or cells) and the lineage names, from ``fate_gate`` or similar."""
    gate = gres.grids[key_add]
    names = list(gate.columns)
    if cells:
        W = np.column_stack([np.asarray(gres.embedding[f'{key_add}_{n}'], dtype=float) for n in names])
        clusters = np.asarray(gres.embedding['clusters']) if 'clusters' in gres.embedding else None
    else:
        W = gate.to_numpy(dtype=float)
        clusters = np.asarray(gres.grids['mapped_grids_clusters']) if 'mapped_grids_clusters' in gres.grids else None
    return W, names, clusters


def plot_fates(gres, key_add='gate', cells=False, dims=(0, 1, 2), view=(22, -58)):
    """
    Lineage weights on one page: the heaviest lineage, how undecided each point is, and the mean weights per
    cluster.

    Works with the results of ``fate_gate`` and ``fate_probabilities`` (``key_add`` as given there). Grid
    points (or cells) without a heaviest lineage are drawn grey ('none'); for lineage shares the per-cluster
    bars end with a grey segment for the walks that arrive at no listed lineage.

    Parameters
    ----------
    gres
        Grids results after ``fate_gate`` or ``fate_probabilities``.
    key_add
        Prefix of the stored lineage weights.
        (Default: 'gate')
    cells
        Draw the cells instead of the grid points.
        (Default: False)
    dims
        Embedding axes to draw for 3-D and higher grids.
        (Default: (0, 1, 2))
    view
        Elevation and azimuth of the 3-D panels.
        (Default: (22, -58))

    Returns
    ----------
    The matplotlib figure.
    """
    if key_add not in gres.grids:
        raise KeyError(f'No lineage weights under {key_add!r}: call fate_gate or fate_probabilities first.')
    three = np.asarray(gres.embedding['embedding']).shape[1] > 2
    W, names, clusters = _weights(gres, key_add, cells)
    fig = plt.figure(figsize=(14, 4.6) if three else (11.5, 4.2))
    ax1 = _map_axes(fig, [0, 0, .27, .93] if three else [.035, .1, .26, .8], three)
    _draw_map(ax1, gres, f'{key_add}_top', cells, dims, view, three)
    ax1.set_title('Heaviest lineage')
    handles = _legend_handles(gres, _layers(gres, f'{key_add}_top', cells, list(dims) if three else [0, 1],
                                            False))
    ax1.legend(handles=handles, loc='center left', bbox_to_anchor=(1 if three else 1.03, .5), frameon=False,
               fontsize='small')
    ax2 = _map_axes(fig, [.34, 0, .27, .93] if three else [.38, .1, .26, .8], three)
    mappable = _draw_map(ax2, gres, f'{key_add}_entropy', cells, dims, view, three, cmap='viridis', vmin=0, vmax=1)
    ax2.set_title('How undecided')
    fig.colorbar(mappable, cax=fig.add_axes([.615, .225, .01, .55] if three else [.645, .25, .011, .5],
                                            label='<colorbar>'), label='Undecided (0-1)')
    ax3 = fig.add_axes([.785, .13, .195, .75] if three else [.815, .12, .17, .76])
    if clusters is None:
        groups, order = np.array(['all'] * len(W)), ['all']
    else:
        groups = clusters
        order = [c for c in _cluster_order(gres) if c in set(groups.tolist())]
    colors = gres.grids.get(f'{key_add}_top_colors', {})
    keep = ~np.isnan(W).any(1)
    means = np.array([W[keep & (groups == c)].mean(0) if (keep & (groups == c)).any() else np.zeros(len(names))
                      for c in order])
    left = np.zeros(len(order))
    y = np.arange(len(order))[::-1]
    for i, n in enumerate(names):
        ax3.barh(y, means[:, i], left=left, color=colors.get(n), label=n, height=.8)
        left += means[:, i]
    if (1 - left > 1e-9).any():                        # walks that arrive at no listed lineage
        ax3.barh(y, np.clip(1 - left, 0, None), left=left, color=colors.get('none', GREY), label='none', height=.8)
    ax3.set_yticks(y, [str(c) for c in order])
    ax3.set_xlim(0, 1)
    ax3.set_xlabel('Mean lineage weight')
    ax3.set_ylabel('Cluster')
    ax3.set_title('Per cluster')
    ax3.spines[['top', 'right']].set_visible(False)
    return fig


def plot_conditions(gres, dims=(0, 1, 2), view=(22, -58)):
    """
    Where two conditions differ in their numbers of cells, on one page (after ``compare_abundance``).

    Panel 1: where each condition is (shared, one condition only, too few cells). Panel 2: the log2 fold change
    of the test against the reference condition where it was tested; points without a significant
    difference are drawn faint and small, untested points grey; the colour scale is symmetric about 0 and ends
    at the 99th percentile of the absolute fold changes (arrows on the colour bar mark larger ones). Panel 3: per cluster, the number of grid
    points called depleted, enriched or not different. Panel 4: per cluster, the grid points held by one
    condition only, observed (solid) against the number expected after shuffling the condition labels
    (hatched). The maps show the travel points; the bar panels count all kept grid points of the abundance
    table.

    Parameters
    ----------
    gres
        Grids results after ``compare_abundance``.
    dims
        Embedding axes to draw for 3-D and higher grids.
        (Default: (0, 1, 2))
    view
        Elevation and azimuth of the 3-D panels.
        (Default: (22, -58))

    Returns
    ----------
    The matplotlib figure.
    """
    if 'abundance' not in gres.grids:
        raise ValueError('No abundance comparison on the grid: call compare_abundance first.')
    three = np.asarray(gres.embedding['embedding']).shape[1] > 2
    dims = list(dims) if three else [0, 1]
    test, reference = gres.grids['abundance_test'], gres.grids['abundance_reference']
    span = np.ptp(np.asarray(gres.grids['grids'])[np.asarray(gres.grids['mapped_travel'])][:, dims], axis=0)
    top_h = float(np.clip(4.8 * (.5 + .5 * span[-1] / span[:2].max()), 3, 4.8)) if three else 4.6
    H = 4.5 + top_h + .3                               # the top row follows the height of the 3-D box
    fig = plt.figure(figsize=(13, H))
    y_top, h_top = 4.5 / H, top_h / H

    ax1 = _map_axes(fig, [0, y_top, .37, h_top] if three else [.05, y_top + .04, .3, h_top - .08], three)
    _draw_map(ax1, gres, 'abundance_class', False, dims, view, three)
    ax1.set_title('Where each condition is')
    ax1.legend(handles=_legend_handles(gres, _layers(gres, 'abundance_class', False, dims, False)),
               loc='center left', bbox_to_anchor=(1, .5), frameon=False, fontsize='small')

    ax2 = _map_axes(fig, [.5, y_top, .37, h_top] if three else [.53, y_top + .04, .3, h_top - .08], three)
    travel = np.asarray(gres.grids['mapped_travel'])
    coords = np.asarray(gres.grids['grids'])[travel][:, dims]
    log2fc = _grid_values(gres, 'abundance_log2fc', travel)
    call = _raw_grid_values(gres, 'abundance_call', travel).astype(str)
    finite = log2fc[~np.isnan(log2fc)]
    half = np.percentile(np.abs(finite), 99) if len(finite) else 1.0
    half = half if half > 0 else 1.0
    s = 8 if three else 12
    _scatter(ax2, coords[call == 'too few cells'], GREY, None, s / 2, None)
    quiet = call == 'no difference'
    called = np.isin(call, ['depleted', 'enriched'])
    _scatter(ax2, coords[quiet], None, log2fc[quiet], s / 2, 'RdBu_r', -half, half, alpha=.4)
    mappable = _scatter(ax2, coords[called], None, log2fc[called], s, 'RdBu_r', -half, half)
    if three:
        _style_3d(ax2, dims, view, coords)
    else:
        ax2.set_aspect('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel(f'Dim {dims[0] + 1}')
        ax2.set_ylabel(f'Dim {dims[1] + 1}')
    ax2.set_title('Change in cell numbers')
    beyond = np.abs(finite) > half
    extend = _extend([(None, None, finite, None)], -half, half) if beyond.any() else 'neither'
    fig.colorbar(mappable, cax=fig.add_axes([.89, y_top + .2 * h_top, .012, .55 * h_top], label='<colorbar>'),
                 label=f'log2 fold change ({test} / {reference})', extend=extend)

    clusters, order = (np.asarray(gres.grids['mapped_grids_clusters']), list(gres.grids['abundance_clusters'].index)) \
        if 'mapped_grids_clusters' in gres.grids else (np.array(['all'] * len(gres.grids['mapped_grids'])), ['all'])
    calls = np.asarray(gres.grids['abundance_call']).astype(str)
    call_colors = gres.grids['abundance_call_colors']
    y = np.arange(len(order))[::-1]
    width = max(len(str(c)) for c in order)
    left_margin = (.45 + .075 * width) / 13
    ax3 = fig.add_axes([left_margin + .02, .67 / H, .42 - left_margin, 3.17 / H])
    left = np.zeros(len(order))
    for name in ('depleted', 'enriched', 'no difference'):
        n = np.array([((clusters == c) & (calls == name)).sum() for c in order])
        ax3.barh(y, n, left=left, color=call_colors[name], label=name, height=.8)
        left += n
    ax3.set_yticks(y, [str(c) for c in order])
    ax3.set_ylabel('Cluster')
    ax3.set_xlabel('Grid points')
    ax3.set_title('Called per cluster')
    ax3.legend(frameon=False, fontsize='small', loc='lower right')
    ax3.spines[['top', 'right']].set_visible(False)

    summary = gres.grids['abundance_clusters']
    class_colors = gres.grids['abundance_class_colors']
    ax4 = fig.add_axes([.52 + left_margin, .67 / H, .46 - left_margin, 3.17 / H])
    h = .2
    for j, name in enumerate((reference, test)):
        label = f'{name} only'
        observed = summary[label].to_numpy(dtype=float)
        chance = summary[f'{label} by chance'].to_numpy(dtype=float)
        base = y + (.3 - j * .4)
        ax4.barh(base, observed, height=h, color=class_colors[label], label=f'{label}, observed')
        ax4.barh(base - h, chance, height=h, color='white', edgecolor=class_colors[label], hatch='////',
                 label=f'{label}, by chance')
    ax4.set_yticks(y, [str(c) for c in order])
    ax4.set_ylabel('Cluster')
    ax4.set_xlabel('Grid points held by one condition')
    ax4.set_title('Observed against label shuffling')
    ax4.legend(frameon=False, fontsize='small', loc='lower right')
    ax4.spines[['top', 'right']].set_visible(False)
    return fig


def _lattice_edges(gres):
    """Stored-position pairs of neighbouring kept grid points, each pair once: the moves of the decision problem."""
    from .GridNDCore import is_nd, lattice_from_scan
    lattice = gres if is_nd(gres) else lattice_from_scan(gres)
    nb = np.asarray(lattice.grids['neighbors'])
    p, a = np.nonzero(nb >= 0)
    q = nb[p, a]
    keep = p < q
    return np.column_stack([p[keep], q[keep]])


def _view_setup(gres, dims, ax):
    """Lattice coordinates, the axes drawn, whether the view is 3-D, and an axes to draw into."""
    three = np.asarray(gres.embedding['embedding']).shape[1] > 2
    dims = list(dims) if three else [0, 1]
    if ax is None:
        fig = plt.figure(figsize=(4.6, 4.6))
        ax = fig.add_subplot(projection='3d') if three else fig.add_subplot()
    elif three and ax.name != '3d':
        raise ValueError("A 3-D grid needs a 3-D axes, for example fig.add_subplot(projection='3d').")
    return np.asarray(gres.grids['grids'], dtype=float), dims, three, ax


def _finish_view(ax, three, dims, view, coords):
    if three:
        lo, hi = coords.min(0), coords.max(0)
        pad = .02 * np.maximum(hi - lo, 1e-9)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
        _style_3d(ax, dims, view, coords)
    else:
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(f'Dim {dims[0] + 1}')
        ax.set_ylabel(f'Dim {dims[1] + 1}')


def _lines(ax, three, segments, **kw):
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    if three:
        lines = Line3DCollection(segments, **kw)
        ax.add_collection3d(lines, autolim=False)                # limits are set from the drawn points
    else:
        lines = LineCollection(segments, **kw)
        ax.add_collection(lines)
    return lines


def plot_mesh(gres, key=None, dims=(0, 1, 2), view=(22, -58), ax=None, cmap='plasma', vmin=None, vmax=None,
              vcenter=None, robust=True, colorbar=True, label=None, s=None, lw=None, alpha=None, color_edges=False,
              edge_color='#9e9e9e', boundary=None):
    """
    Draw the grid as a mesh: a line for every move between two neighbouring kept grid points, and the grid points.

    The lines are the moves of the decision problem (one per move of the grid: six per interior point of a 3-D grid
    with axis moves, eight on a 2-D scan grid), so the drawing shows the space an episode or a walk can travel. Without a key, travel points are dark
    and boundary points light; with a key, the grid points are coloured by its value and, with ``color_edges``,
    each line by the mean value of its two ends.

    Parameters
    ----------
    gres
        Grids results.
    key
        Value of ``gres.grids`` to colour by (one value per kept grid point), or None.
        (Default: None)
    dims
        Embedding axes to draw for 3-D and higher grids.
        (Default: (0, 1, 2))
    view
        Elevation and azimuth of the 3-D view.
        (Default: (22, -58))
    ax
        Axes to draw into (a 3-D axes for 3-D grids). A new figure is made when None.
        (Default: None)
    cmap, vmin, vmax, vcenter, robust
        Colour map and colour limits, as in ``plot_grids``.
    colorbar
        Show a colour bar when a value is drawn.
        (Default: True)
    label
        Colour bar label.
        (Default: None, the key)
    s
        Marker size of the grid points (0 draws the lines only).
        (Default: None, 3 in 3-D and 8 in 2-D)
    lw, alpha
        Width and opacity of the lines.
        (Default: None, 0.25 and 0.45 in 3-D, 0.4 and 0.6 in 2-D)
    color_edges
        Colour the lines by the value (needs ``key``).
        (Default: False)
    edge_color
        Colour of the lines when they are not coloured by the value.
        (Default: '#9e9e9e')
    boundary
        How the boundary points are drawn: 'hide', 'grey' (light grey) or 'value' (coloured by the value like the
        travel points). None leaves them out when a value is drawn, as ``plot_grids`` does (a move onto a boundary
        point outside the start ends an episode, so no decision is taken there and the value describes the travel
        points), and draws them light grey without a value.
        (Default: None)

    Returns
    ----------
    The matplotlib axes.
    """
    boundary = _boundary_rule(boundary, key)
    G, dims, three, ax = _view_setup(gres, dims, ax)
    mapped = np.asarray(gres.grids['mapped_grids'])
    travel = np.isin(mapped, np.asarray(gres.grids['mapped_travel']))
    P = G[mapped][:, dims]
    E = _lattice_edges(gres)
    lw = (.25 if three else .4) if lw is None else lw
    alpha = (.45 if three else .6) if alpha is None else alpha
    s = (3 if three else 8) if s is None else s
    segments = np.stack([P[E[:, 0]], P[E[:, 1]]], axis=1)
    values = lo = hi = None
    if key is not None:
        values = _grid_values(gres, key, mapped)
        if boundary != 'value':
            values = np.where(travel, values, np.nan)
        ok = np.isfinite(values)
        if not ok.any():
            raise ValueError(f'{key!r} has no values to draw: every value is NaN.')
        layer = [_layer(P[ok], None, values[ok], None)]
        lo, hi = _value_limits(layer, vmin, vmax, vcenter, robust)
    elif color_edges:
        raise ValueError('color_edges needs a key to colour by.')
    if values is not None and color_edges:
        ends = np.column_stack([values[E[:, 0]], values[E[:, 1]]])
        has = np.isfinite(ends).any(1)
        if (~has).any():                                  # lines with no value at either end stay plain
            _lines(ax, three, segments[~has], colors=edge_color, linewidths=lw, alpha=alpha)
        with np.errstate(all='ignore'):
            mean_end = np.nanmean(np.where(np.isfinite(ends[has]), ends[has], np.nan), axis=1)
        lines = _lines(ax, three, segments[has], cmap=cmap, linewidths=lw, alpha=alpha)
        lines.set_array(mean_end)
        lines.set_clim(lo, hi)
    else:
        _lines(ax, three, segments, colors=edge_color, linewidths=lw, alpha=alpha)
    mappable = None
    if s > 0:
        if values is None:
            if boundary != 'hide':
                _scatter(ax, P[~travel], GREY, None, s, None)
            _scatter(ax, P[travel], DARK, None, s, None)
        else:
            grey = ~ok & travel if boundary == 'hide' else ~ok
            if grey.any():
                _scatter(ax, P[grey], GREY, None, s * .5, None)
            mappable = _scatter(ax, P[ok], None, values[ok], s, cmap, lo, hi)
    elif values is not None and color_edges:
        mappable = lines
    else:
        _scatter(ax, P, GREY, None, 0, None)                  # sets the data limits
    _finish_view(ax, three, dims, view, P)
    if colorbar and mappable is not None:
        ax.figure.colorbar(mappable, ax=ax, shrink=.6, pad=.02 if three else .05,
                           label=key if label is None else label,
                           extend=_extend([_layer(P[ok], None, values[ok], None)], lo, hi))
    return ax


def _boundary_rule(boundary, key):
    """How boundary points are drawn: None means 'hide' with a value and 'grey' without one."""
    if boundary is None:
        return 'grey' if key is None else 'hide'
    if boundary not in ('grey', 'value', 'hide'):
        raise ValueError("boundary is None, 'grey', 'value' or 'hide'.")
    return boundary


def _mean_by_length(paths, k=60):
    """Mean of several paths after resampling each to ``k`` points at equal fractions of its own length."""
    out = []
    for P in paths:
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        if cum[-1] <= 0:
            continue
        f = np.linspace(0, cum[-1], k)
        out.append(np.column_stack([np.interp(f, cum, P[:, j]) for j in range(P.shape[1])]))
    return np.mean(out, axis=0) if out else np.empty((0, paths[0].shape[1] if paths else 0))


def plot_trajectories(gres, keys, labels=None, colors=None, walks=True, mean=True, context='grid', dims=(0, 1, 2),
                      view=(22, -58), ax=None, max_walks=40, seed=0, lw=None, legend=True):
    """
    Draw stored trajectories of one or several lineages over the grid, in 2-D or 3-D.

    ``keys`` are the prefixes under which ``exact_trajectories`` (its ``key_add``) or ``get_traj_df`` (its ``key``)
    stored the paths (``gres.trajectory['<key>_idx']``, grid point indices). Each walk is drawn as a thin line, and
    the mean path — the mean of the walks after each is resampled to the same number of points along its own
    length — as a thick line with a white outline, a dot at its start and an arrow at its end, one colour per key.
    The mean path is a route only when the walks of a key follow one route; walks of one key that part towards
    different end clusters (for example rollouts of a learner trained on several lineages) average to a line that no
    walk takes, so draw them with ``mean=False`` or under separate keys. On 3-D axes the trajectories are drawn in
    front of the grid points (the axes' ``computed_zorder`` is turned off).

    Parameters
    ----------
    gres
        Grids results with stored trajectories.
    keys
        One prefix or a list of prefixes.
    labels
        Legend labels, one per key.
        (Default: None, the keys)
    colors
        Colours, one per key.
        (Default: None, a qualitative palette)
    walks
        Draw the individual walks.
        (Default: True)
    mean
        Draw the mean path of the walks.
        (Default: True)
    context
        'grid' (kept grid points in light grey), 'cells' (cells in light grey) or None.
        (Default: 'grid')
    dims
        Embedding axes to draw for 3-D and higher grids.
        (Default: (0, 1, 2))
    view
        Elevation and azimuth of the 3-D view.
        (Default: (22, -58))
    ax
        Axes to draw into (a 3-D axes for 3-D grids). A new figure is made when None.
        (Default: None)
    max_walks
        Largest number of walks drawn per key; more are thinned to this number at random with ``seed``.
        (Default: 40)
    seed
        Random seed for thinning the walks.
        (Default: 0)
    lw
        Width of the mean path; walks are drawn at a quarter of it.
        (Default: None, 2.2)
    legend
        Show a legend of the keys.
        (Default: True)

    Returns
    ----------
    The matplotlib axes.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    labels = list(keys) if labels is None else list(labels)
    if len(labels) != len(keys):
        raise ValueError(f'{len(labels)} labels for {len(keys)} keys.')
    colors = _palette(len(keys)) if colors is None else list(colors)
    if len(colors) < len(keys):
        raise ValueError(f'{len(colors)} colours for {len(keys)} keys.')
    for key in keys:
        if f'{key}_idx' not in gres.trajectory:
            raise KeyError(f'No trajectories under {key!r}: store them with exact_trajectories(key_add={key!r}) or '
                           f'get_traj_df(key={key!r}) first.')
    made = ax is None
    G, dims, three, ax = _view_setup(gres, dims, ax)
    if made and legend:                           # room for the legend on the right
        ax.figure.set_size_inches(6.0, 4.6)
        ax.set_position([.0, .05, .76, .9] if three else [.09, .1, .62, .82])
    if three:
        ax.computed_zorder = False                # trajectories stay in front of the grid points
    mapped = np.asarray(gres.grids['mapped_grids'])
    lw = 2.2 if lw is None else lw
    if context == 'grid':
        C = G[mapped][:, dims]
        _scatter(ax, C, GREY, None, 2 if three else 6, None, alpha=.6, zorder=1)
    elif context == 'cells':
        C = np.asarray(gres.embedding['embedding'], dtype=float)[:, dims]
        _scatter(ax, C, GREY, None, 1.5 if three else 4, None, alpha=.6, zorder=1)
    elif context is None:
        C = G[mapped][:, dims]
    else:
        raise ValueError("context must be 'grid', 'cells' or None.")
    rng = np.random.default_rng(seed)
    step = float(np.abs(_lattice_spacing(gres)[0][dims]).mean()) if three else None
    handles, drawn = [], [C]
    for key, label, colour in zip(keys, labels, colors):
        paths = [np.asarray(p, dtype=int) for p in gres.trajectory[f'{key}_idx']]
        paths = [p[p >= 0] for p in paths]
        paths = [G[p][:, dims] for p in paths if len(p) > 1]
        M = _mean_by_length(paths) if (mean and paths) else np.empty((0, len(dims)))
        if walks and paths:
            shown = paths if len(paths) <= max_walks else [paths[i] for i in
                                                           np.sort(rng.choice(len(paths), max_walks, replace=False))]
            _lines(ax, three, shown, colors=[colour], linewidths=lw / 3, alpha=.45, zorder=3, label=f'_walks {key}')
            drawn += shown
        if len(M) >= 2:
            ax.plot(*M.T, color='white', lw=lw + 1.8, solid_capstyle='round', zorder=4)
            ax.plot(*M.T, color=colour, lw=lw, solid_capstyle='round', zorder=5)
            ax.scatter(*M[0], s=lw * 9, color='black', lw=0, zorder=6, **({'depthshade': False} if three else {}))
            d = M[-1] - M[-2]
            if np.linalg.norm(d) > 0:
                if three:
                    d = d / np.linalg.norm(d) * 2 * step
                    ax.quiver(*M[-1], *d, color=colour, linewidth=lw * .8, arrow_length_ratio=.9, zorder=6)
                    drawn.append((M[-1] + d)[None, :])
                else:
                    ax.annotate('', xy=M[-1], xytext=M[-1] - d, arrowprops=dict(arrowstyle='-|>', color=colour,
                                                                                   lw=lw, mutation_scale=4 * lw))
            drawn.append(M)
        handles.append(Line2D([], [], color=colour, lw=2, label=label))
    _finish_view(ax, three, dims, view, np.vstack(drawn))
    if legend:
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02 if three else 1.04, 1), frameon=False,
                  borderaxespad=0)
    return ax


def plot_voxels(gres, key=None, cut=None, view=(22, -58), ax=None, cmap='plasma', vmin=None, vmax=None,
                vcenter=None, robust=True, colorbar=True, label=None, edgecolor=(0, 0, 0, .25), lw=.15, alpha=1.,
                shade=False, boundary=None):
    """
    Draw a 3-D grid as cubes, one per kept grid point, coloured by a stored value.

    Each cube is the cell of the regular lattice around a kept grid point, so the drawing shows the grid itself:
    which part of the embedding it covers and the value at every point. Only outer faces are visible; ``cut``
    removes the grid points beyond a fraction of one axis to open a section through the interior.

    Parameters
    ----------
    gres
        Grids results of a 3-D grid (for 4-D and 5-D grids use ``plot_grids`` with ``dims``).
    key
        Value of ``gres.grids`` to colour by (one value per kept grid point); None colours travel points dark and
        boundary points light.
        (Default: None)
    cut
        (axis, fraction): hide the grid points whose coordinate along embedding axis ``axis`` (0, 1 or 2) lies above
        that fraction (0 to 1) of the axis' range; for example (1, .5) opens the grid half-way along Dim 2.
        (Default: None)
    view
        Elevation and azimuth of the view.
        (Default: (22, -58))
    ax
        3-D axes to draw into. A new figure is made when None.
        (Default: None)
    cmap, vmin, vmax, vcenter, robust
        Colour map and colour limits, as in ``plot_grids``.
    colorbar
        Show a colour bar when a value is drawn.
        (Default: True)
    label
        Colour bar label.
        (Default: None, the key)
    edgecolor, lw
        Colour and width of the cube edges.
        (Default: translucent black, 0.15)
    alpha
        Opacity of the cubes.
        (Default: 1)
    shade
        Shade the faces by their orientation; the colours then no longer match the colour bar exactly.
        (Default: False)
    boundary
        How the boundary points are drawn: 'hide', 'grey' (light grey) or 'value' (coloured by the value like the
        travel points). None leaves them out when a value is drawn, as ``plot_grids`` does (a move onto a boundary
        point outside the start ends an episode, so no decision is taken there and the value describes the travel
        points), and draws them light grey without a value.
        (Default: None)

    Returns
    ----------
    The matplotlib axes.
    """
    _require_3d(gres, 'plot_voxels')
    boundary = _boundary_rule(boundary, key)
    shape = tuple(int(x) for x in gres.grids.get('shape', ()))
    if len(shape) != 3:
        raise ValueError('plot_voxels draws 3-D grids; for 4-D and 5-D grids use plot_grids with dims.')
    G, dims, three, ax = _view_setup(gres, (0, 1, 2), ax)
    mapped = np.asarray(gres.grids['mapped_grids'])
    filled = np.zeros(int(np.prod(shape)), bool)
    filled[mapped] = True
    is_travel = np.zeros(len(filled), bool)
    is_travel[np.asarray(gres.grids['mapped_travel'])] = True
    show = filled & is_travel if boundary == 'hide' else filled.copy()
    if cut is not None:
        axis, fraction = int(cut[0]), float(cut[1])
        if axis not in (0, 1, 2) or not 0 <= fraction <= 1:
            raise ValueError('cut is (axis, fraction) with axis 0, 1 or 2 and fraction between 0 and 1.')
        coord = G[:, axis]
        lo, hi = coord[mapped].min(), coord[mapped].max()
        show &= coord <= lo + fraction * (hi - lo)
    step, _ = _lattice_spacing(gres)
    corner = np.indices(np.array(shape) + 1).astype(float)
    X, Y, Z = (G[0, k] + (corner[k] - .5) * step[k] for k in range(3))
    colours = np.zeros(filled.shape + (4,))
    mappable = None
    if key is None:
        colours[filled & is_travel] = plt.matplotlib.colors.to_rgba(DARK, alpha)
        colours[filled & ~is_travel] = plt.matplotlib.colors.to_rgba(GREY, alpha)
    else:
        values = _grid_values(gres, key, mapped)
        if boundary != 'value':
            values = np.where(is_travel[mapped], values, np.nan)
        ok = np.isfinite(values)
        if not ok.any():
            raise ValueError(f'{key!r} has no values to draw: every value is NaN.')
        layer = [_layer(G[mapped][ok], None, values[ok], None)]
        lo_c, hi_c = _value_limits(layer, vmin, vmax, vcenter, robust)
        norm = plt.Normalize(lo_c, hi_c)
        cm = plt.get_cmap(cmap)
        full = np.full(len(filled), np.nan)
        full[mapped] = values
        rgba = cm(norm(np.nan_to_num(full, nan=lo_c)))
        rgba[~np.isfinite(full)] = plt.matplotlib.colors.to_rgba(GREY)
        rgba[:, 3] = alpha
        colours = rgba
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    ax.voxels(X, Y, Z, show.reshape(shape), facecolors=colours.reshape(shape + (4,)), edgecolor=edgecolor,
              linewidth=lw, shade=shade)
    _finish_view(ax, True, [0, 1, 2], view, G[mapped])
    if colorbar and mappable is not None:
        ax.figure.colorbar(mappable, ax=ax, shrink=.6, pad=.02, label=key if label is None else label,
                           extend=_extend(layer, lo_c, hi_c))
    return ax

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
    the legend order). ``size`` scales the marker area.
    """
    size = 1.0


def _layer(coords, colour, values, labels, size=1.0):
    layer = _Layer((coords, colour, values, labels))
    layer.size = size
    return layer


def _category_layer(gres, coords, labels, key, cells):
    """Layer coloured by category: the stored colour dict ``<key>_colors`` when present, else tab10/tab20/husl."""
    labels = np.asarray(labels).astype(str)
    present = sorted(set(labels))
    stored = (gres.embedding if cells else gres.grids).get(key + '_colors')
    if isinstance(stored, dict) and stored:
        stored = {str(k): v for k, v in stored.items()}
        order = [c for c in stored if c in present] + [c for c in present if c not in stored]
        missing = [c for c in order if c not in stored]
        lut = {**stored, **dict(zip(missing, _palette(len(missing))))}
    else:
        order = present
        lut = dict(zip(present, _palette(len(present))))
    from matplotlib.colors import to_rgba_array
    return _layer(coords, to_rgba_array([lut[c] for c in labels]), None, pd.Categorical(labels, categories=order))


def _palette(n):
    """Qualitative colours: tab10 up to 10, tab20 up to 20, husl beyond."""
    if n <= 10:
        return list(plt.get_cmap('tab10').colors[:n])
    if n <= 20:
        return list(plt.get_cmap('tab20').colors[:n])
    import seaborn as sns
    return list(sns.husl_palette(n))


def _value_layers(coords, values, key):
    """Layers for numeric values: the points without a value (NaN) first, small and grey, then the others."""
    missing = np.isnan(values)
    if missing.all():
        raise ValueError(f'{key!r} has no values to draw: every value is NaN.')
    layers = [_layer(coords[missing], GREY, None, None, .5)] if missing.any() else []
    return layers + [_layer(coords[~missing], None, values[~missing], None)]


def _layers(gres, key, cells, dims, rim):
    """Point sets to draw (see ``_Layer``)."""
    if cells:
        X = np.asarray(gres.embedding['embedding'])[:, dims]
        if key is None:
            return [_layer(X, GREY, None, None)]
        if key == 'clusters':
            return [_layer(X, np.asarray(gres.embedding['cluster_colors']), None,
                           pd.Categorical(np.asarray(gres.embedding['clusters']), categories=_cluster_order(gres)))]
        if key not in gres.embedding:
            raise KeyError(f'{key!r} is not stored for the cells; available keys: '
                           f'{sorted(map(str, gres.embedding))}')
        values = gres.embedding[key]
        if _is_categorical(values):
            return [_category_layer(gres, X, values, key, True)]
        return _value_layers(X, np.asarray(values, dtype=float), key)
    coords = np.asarray(gres.grids['grids'])
    travel = np.asarray(gres.grids['mapped_travel'])
    boundary = np.asarray(gres.grids['mapped_boundary'])
    if key is None:
        return [_layer(coords[boundary][:, dims], GREY, None, None), _layer(coords[travel][:, dims], DARK, None, None)]
    layers = [_layer(coords[boundary][:, dims], GREY, None, None)] if rim else []
    if key == 'clusters':
        p = _positions(gres, travel)
        return layers + [_layer(coords[travel][:, dims], np.asarray(gres.grids['mapped_grids_colors'])[p], None,
                                pd.Categorical(np.asarray(gres.grids['mapped_grids_clusters'])[p],
                                               categories=_cluster_order(gres)))]
    values = _raw_grid_values(gres, key, travel)
    if _is_categorical(values):
        return layers + [_category_layer(gres, coords[travel][:, dims], values, key, False)]
    return layers + _value_layers(coords[travel][:, dims], values.astype(float), key)


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


def _value_limits(layers, vmin=None, vmax=None, vcenter=None):
    """Colour limits from the finite values drawn; ``vcenter`` makes them symmetric about it; explicit limits win."""
    values = [v for _, _, v, _ in layers if v is not None]
    if not values:
        return vmin, vmax
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    lo, hi = finite.min(), finite.max()
    if vcenter is not None:
        half = max(abs(lo - vcenter), abs(hi - vcenter))
        lo, hi = vcenter - half, vcenter + half
    return (lo if vmin is None else vmin), (hi if vmax is None else vmax)


def _scatter(ax, coords, colour, values, s, cmap, vmin=None, vmax=None, **kw):
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
    if abs(view[0]) >= 85:                    # looking along the third axis: its name has no place
        ax.zaxis.label.set_visible(False)
        ax.set_zticks([])


def plot_grids(gres, key=None, cells=False, dims=(0, 1, 2), view=(22, -58), ax=None, cmap='plasma',
               s=None, rim=False, interactive=False, title=None, colorbar=True, vmin=None, vmax=None, vcenter=None):
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

    Returns
    ----------
    The matplotlib axes, or a plotly figure when ``interactive`` is True.
    """
    ndim = np.asarray(gres.embedding['embedding']).shape[1]
    three = ndim > 2
    dims = list(dims) if three else [0, 1]
    if interactive:
        keys = key if isinstance(key, (list, tuple)) else [key]
        return _plotly(gres, keys, cells, rim, three, cmap, s, title, colorbar, dims, vmin, vmax, vcenter)
    if isinstance(key, (list, tuple)):
        raise ValueError('A list of keys needs interactive=True; draw one key at a time otherwise.')
    layers = _layers(gres, key, cells, dims, rim)
    if ax is None:
        fig = plt.figure(figsize=(6.5, 5.5) if three else (5, 5))
        ax = fig.add_subplot(projection='3d' if three else None)
        if three:
            fig.subplots_adjust(left=0, right=.85, bottom=.02, top=.92)
    elif three and ax.name != '3d':
        raise ValueError(f'A {ndim}-D grid needs a 3-D axes, for example fig.add_subplot(projection=\'3d\').')
    if s is None:
        s = 4 if cells else (8 if three else 12)
    lo, hi = _value_limits(layers, vmin, vmax, vcenter)
    mappable = None
    for layer in layers:
        coords, colour, values, _ = layer
        artist = _scatter(ax, coords, colour, values, s * layer.size, cmap, lo, hi)
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
    handles = _legend_handles(gres, layers)
    if colorbar and handles:
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    elif colorbar and mappable is not None:
        ax.figure.colorbar(mappable, ax=ax, shrink=.6, pad=.02 if three else .05)
    if title is not None:
        ax.set_title(title)
    return ax


def _plotly(gres, keys, cells, rim, three, cmap, s, title, colorbar, dims, vmin=None, vmax=None, vcenter=None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('plot_grids(interactive=True) needs plotly: pip install plotly') from None
    from matplotlib.colors import to_hex

    def trace(coords, colour, values, visible=True, limits=(None, None), size=1.0):
        if values is not None:
            marker = dict(color=values, colorscale=cmap, showscale=colorbar, cmin=limits[0], cmax=limits[1])
        elif np.ndim(colour) > 1:
            marker = dict(color=[to_hex(c) for c in colour])
        else:
            marker = dict(color=to_hex(colour))
        marker['size'] = (np.sqrt(s) if s else 3) * np.sqrt(size)
        xyz = dict(zip('xyz', coords.T))
        return (go.Scatter3d(mode='markers', marker=marker, visible=visible, **xyz) if three
                else go.Scatter(mode='markers', marker=marker, visible=visible, **xyz))

    if len(keys) == 1:
        layers = _layers(gres, keys[0], cells, dims, rim)
        limits = _value_limits(layers, vmin, vmax, vcenter)
        traces = [trace(*layer[:3], limits=limits, size=layer.size) for layer in layers]
        fig = go.Figure(traces)
        fig.update_layout(title=title)
    else:
        traces = []
        for i, k in enumerate(keys):
            layers = _layers(gres, k, cells, dims, False)
            traces.append(trace(*layers[-1][:3], visible=i == 0, limits=_value_limits(layers, vmin, vmax, vcenter)))
        fig = go.Figure(traces)
        buttons = [dict(label=str(k), method='update',
                        args=[{'visible': [j == i for j in range(len(keys))]}, {'title': str(k)}])
                   for i, k in enumerate(keys)]
        fig.update_layout(title=str(keys[0]), updatemenus=[dict(buttons=buttons, x=0, y=1.12, xanchor='left')])
    labels = {f'{a}axis_title': f'Dim {d + 1}' for a, d in zip('xyz', dims)}
    if three:
        fig.update_layout(scene=dict(aspectmode='data', **labels))
    else:
        fig.update_layout(**labels)
        fig.update_yaxes(scaleanchor='x')
    fig.update_layout(showlegend=False)
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
    kept grid point (for example the best moves stored by ``exact_values``).
    """
    import torch
    from .GridNDCore import OFFSETS_2D
    offsets = np.asarray(gres.grids['offsets'] if 'offsets' in gres.grids else OFFSETS_2D, dtype=float)
    if isinstance(trainer, np.ndarray):
        if len(trainer) != len(gres.grids['mapped_grids']):
            raise ValueError('A move array needs one move index per kept grid point.')
        return offsets[trainer[_positions(gres, travel)].astype(int)] * _lattice_spacing(gres)[0]
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
        tails, moves, length = arrows
        here = slab_of(tails) == k
        inplane, across = moves[here][:, shown], moves[here][:, cut]
        size = np.linalg.norm(inplane, axis=1)
        mostly_across = np.abs(across) > size
        draw = ~mostly_across & (size > 0)
        if draw.any():
            uv = inplane[draw] / size[draw, None] * length
            ax.quiver(*tails[here][draw][:, shown].T, *uv.T, angles='xy', scale_units='xy', scale=1,
                      width=.005, headwidth=3.8, headlength=4, headaxislength=3.6, color=ARROW, alpha=.85, zorder=3)
        for sign, marker in ((1, '^'), (-1, 'v')):
            m = mostly_across & (np.sign(across) == sign)
            if m.any():
                ax.plot(*tails[here][m][:, shown].T, ls='', marker=marker, ms=4, mfc='white', mec=ARROW, mew=.8,
                        zorder=3)
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
    """Width in inches of a one-column legend of these entries at the default font size."""
    size = plt.rcParams['legend.fontsize']
    size = plt.rcParams['font.size'] if isinstance(size, str) else size
    return .6 + .62 * size / 72 * max(len(h.get_label()) for h in handles)


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
                ncols=None, figsize=None, policy=None, policy_every=None, vmin=None, vmax=None, vcenter=None):
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
        A trained ``scRL.trainer``, or an integer array of move indices (one per kept grid point): draw the
        direction the agent tends to move in, as in ``plot_policy``, at travel points on a regular sub-lattice
        of the slab. Moves mainly in the plane are drawn as
        arrows 0.8 sub-lattice steps long; moves mainly across the slab as small open markers, ``^`` towards
        higher and ``v`` towards lower values of the slicing axis.
        (Default: None)
    policy_every
        Sub-lattice step, in grid steps, of the policy arrows.
        (Default: None, so that about 15 arrows span the longer side of a panel)
    vmin, vmax
        Limits of the colour scale.
        (Default: None, from the finite values drawn)
    vcenter
        Value at the centre of the colour scale: the limits are made symmetric about it.
        (Default: None)

    Returns
    ----------
    The matplotlib figure.
    """
    _require_3d(gres, 'plot_slices')
    cut, shown, slab_of, bounds = _sections(gres, axis, n_slices, dims)
    layers = _layers(gres, key, cells, list(range(np.asarray(gres.grids['grids']).shape[1])), rim=False)
    vmin, vmax = _value_limits(layers, vmin, vmax, vcenter)
    allxy = np.vstack([c[:, shown] for c, _, _, _ in layers])
    limits = _limits(allxy)
    arrows = None
    if policy is not None:
        step = _auto_step(gres, shown) if policy_every is None else int(policy_every)
        travel = _thinned_travel(gres, shown, step)
        length = .8 * step * _lattice_spacing(gres)[0][shown].mean()
        arrows = (np.asarray(gres.grids['grids'])[travel], _policy_moves(gres, policy, travel), length)
    handles = _legend_handles(gres, layers)
    ncols = ncols or min(n_slices, 3)
    nrows = int(np.ceil(n_slices / ncols))
    if figsize is None:
        aspect = np.ptp(allxy[:, 1]) / np.ptp(allxy[:, 0])
        fig, axes, right = _panel_grid(_legend_width(handles) if handles else 1.1, ncols, nrows, aspect, 2.8)
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
        fig.colorbar(mappable, cax=fig.add_axes([right + .25 / W, .2, .14 / W, .6], label='<colorbar>'), label=key)
    return fig


def plot_views(gres, key=None, views=((20, -60), (20, 30), (90, -90)), cells=False, dims=(0, 1, 2),
               cmap='plasma', s=None, figsize=None, vmin=None, vmax=None, vcenter=None):
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

    Returns
    ----------
    The matplotlib figure.
    """
    _require_3d(gres, 'plot_views')
    n = len(views)
    layers = _layers(gres, key, cells, list(dims), False) if key is not None else []
    handles = _legend_handles(gres, layers) if key is not None else None
    vmin, vmax = _value_limits(layers, vmin, vmax, vcenter)
    extra = _legend_width(handles) if handles else 1.4
    fig = plt.figure(figsize=figsize or (4 * n + extra, 4))
    right = 1 - extra / fig.get_figwidth()
    axes = [fig.add_axes([i * right / n, .01, right / n, .9], projection='3d') for i in range(n)]
    for ax, view in zip(axes, views):
        plot_grids(gres, key=key, cells=cells, dims=dims, view=view, ax=ax, cmap=cmap, s=s, colorbar=False,
                   title=f'elevation {view[0]}, azimuth {view[1]}', vmin=vmin, vmax=vmax)
    mappables = [c for c in axes[-1].collections if c.get_array() is not None]
    if handles:
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(right, .5), frameon=False)
    elif key is not None and mappables:
        fig.colorbar(mappables[-1], cax=fig.add_axes([right + .2 / fig.get_figwidth(), .2, .015, .6], label='<colorbar>'), label=key)
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
    layer = _layers(gres, key, False, dims, False)[-1] if key is not None else None
    if layer is not None and layer[3] is not None:
        labels, colours = np.asarray(layer[3]), layer[1]
        lut = dict(zip(labels.tolist(), colours))
        order = {name: i for i, name in enumerate(layer[3].categories)}
        image = np.zeros((ny * nx, 4))
        frame = pd.DataFrame({'pix': flat, 'label': labels})
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


def _draw_projection(ax, data, collapse, how, key, cmap, vmin, vmax, title_size='medium'):
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
    ax.set_title(f'Dim {shown[0] + 1}-Dim {shown[1] + 1} plane, {what} along Dim {collapse + 1}',
                 fontsize=title_size)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f'Dim {shown[0] + 1}')
    ax.set_ylabel(f'Dim {shown[1] + 1}')
    return mappable


def plot_projections(gres, key=None, how='max', dims=(0, 1, 2), cmap='plasma', figsize=None, vmin=None, vmax=None,
                     vcenter=None):
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
        finite = [np.ma.compressed(p[0]) for p in data]
        vmin, vmax = _value_limits([(None, None, np.concatenate(finite), None)], vmin, vmax, vcenter)
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
        mappable = _draw_projection(ax, p, d, how, key, cmap, vmin, vmax, title_size='small') or mappable
        x += w + gap
    right = (x - gap) / W
    if handles:
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(right + .1 / W, .5), frameon=False)
    else:
        fig.colorbar(mappable, cax=fig.add_axes([right + .25 / W, .2, .14 / W, .6], label='<colorbar>'),
                     label='grid points' if key is None else key)
    return fig


def _values(gres, key, cells):
    """Cell values, or grid values at the travel points."""
    if cells:
        if key not in gres.embedding:
            raise KeyError(f'{key!r} is not stored for the cells; available keys: '
                           f'{sorted(map(str, gres.embedding))}')
        return np.asarray(gres.embedding[key], dtype=float).ravel()
    return _grid_values(gres, key, np.asarray(gres.grids['mapped_travel']))


def _clusters(gres, cells):
    """Cluster label of every cell, or of every travel point."""
    if cells:
        return np.asarray(gres.embedding['clusters'])
    travel = np.asarray(gres.grids['mapped_travel'])
    return np.asarray(gres.grids['mapped_grids_clusters'])[_positions(gres, travel)]


def plot_trend(gres, keys, x='pseudotime', bins=20, cells=True, labels=None, colors=None, ax=None, band=True,
               clusters=None):
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
        ax = plt.figure(figsize=(5, 4)).add_subplot()
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
    ax.set_xlabel(x)
    ax.set_ylabel('Value')
    ax.legend()
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
        ax = plt.figure(figsize=(max(4, .6 * len(order) + 1), 4)).add_subplot()
    boxes = ax.boxplot([values[(clusters == c) & ~np.isnan(values)] for c in order], patch_artist=True,
                       showfliers=False)
    if 'cluster_colors' in gres.embedding:
        lut = dict(zip(np.asarray(gres.embedding['clusters']), gres.embedding['cluster_colors']))
        for patch, c in zip(boxes['boxes'], order):
            patch.set_facecolor(lut[c])
    ax.set_xticks(range(1, len(order) + 1), [str(c) for c in order], rotation=45, ha='right')
    ax.set_ylabel(key)
    return ax


def plot_policy(gres, trainer, dims=(0, 1, 2), view=(22, -58), ax=None, every=None, color=ARROW, key=None,
                cmap='plasma', interactive=False):
    """
    Arrows showing where the agent tends to move, at travel grid points on a regular sub-lattice.

    ActorCritic and SAC: the expected move, the moves weighted by the actor's probabilities at the grid
    point's state. DDQN and TabularQ: the move with the largest Q-value. On 2-D grids the arrows are the
    moves in embedding units; on 3-D and higher grids they show the direction only, each 0.8 sub-lattice
    steps long, over small faint grid points coloured by ``key`` (no points when ``key`` is None).

    Parameters
    ----------
    gres
        Grids results.
    trainer
        A trained ``scRL.trainer``, or an integer array with one move index per kept grid point (for example
        ``gres.grids['exact_<key>_move']``).
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
        Sub-lattice step in grid steps: arrows are drawn at travel points whose lattice coordinates are
        multiples of ``every``.
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
        Return a rotatable plotly figure with one cone per arrow (3-D and higher grids, needs plotly).
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
    travel = _thinned_travel(gres, dims, step)
    arrows = _policy_moves(gres, trainer, travel)[:, dims]
    tails = np.asarray(gres.grids['grids'])[travel][:, dims]
    if not three:
        if key is not None:
            ax = plot_grids(gres, key=key, dims=dims, view=view, ax=ax, cmap=cmap)
        else:
            ax = plot_grids(gres, dims=dims, view=view, ax=ax)
            for c in ax.collections:
                c.set_color(GREY)
        ax.quiver(*tails.T, *arrows.T, color=color, angles='xy', scale_units='xy', scale=1, width=.003)
        return ax
    length = np.linalg.norm(arrows, axis=1)
    moving = length > 0
    unit = arrows[moving] / length[moving, None]
    size = .8 * step * _lattice_spacing(gres)[0][dims].mean()
    if interactive:
        return _plotly_policy(gres, tails[moving], unit, size, key, dims, cmap)
    if ax is None:
        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_subplot(projection='3d')
        fig.subplots_adjust(left=0, right=.85, bottom=.02, top=.92)
    if key is not None:
        points = _layers(gres, key, False, dims, False)[-1]
        mappable = _scatter(ax, points[0], points[1], points[2], 3, cmap, alpha=.35)
        if points[3] is None:
            ax.figure.colorbar(mappable, ax=ax, shrink=.6, pad=.02, label=key)
        else:
            ax.legend(handles=_legend_handles(gres, [points]), loc='upper left', bbox_to_anchor=(1.02, 1),
                      frameon=False)
    ax.quiver(*tails[moving].T, *(unit * size).T, color=color, lw=1.1, arrow_length_ratio=.45)
    _style_3d(ax, dims, view, np.asarray(gres.grids['grids'])[np.asarray(gres.grids['mapped_grids'])][:, dims])
    return ax


def _plotly_policy(gres, tails, unit, size, key, dims, cmap):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('plot_policy(interactive=True) needs plotly: pip install plotly') from None
    from matplotlib.colors import to_hex
    coords, colour, values, _ = _layers(gres, 'clusters' if key == 'clusters' else key, False, dims, False)[-1]
    if values is not None:
        marker = dict(size=3, color=values, colorscale=cmap, showscale=True, colorbar=dict(title=key))
    elif np.ndim(colour) > 1:
        marker = dict(size=3, color=[to_hex(c) for c in colour])
    else:
        marker = dict(size=3, color=to_hex(GREY))
    points = go.Scatter3d(mode='markers', marker=marker, opacity=.35, hoverinfo='skip',
                          **dict(zip('xyz', coords.T)))
    cones = go.Cone(**dict(zip('xyz', tails.T)), **dict(zip('uvw', unit.T)), sizemode='absolute',
                    sizeref=size, anchor='tail', colorscale=[[0, ARROW], [1, ARROW]], showscale=False)
    fig = go.Figure([points, cones])
    labels = {f'{a}axis_title': f'Dim {d + 1}' for a, d in zip('xyz', dims)}
    fig.update_layout(scene=dict(aspectmode='data', **labels), showlegend=False)
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
    ax = plot_grids(gres, key=key, cells=cells, view=(elevation, 0), **kwargs)

    def turn(i):
        ax.view_init(elevation, 360 * i / frames)
        return []

    anim = animation.FuncAnimation(ax.figure, turn, frames=frames, blit=False)
    anim.save(str(path), writer=writer, dpi=dpi)
    plt.close(ax.figure)
    return path


def plot_report(gres, key, clusters=None, title=None, cmap='plasma', vmin=None, vmax=None, vcenter=None):
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

    Returns
    ----------
    The matplotlib figure.
    """
    _require_3d(gres, 'plot_report')
    dims = [0, 1, 2]
    vmin, vmax = _value_limits(_layers(gres, key, False, dims, False), vmin, vmax, vcenter)
    fig = plt.figure(figsize=(12, 13))
    if title is not None:
        fig.suptitle(title, y=.985)

    ax3 = fig.add_axes([0, .66, .29, .31], projection='3d')
    plot_grids(gres, key=key, ax=ax3, cmap=cmap, colorbar=False, vmin=vmin, vmax=vmax)
    mappable = None
    for i, d in enumerate(dims):
        ax = fig.add_axes([.31 + i * .21, .72, .17, .21])
        mappable = _draw_projection(ax, _projection(gres, key, d, dims, 'max'), d, 'max', key, cmap, vmin, vmax,
                                    title_size='small') or mappable

    cut, shown, slab_of, bounds = _sections(gres, 2, 6, dims)
    layers = _layers(gres, key, False, dims, False)
    limits = _limits(np.vstack([c[:, shown] for c, _, _, _ in layers]))
    for k in range(6):
        ax = fig.add_axes([.04 + k * .152, .42, .125, .2])
        _draw_section(ax, k, layers, cut, shown, slab_of, bounds, limits, cmap, vmin, vmax, 6, title_size='small')
    fig.colorbar(mappable, cax=fig.add_axes([.925, .46, .012, .45], label='<colorbar>'), label=key)

    plot_clusters(gres, key, cells=False, ax=fig.add_axes([.07, .07, .4, .27]))
    plot_trend(gres, key, cells=False, clusters=clusters, ax=fig.add_axes([.57, .07, .38, .27]))
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
    length the value estimate is drawn on a second y axis; otherwise it is drawn in a second panel against
    the step number, and the figure is returned.

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
    r = np.asarray(returns, dtype=float)
    if window % 2 == 1 and window < len(r):
        y, x = moving_average(r, window), np.arange(1, len(r) + 1)
    else:
        w = min(window, len(r))
        y = np.convolve(r, np.ones(w) / w, mode='valid')
        x = np.arange(w, len(r) + 1)
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
    twin = ax.twinx()
    twin.plot(np.arange(1, len(v) + 1), v, color='C1', lw=1)
    twin.set_ylabel('Value estimate')
    return ax

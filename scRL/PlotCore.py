"""Views of a grid and of the values stored on it."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .GridNDCore import _positions

GREY, DARK = '#d9d9d9', '#4d4d4d'


def _grid_values(gres, key, travel):
    """Values of ``gres.grids[key]`` at the travel points, in the order of ``travel``."""
    if key not in gres.grids:
        raise KeyError(f'{key!r} is not stored on the grid; available keys: {sorted(map(str, gres.grids))}')
    values = gres.grids[key]
    mapped = np.asarray(gres.grids['mapped_grids'])
    if isinstance(values, pd.Series) and len(values) == len(mapped) and set(values.index) == set(mapped):
        return values.loc[travel].to_numpy(dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    if len(values) != len(mapped):
        raise ValueError(f'gres.grids[{key!r}] has {len(values)} values but the grid has {len(mapped)} points.')
    return values[_positions(gres, travel)]


def _cluster_order(gres):
    """Cluster names in their categorical order."""
    clusters = gres.embedding['clusters']
    if hasattr(clusters, 'cat'):
        return list(clusters.cat.categories)
    return list(dict.fromkeys(np.asarray(clusters).tolist()))


def _layers(gres, key, cells, dims, rim):
    """
    Point sets to draw: (coordinates, colour, values, labels). ``colour`` is one colour or one per point;
    ``values`` are numbers for a colour map; ``labels`` are the cluster names when colouring by cluster.
    """
    if cells:
        X = np.asarray(gres.embedding['embedding'])[:, dims]
        if key is None:
            return [(X, GREY, None, None)]
        if key == 'clusters':
            return [(X, np.asarray(gres.embedding['cluster_colors']), None, np.asarray(gres.embedding['clusters']))]
        if key not in gres.embedding:
            raise KeyError(f'{key!r} is not stored for the cells; available keys: '
                           f'{sorted(map(str, gres.embedding))}')
        return [(X, None, np.asarray(gres.embedding[key], dtype=float), None)]
    coords = np.asarray(gres.grids['grids'])
    travel = np.asarray(gres.grids['mapped_travel'])
    boundary = np.asarray(gres.grids['mapped_boundary'])
    if key is None:
        return [(coords[boundary][:, dims], GREY, None, None), (coords[travel][:, dims], DARK, None, None)]
    layers = [(coords[boundary][:, dims], GREY, None, None)] if rim else []
    if key == 'clusters':
        p = _positions(gres, travel)
        return layers + [(coords[travel][:, dims], np.asarray(gres.grids['mapped_grids_colors'])[p], None,
                          np.asarray(gres.grids['mapped_grids_clusters'])[p])]
    return layers + [(coords[travel][:, dims], None, _grid_values(gres, key, travel), None)]


def _legend_handles(gres, layers):
    """One legend entry per cluster drawn, in cluster order, or None when no layer is coloured by cluster."""
    drawn = [(c, l) for _, c, _, l in layers if l is not None]
    if not drawn:
        return None
    lut = {}
    for colours, labels in drawn:
        for colour, label in zip(colours, labels):
            lut.setdefault(label, colour)
    return [Line2D([], [], ls='', marker='o', color=lut[c], label=str(c)) for c in _cluster_order(gres) if c in lut]


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
               s=None, rim=False, interactive=False, title=None, colorbar=True):
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
        ``'clusters'`` colours by cluster, with a legend. None shows the grid structure: travel points dark
        grey, boundary points light grey. With ``interactive=True`` a list of keys gives one trace per key
        and a menu to switch between them.
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
        Show a colour bar when a value is drawn (a legend for clusters).
        (Default: True)

    Returns
    ----------
    The matplotlib axes, or a plotly figure when ``interactive`` is True.
    """
    ndim = np.asarray(gres.embedding['embedding']).shape[1]
    three = ndim > 2
    dims = list(dims) if three else [0, 1]
    if interactive:
        keys = key if isinstance(key, (list, tuple)) else [key]
        return _plotly(gres, keys, cells, rim, three, cmap, s, title, colorbar, dims)
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
    mappable = None
    for coords, colour, values, _ in layers:
        artist = _scatter(ax, coords, colour, values, s, cmap)
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


def _plotly(gres, keys, cells, rim, three, cmap, s, title, colorbar, dims):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('plot_grids(interactive=True) needs plotly: pip install plotly') from None
    from matplotlib.colors import to_hex

    def trace(coords, colour, values, visible=True):
        if values is not None:
            marker = dict(color=values, colorscale=cmap, showscale=colorbar)
        elif np.ndim(colour) > 1:
            marker = dict(color=[to_hex(c) for c in colour])
        else:
            marker = dict(color=to_hex(colour))
        marker['size'] = np.sqrt(s) if s else 3
        xyz = dict(zip('xyz', coords.T))
        return (go.Scatter3d(mode='markers', marker=marker, visible=visible, **xyz) if three
                else go.Scatter(mode='markers', marker=marker, visible=visible, **xyz))

    if len(keys) == 1:
        traces = [trace(c, colour, v) for c, colour, v, _ in _layers(gres, keys[0], cells, dims, rim)]
        fig = go.Figure(traces)
        fig.update_layout(title=title)
    else:
        traces = [trace(*_layers(gres, k, cells, dims, False)[-1][:3], visible=i == 0) for i, k in enumerate(keys)]
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

    ActorCritic: the expected move, the moves weighted by the actor's probabilities at the grid point's
    state. DDQN and TabularQ: the move with the largest Q-value.
    """
    import torch
    from .GridNDCore import OFFSETS_2D
    offsets = np.asarray(gres.grids['offsets'] if 'offsets' in gres.grids else OFFSETS_2D, dtype=float)
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


def _default_every(gres):
    """Step through the travel points so that about 400 arrows are drawn."""
    return max(1, int(round(len(gres.grids['mapped_travel']) / 400)))


def plot_slices(gres, key=None, axis=2, n_slices=6, cells=False, dims=(0, 1, 2), cmap='plasma', s=None,
                ncols=None, figsize=None, policy=None):
    """
    Cut a 3-D (or higher) grid into slabs along one axis and draw each slab as a 2-D section.

    The slabs have equal width in grid steps along ``dims[axis]``, so every kept grid point (or every cell,
    with ``cells=True``) falls in exactly one slab. All panels share the axis limits and one colour scale;
    each panel title gives the slab's coordinate range.

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
        Figure size.
        (Default: None, from the number of panels)
    policy
        A trained ``scRL.trainer``: draw at each travel point of a slab the in-plane part of the move the
        agent tends to make (as in ``plot_policy``, in embedding units). A move mainly across the slab is
        drawn as a marker instead: ``^`` towards higher, ``v`` towards lower values of the slicing axis.
        (Default: None)

    Returns
    ----------
    The matplotlib figure.
    """
    _require_3d(gres, 'plot_slices')
    dims = list(dims)
    cut, shown = dims[axis], [d for i, d in enumerate(dims) if i != axis]
    spacing, shape = _lattice_spacing(gres)
    origin = np.asarray(gres.grids['grids'])[0]
    kept = np.unravel_index(np.asarray(gres.grids['mapped_grids']), shape)[cut]
    lo, width = kept.min() - .5, (kept.max() - kept.min() + 1) / n_slices

    def slab_of(coords):
        u = np.rint((coords[:, cut] - origin[cut]) / spacing[cut])
        return np.clip(np.floor((u - lo) / width), 0, n_slices - 1)

    layers = _layers(gres, key, cells, list(range(len(shape))), rim=False)
    values = [v for _, _, v, _ in layers if v is not None]
    vmin, vmax = (min(v.min() for v in values), max(v.max() for v in values)) if values else (None, None)
    allxy = np.vstack([c[:, shown] for c, _, _, _ in layers])
    pad = .03 * np.ptp(allxy, axis=0)
    limits = list(zip(allxy.min(0) - pad, allxy.max(0) + pad))
    if policy is not None:
        travel = np.asarray(gres.grids['mapped_travel'])
        tails = np.asarray(gres.grids['grids'])[travel]
        moves, tail_slab = _policy_moves(gres, policy, travel), slab_of(tails)
    ncols = ncols or min(n_slices, 3)
    nrows = int(np.ceil(n_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (3 * ncols + 1.5, 3.2 * nrows), squeeze=False)
    axes = axes.ravel()
    if s is None:
        s = 4 if cells else 10
    mappable = None
    for k, ax in enumerate(axes):
        if k >= n_slices:
            ax.set_visible(False)
            continue
        for coords, colour, vals, _ in layers:
            slab = slab_of(coords) == k
            colour_k = colour[slab] if np.ndim(colour) > 1 else colour
            artist = _scatter(ax, coords[slab][:, shown], colour_k, None if vals is None else vals[slab], s, cmap,
                              vmin, vmax)
            if vals is not None:
                mappable = artist
        if policy is not None:
            here = tail_slab == k
            inplane, across = moves[here][:, shown], moves[here][:, cut]
            mostly_across = np.abs(across) > np.linalg.norm(inplane, axis=1)
            arrow = ~mostly_across & (np.linalg.norm(inplane, axis=1) > 0)
            if arrow.any():
                ax.quiver(*tails[here][arrow][:, shown].T, *inplane[arrow].T, angles='xy', scale_units='xy',
                          scale=1, width=.006, color='k', zorder=3)
            for sign, marker in ((1, '^'), (-1, 'v')):
                m = mostly_across & (np.sign(across) == sign)
                if m.any():
                    ax.plot(*tails[here][m][:, shown].T, ls='', marker=marker, ms=3, color='k', zorder=3)
        a, b = origin[cut] + (lo + k * width) * spacing[cut], origin[cut] + (lo + (k + 1) * width) * spacing[cut]
        ax.set_title(f'Dim {cut + 1}: {a:.2f} to {b:.2f}', fontsize='medium')
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'Dim {shown[0] + 1}')
        ax.set_ylabel(f'Dim {shown[1] + 1}')
    handles = _legend_handles(gres, layers)
    if handles:
        fig.subplots_adjust(right=.8)
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.81, .5), frameon=False)
    elif mappable is not None:
        fig.colorbar(mappable, ax=axes[:n_slices].tolist(), shrink=.8, label=key)
    return fig


def plot_views(gres, key=None, views=((20, -60), (20, 30), (90, -90)), cells=False, dims=(0, 1, 2),
               cmap='plasma', s=None, figsize=None):
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
        (Default: None, 4.5 inches per view)

    Returns
    ----------
    The matplotlib figure.
    """
    _require_3d(gres, 'plot_views')
    n = len(views)
    fig = plt.figure(figsize=figsize or (4.5 * n + 1.5, 4.8))
    right = 1 - 1.5 / fig.get_figwidth()
    axes = [fig.add_axes([i * right / n, .02, right / n, .88], projection='3d') for i in range(n)]
    for ax, view in zip(axes, views):
        plot_grids(gres, key=key, cells=cells, dims=dims, view=view, ax=ax, cmap=cmap, s=s, colorbar=False,
                   title=f'elevation {view[0]}, azimuth {view[1]}')
    handles = _legend_handles(gres, _layers(gres, key, cells, list(dims), False)) if key == 'clusters' else None
    mappables = [c for c in axes[-1].collections if c.get_array() is not None]
    if handles:
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(right, .5), frameon=False)
    elif key is not None and mappables:
        fig.colorbar(mappables[-1], cax=fig.add_axes([right + .2 / fig.get_figwidth(), .2, .015, .6]), label=key)
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
    xv = xv[keep]
    if ax is None:
        ax = plt.figure(figsize=(5, 4)).add_subplot()
    edges = np.linspace(xv.min(), xv.max(), bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    k = np.minimum(((xv - xv.min()) / (xv.max() - xv.min()) * bins).astype(int), bins - 1)
    used = [b for b in range(bins) if (k == b).any()]
    for key, label, colour in zip(keys, labels, colors):
        y = _values(gres, key, cells)[keep]
        q = np.array([np.percentile(y[k == b], [25, 50, 75]) for b in used])
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
    order = pd.Series(values).groupby(clusters).median().sort_values(ascending=False).index
    if ax is None:
        ax = plt.figure(figsize=(max(4, .6 * len(order) + 1), 4)).add_subplot()
    boxes = ax.boxplot([values[clusters == c] for c in order], patch_artist=True, showfliers=False)
    if 'cluster_colors' in gres.embedding:
        lut = dict(zip(np.asarray(gres.embedding['clusters']), gres.embedding['cluster_colors']))
        for patch, c in zip(boxes['boxes'], order):
            patch.set_facecolor(lut[c])
    ax.set_xticks(range(1, len(order) + 1), [str(c) for c in order], rotation=45, ha='right')
    ax.set_ylabel(key)
    return ax


def plot_policy(gres, trainer, dims=(0, 1, 2), view=(22, -58), ax=None, every=None, color='k', key=None,
                cmap='plasma'):
    """
    Arrows showing where the agent tends to move from the travel grid points.

    ActorCritic: the expected move, the moves weighted by the actor's probabilities at the grid point's
    state. DDQN and TabularQ: the move with the largest Q-value. On 2-D grids the arrows are the moves in
    embedding units; on 3-D and higher grids they show the direction only, each 0.8 grid steps long, over
    small faint grid points coloured by ``key`` (no points when ``key`` is None).

    Parameters
    ----------
    gres
        Grids results.
    trainer
        A trained ``scRL.trainer``.
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
        Draw an arrow at every ``every``-th travel point.
        (Default: None, so that about 400 arrows are drawn)
    color
        Arrow colour.
        (Default: 'k')
    key
        Colour the grid points underneath by this stored value.
        (Default: None)
    cmap
        Colour map for ``key``.
        (Default: 'plasma')

    Returns
    ----------
    The matplotlib axes.
    """
    three = np.asarray(gres.embedding['embedding']).shape[1] > 2
    dims = list(dims) if three else [0, 1]
    every = _default_every(gres) if every is None else every
    travel = np.asarray(gres.grids['mapped_travel'])[::every]
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
    step = _lattice_spacing(gres)[0][dims].mean()
    length = np.linalg.norm(arrows, axis=1)
    moving = length > 0
    shown = arrows[moving] / length[moving, None] * .8 * step
    ax.quiver(*tails[moving].T, *shown.T, color=color, lw=1.2, arrow_length_ratio=.35)
    _style_3d(ax, dims, view, np.asarray(gres.grids['grids'])[np.asarray(gres.grids['mapped_grids'])][:, dims])
    return ax


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

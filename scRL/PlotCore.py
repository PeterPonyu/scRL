"""Views of a grid and of the values stored on it."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _layers(gres, key, cells, dims, rim):
    """Point sets to draw: (coordinates, fixed colour or None, values or None)."""
    if cells:
        X = np.asarray(gres.embedding['embedding'])[:, dims]
        if key is None:
            return [(X, GREY, None)]
        if key not in gres.embedding:
            raise KeyError(f'{key!r} is not stored for the cells; available keys: '
                           f'{sorted(map(str, gres.embedding))}')
        return [(X, None, np.asarray(gres.embedding[key], dtype=float))]
    coords = np.asarray(gres.grids['grids'])
    travel = np.asarray(gres.grids['mapped_travel'])
    boundary = np.asarray(gres.grids['mapped_boundary'])
    if key is None:
        return [(coords[boundary][:, dims], GREY, None), (coords[travel][:, dims], DARK, None)]
    layers = [(coords[boundary][:, dims], GREY, None)] if rim else []
    return layers + [(coords[travel][:, dims], None, _grid_values(gres, key, travel))]


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
        None shows the grid structure: travel points dark grey, boundary points light grey.
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
        Show a colour bar when a value is drawn.
        (Default: True)

    Returns
    ----------
    The matplotlib axes, or a plotly figure when ``interactive`` is True.
    """
    ndim = np.asarray(gres.embedding['embedding']).shape[1]
    three = ndim > 2
    dims = list(dims) if three else [0, 1]
    layers = _layers(gres, key, cells, dims, rim)
    if interactive:
        return _plotly(layers, three, cmap, s, title, colorbar, dims)
    if ax is None:
        fig = plt.figure(figsize=(6, 5) if three else (5, 5))
        ax = fig.add_subplot(projection='3d' if three else None)
    elif three and ax.name != '3d':
        raise ValueError(f'A {ndim}-D grid needs a 3-D axes, for example fig.add_subplot(projection=\'3d\').')
    if s is None:
        s = 4 if cells else (8 if three else 12)
    mappable = None
    for coords, colour, values in layers:
        kw = dict(color=colour) if values is None else dict(c=values, cmap=cmap)
        artist = ax.scatter(*coords.T, s=s, lw=0, **kw)
        if values is not None:
            mappable = artist
    if three:
        ax.view_init(*view)
        span = np.ptp(np.vstack([c for c, _, _ in layers]), axis=0)
        ax.set_box_aspect(np.maximum(span, span.max() * 1e-3))
        ax.set_zticklabels([])
        ax.set_zlabel(f'Dim {dims[2] + 1}')
    else:
        ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel(f'Dim {dims[0] + 1}')
    ax.set_ylabel(f'Dim {dims[1] + 1}')
    if colorbar and mappable is not None:
        ax.figure.colorbar(mappable, ax=ax, shrink=.6, pad=.16 if three else .05)
    if title is not None:
        ax.set_title(title)
    return ax


def _plotly(layers, three, cmap, s, title, colorbar, dims):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('plot_grids(interactive=True) needs plotly: pip install plotly') from None
    traces = []
    for coords, colour, values in layers:
        marker = dict(size=np.sqrt(s) if s else 3, color=colour if values is None else values)
        if values is not None:
            marker.update(colorscale=cmap, showscale=colorbar)
        xyz = dict(zip('xyz', coords.T))
        traces.append(go.Scatter3d(mode='markers', marker=marker, **xyz) if three
                      else go.Scatter(mode='markers', marker=marker, **xyz))
    fig = go.Figure(traces)
    labels = {f'{a}axis_title': f'Dim {d + 1}' for a, d in zip('xyz', dims)}
    if three:
        fig.update_layout(scene=dict(aspectmode='data', **labels))
    else:
        fig.update_layout(**labels)
        fig.update_yaxes(scaleanchor='x')
    fig.update_layout(showlegend=False, title=title)
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


def plot_slices(gres, key=None, axis=2, n_slices=6, cells=False, dims=(0, 1, 2), cmap='plasma', s=None,
                ncols=None, figsize=None):
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
        Value to colour by (see ``plot_grids``). None shows travel points dark and boundary points light.
        With a key and ``cells=False`` only travel points are drawn.
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
    layers = _layers(gres, key, cells, list(range(len(shape))), rim=False)
    values = [v for _, _, v in layers if v is not None]
    vmin, vmax = (min(v.min() for v in values), max(v.max() for v in values)) if values else (None, None)
    allxy = np.vstack([c[:, shown] for c, _, _ in layers])
    pad = .03 * np.ptp(allxy, axis=0)
    limits = list(zip(allxy.min(0) - pad, allxy.max(0) + pad))
    ncols = ncols or min(n_slices, 3)
    nrows = int(np.ceil(n_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (3 * ncols + 1, 3 * nrows), squeeze=False)
    axes = axes.ravel()
    if s is None:
        s = 4 if cells else 10
    mappable = None
    for k, ax in enumerate(axes):
        if k >= n_slices:
            ax.set_visible(False)
            continue
        for coords, colour, vals in layers:
            u = np.rint((coords[:, cut] - origin[cut]) / spacing[cut])
            slab = np.clip(np.floor((u - lo) / width), 0, n_slices - 1) == k
            kw = dict(color=colour) if vals is None else dict(c=vals[slab], cmap=cmap, vmin=vmin, vmax=vmax)
            artist = ax.scatter(*coords[slab][:, shown].T, s=s, lw=0, **kw)
            if vals is not None:
                mappable = artist
        a, b = origin[cut] + (lo + k * width) * spacing[cut], origin[cut] + (lo + (k + 1) * width) * spacing[cut]
        ax.set_title(f'Dim {cut + 1}: {a:.2f} to {b:.2f}', fontsize='medium')
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'Dim {shown[0] + 1}')
        ax.set_ylabel(f'Dim {shown[1] + 1}')
    if mappable is not None:
        fig.colorbar(mappable, ax=axes[:n_slices].tolist(), shrink=.8, label=key)
    return fig


def plot_views(gres, key=None, views=((20, -60), (20, 30), (90, -90)), cells=False, dims=(0, 1, 2),
               cmap='plasma', s=None, figsize=None):
    """
    The 3-D view of ``plot_grids`` from several angles in one row, with one shared colour bar.

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
    fig = plt.figure(figsize=figsize or (4.5 * len(views) + 1, 4.5))
    axes = [fig.add_subplot(1, len(views), i + 1, projection='3d') for i in range(len(views))]
    fig.subplots_adjust(wspace=.25, bottom=.12)
    for ax, view in zip(axes, views):
        plot_grids(gres, key=key, cells=cells, dims=dims, view=view, ax=ax, cmap=cmap, s=s, colorbar=False,
                   title=f'elevation {view[0]}, azimuth {view[1]}')
    mappables = [c for c in axes[-1].collections if c.get_array() is not None]
    if key is not None and mappables:
        fig.colorbar(mappables[-1], ax=axes, shrink=.6, label=key)
    return fig


def _values(gres, key, cells):
    """Cell values, or grid values at the travel points, with the matching cluster labels."""
    if cells:
        if key not in gres.embedding:
            raise KeyError(f'{key!r} is not stored for the cells; available keys: '
                           f'{sorted(map(str, gres.embedding))}')
        return np.asarray(gres.embedding[key], dtype=float).ravel()
    return _grid_values(gres, key, np.asarray(gres.grids['mapped_travel']))


def plot_trend(gres, keys, x='pseudotime', bins=20, cells=True, labels=None, colors=None, ax=None, band=True):
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

    Returns
    ----------
    The matplotlib axes.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    labels = keys if labels is None else list(labels)
    colors = [None] * len(keys) if colors is None else list(colors)
    if ax is None:
        ax = plt.figure(figsize=(5, 4)).add_subplot()
    xv = _values(gres, x, cells)
    edges = np.linspace(xv.min(), xv.max(), bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    k = np.minimum(((xv - xv.min()) / (xv.max() - xv.min()) * bins).astype(int), bins - 1)
    for key, label, colour in zip(keys, labels, colors):
        y = _values(gres, key, cells)
        used = [b for b in range(bins) if (k == b).any()]
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
    if cells:
        clusters = np.asarray(gres.embedding['clusters'])
    else:
        travel = np.asarray(gres.grids['mapped_travel'])
        clusters = np.asarray(gres.grids['mapped_grids_clusters'])[_positions(gres, travel)]
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


def _policy_arrows(gres, trainer, every=1):
    """Travel points (flat indices) and the move the agent tends to make there, in embedding units."""
    import torch
    from .GridNDCore import OFFSETS_2D
    travel = np.asarray(gres.grids['mapped_travel'])[::every]
    positions = _positions(gres, travel)
    offsets = np.asarray(gres.grids['offsets'] if 'offsets' in gres.grids else OFFSETS_2D, dtype=float)
    agent = trainer.agent
    if hasattr(agent, 'Q'):
        move = offsets[np.argmax(agent.Q.loc[travel].to_numpy(dtype=float), axis=1)]
    else:
        net = agent.actor if hasattr(agent, 'actor') else agent.policy_net
        device = next(iter(net.parameters())).device if hasattr(net, 'parameters') else None
        states = torch.tensor(trainer.env.state_space.mean(axis=1)[positions], dtype=torch.float, device=device)
        out = net(states).detach().cpu().numpy()
        move = out @ offsets if hasattr(agent, 'actor') else offsets[np.argmax(out, axis=1)]
    return travel, move * _lattice_spacing(gres)[0]


def plot_policy(gres, trainer, dims=(0, 1, 2), view=(22, -58), ax=None, every=1, color='k', key=None,
                cmap='plasma'):
    """
    Arrows showing where the agent tends to move from each travel grid point.

    ActorCritic: the expected move, the moves weighted by the actor's probabilities at the grid point's
    state. DDQN and TabularQ: the move with the largest Q-value. Arrows are in embedding units.

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
        (Default: 1)
    color
        Arrow colour.
        (Default: 'k')
    key
        Colour the grid points underneath by this stored value.
        (Default: None, light grey)
    cmap
        Colour map for ``key``.
        (Default: 'plasma')

    Returns
    ----------
    The matplotlib axes.
    """
    three = np.asarray(gres.embedding['embedding']).shape[1] > 2
    dims = list(dims) if three else [0, 1]
    if key is not None:
        ax = plot_grids(gres, key=key, dims=dims, view=view, ax=ax, cmap=cmap)
    else:
        ax = plot_grids(gres, dims=dims, view=view, ax=ax)
        for c in ax.collections:
            c.set_color(GREY)
    travel, arrows = _policy_arrows(gres, trainer, every)
    tails = np.asarray(gres.grids['grids'])[travel][:, dims]
    arrows = arrows[:, dims]
    if three:
        ax.quiver(*tails.T, *arrows.T, color=color, lw=.8, arrow_length_ratio=.3)
    else:
        ax.quiver(*tails.T, *arrows.T, color=color, angles='xy', scale_units='xy', scale=1, width=.003)
    return ax


def plot_training(returns, values=None, window=100, ax=None):
    """
    Training curve: moving average of the episode returns, and optionally the value estimate.

    Parameters
    ----------
    returns
        Episode returns, as returned by ``trainer.train()``.
    values
        Value estimates recorded during training (second output of ``trainer.train()``), drawn on a second
        y axis spread over the same episodes.
        (Default: None)
    window
        Moving-average window in episodes.
        (Default: 100)
    ax
        Axes to draw into.
        (Default: None, a new figure)

    Returns
    ----------
    The matplotlib axes of the returns.
    """
    from .utils import moving_average
    r = np.asarray(returns, dtype=float)
    if window % 2 == 1 and window < len(r):
        y, x = moving_average(r, window), np.arange(1, len(r) + 1)
    else:
        w = min(window, len(r))
        y = np.convolve(r, np.ones(w) / w, mode='valid')
        x = np.arange(w, len(r) + 1)
    if ax is None:
        ax = plt.figure(figsize=(5, 3.5)).add_subplot()
    ax.plot(x, y, color='C0')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    if values is not None:
        v = np.asarray(values, dtype=float)
        twin = ax.twinx()
        twin.plot(np.linspace(1, len(r), len(v)), v, color='C1', lw=1)
        twin.set_ylabel('Value estimate')
    return ax

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

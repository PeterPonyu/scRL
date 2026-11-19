import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def traj_results(gres, lineages, key, points=False, multiple=False):
    """
    Function of plotting the trajectory for lineages.
    
    Parameters
    ----------
    gres
        Grids results with after computing the trajectory.
    lineages
        Lineages is to show.
    key
        Prefix for trajectory.
    points
        Plot the trajectory in grids.
        (Default: False)
    multiple
        Plot multiple lineages simutaneously
        (Default: False)
        
    Returns
    ----------
    None
    """
    clusters = gres.embedding['clusters']
    cluster_colors = gres.embedding['cluster_colors']
    mask = clusters.isin(lineages)
    if gres.embedding['embedding'].shape[1] >= 3:
        return _traj_results_3d(gres, np.asarray(mask), key, points, multiple)
    with sns.axes_style('white'):
        if not multiple:
            sns.scatterplot(x=gres.embedding['embedding'][:,0], y=gres.embedding['embedding'][:,1]
                            , linewidth=0, color='lightgrey')

            sns.scatterplot(x=gres.embedding['embedding'][mask,0], y=gres.embedding['embedding'][mask,1]
                            , linewidth=0, c=gres.embedding['cluster_colors'][mask])
            
        else:
            sns.scatterplot(x=gres.embedding['embedding'][:,0], y=gres.embedding['embedding'][:,1]
                            ,s=0)
        ax = plt.gca()
        ax.set_frame_on(False)
        ax.tick_params(labelleft=False, labelbottom=False)
        if points:
            grids = gres.grids['grids']
            traj_idx = gres.trajectory[key+'_idx']
            for i in range(len(traj_idx)):
                ax.scatter(grids[traj_idx[i],0], grids[traj_idx[i],1]
                           , c=np.arange(len(traj_idx[i])), lw=0, cmap='viridis', alpha=.2)
        else:
            trajs = gres.trajectory[key+'_traj']
            for i in range(trajs.shape[0]-1):
                c = mpl.patches.ConnectionPatch(trajs[i,:], trajs[i+1,:], ax.transData ,arrowstyle='->'
                                    , color=mpl.cm.rainbow(np.linspace(0,1,len(trajs)))[i]
                                                , lw=2, mutation_scale=20, capstyle='round')
                ax.add_patch(c)
    return


VIEW = (22, -58)
PANEL_W, FIG_H = 4.4, 4.6


def _figure_3d(handles=None):
    """
    New figure with a 3-D axes on the left and a strip on the right, wide enough for the legend entries
    ``handles``. Returns the figure, the axes and the left edge of the strip in figure coordinates.
    """
    from ..PlotCore import _legend_width
    strip = max(1.3, _legend_width(handles) if handles else 0)
    W = PANEL_W + strip
    fig = plt.figure(figsize=(W, FIG_H))
    ax = fig.add_axes([0, .02, PANEL_W / W, .9], projection='3d')
    ax.computed_zorder = False
    return fig, ax, PANEL_W / W


def _frame_3d(ax, X):
    """Style of the ``PlotCore`` views on the first three axes, with the limits of the embedding ``X``."""
    from ..PlotCore import _style_3d, _limits
    _style_3d(ax, (0, 1, 2), VIEW, X)
    for set_lim, lim in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), _limits(X)):
        set_lim(lim)


def _legend_3d(ax, handles, x):
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(x, .95), bbox_transform=ax.figure.transFigure,
              frameon=False)


def _colorbar_3d(ax, mappable, label, x, low=False, **kw):
    """Colour bar in the strip at ``x`` (figure coordinates), in its lower part when a legend sits above it."""
    fig = ax.figure
    rect = [x + .15 / fig.get_figwidth(), .1 if low else .25, .14 / fig.get_figwidth(), .35 if low else .5]
    return fig.colorbar(mappable, cax=fig.add_axes(rect, label='<colorbar>'), label=label, **kw)


def _cluster_handles(gres, mask):
    """Legend entries of the clusters of the cells in ``mask``, in cluster order."""
    from ..PlotCore import _cluster_order, _legend_handles
    labels = pd.Categorical(np.asarray(gres.embedding['clusters'])[mask], categories=_cluster_order(gres))
    return _legend_handles(gres, [(None, np.asarray(gres.embedding['cluster_colors'])[mask], None, labels)])


def _cluster_context_3d(ax, gres, mask):
    """All cells small and faint grey, the cells of ``mask`` in their cluster colours."""
    from ..PlotCore import GREY
    X = np.asarray(gres.embedding['embedding'])[:, :3]
    ax.scatter(*X.T, color=GREY, s=2, lw=0, alpha=.5, depthshade=False, zorder=1)
    ax.scatter(*X[mask].T, color=np.asarray(gres.embedding['cluster_colors'])[mask], s=4, lw=0,
               depthshade=False, zorder=2)


def _path_3d(ax, path, cmap='rainbow'):
    """
    A path drawn as one line coloured along its length, over a wider white line, with a round marker at the
    start and an arrow head on the last step. Returns the mappable of the colour scale (0 start, 1 end).
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    cmap = plt.get_cmap(cmap)
    lw = 2.5
    ax.plot(*path.T, color='white', lw=lw + 2, solid_capstyle='round', zorder=3)
    segments = np.stack([path[:-1], path[1:]], axis=1)
    ax.add_collection3d(Line3DCollection(segments, colors=cmap(np.linspace(0, 1, len(segments))), lw=lw,
                                         capstyle='round', zorder=4))
    ax.scatter(*path[0], color=cmap(0.), s=40, edgecolor='white', lw=1, depthshade=False, zorder=5)
    step = path[-1] - path[-2]
    if np.linalg.norm(step) > 0:
        head = .06 * max(np.ptp(lim) for lim in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()))
        step = step / np.linalg.norm(step) * head
        ax.quiver(*(path[-1] - step), *step, color=cmap(1.), lw=lw, arrow_length_ratio=1, zorder=5)
    return mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap)


def _traj_results_3d(gres, mask, key, points, multiple):
    """
    3-D view for ``traj_results`` on the first three axes of the embedding, styled like the ``PlotCore`` views.
    With ``multiple`` the current 3-D axes is reused so several lineages can be overlaid (no context, legend or
    colour bar is added); otherwise a new figure is made with the chosen clusters, their legend and a colour bar.
    """
    X = np.asarray(gres.embedding['embedding'])[:, :3]
    drawn = [a for a in plt.gcf().axes if a.name == '3d'] if multiple and plt.get_fignums() else []
    reuse = bool(drawn)
    if reuse:
        ax = drawn[0]
    else:
        handles = _cluster_handles(gres, mask)
        fig, ax, x = _figure_3d(handles)
        _cluster_context_3d(ax, gres, mask)
        _frame_3d(ax, X)
    if points:
        grids = np.asarray(gres.grids['grids'])
        idx = gres.trajectory[key + '_idx']
        visited = grids[np.concatenate([np.asarray(i, dtype=int) for i in idx])][:, :3]
        steps = np.concatenate([np.arange(len(i)) for i in idx])
        mappable = ax.scatter(*visited.T, c=steps, cmap='viridis', s=8, lw=0, depthshade=False, zorder=3)
        label, ticks = 'Step', None
    else:
        mappable = _path_3d(ax, gres.trajectory[key + '_traj'][:, :3])
        label, ticks = 'Along the path', [0, 1]
    if not reuse:
        if handles:
            _legend_3d(ax, handles, x)
        bar = _colorbar_3d(ax, mappable, label, x, low=bool(handles), ticks=ticks)
        if ticks:
            bar.set_ticklabels(['start', 'end'])
        ax.figure.sca(ax)
    return

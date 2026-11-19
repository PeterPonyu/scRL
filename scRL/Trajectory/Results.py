import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


def _traj_results_3d(gres, mask, key, points, multiple):
    """
    3-D view for ``traj_results`` on the first three axes of the embedding. With ``multiple`` the current
    3-D axes is reused so several lineages can be overlaid; otherwise a new figure is made.
    """
    X = gres.embedding['embedding'][:, :3]
    if multiple and plt.get_fignums() and plt.gcf().axes and plt.gca().name == '3d':
        ax = plt.gca()
    else:
        ax = plt.figure(figsize=(6, 5)).add_subplot(projection='3d')
    if not multiple:
        ax.scatter(*X.T, c='lightgrey', s=4, lw=0)
        ax.scatter(*X[mask].T, c=gres.embedding['cluster_colors'][mask], s=4, lw=0)
    if points:
        grids = gres.grids['grids']
        for idx in gres.trajectory[key + '_idx']:
            ax.scatter(*grids[idx, :3].T, c=np.arange(len(idx)), lw=0, cmap='viridis', alpha=.2)
    else:
        trajs = gres.trajectory[key + '_traj'][:, :3]
        colors = mpl.cm.rainbow(np.linspace(0, 1, len(trajs)))
        for i in range(len(trajs) - 1):
            ax.plot(*trajs[i:i + 2].T, color=colors[i], lw=2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    return

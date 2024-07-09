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


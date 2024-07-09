import pandas as pd
import numpy as np

def get_traj_df(gres, trainer, n = 100, key = None):
    """
    Function for calculating the trajectory.
    
    Parameters
    ----------
    gres
        Grids results.
    trainer
        The trained agent for sampling differentiation paths.
    n
        The original learned trajectory.
        (Default: 100)
    key
        Prefix for the trajectory.
        (Default: None)
        
    Returns
    ----------
    None
    """
    grids = gres.grids['grids']
    time = gres.grids['pseudotime']
    idxs = get_traj(trainer, n=n)
    gres.trajectory[key + '_idx'] = idxs
    df = pd.DataFrame(idxs)
    ls = []
    for i, c in enumerate(df.columns):
        if df[c].isna().sum() > n * .9:
            continue
        else:
            idx = df[c].dropna().values.astype('int')
            mask = time[idx].values < i / gres.grids['n']
            if any(mask):
                idx = idx[mask]
                ls.append(grids[idx,:].mean(axis=0))
    trajs = np.vstack(ls)
    gres.trajectory[key + '_traj'] = trajs
    return


def get_traj(agent, n=50, deep=True):
    """
    Trajectory sampling with learned algorithm
    
    Parameters
    ----------
    agent
        Agent after training.
    n
        Number of trajectories to sample.
        (Default: 50)
    deep
        The environment mode.
        (Default: True)
    
    Returns
    ----------
    list
        Grid index for trajectory points
    """
    traj = []
    for i in range(n):
        trajectory = []
        state = agent.env.reset()
        if deep:
            trajectory.append(agent.env.state_idx)
        else:
            trajectory.append(state)
        done, trunc = False, False
        episode_return = 0
        while not done and not trunc:
            action = agent.agent.take_action(state)
            next_state, reward, done, trunc = agent.env.step(action)
            if deep:
                trajectory.append(agent.env.state_idx)
            else:
                trajectory.append(next_state)
            state = next_state
            episode_return += reward
        traj.append(trajectory) 
    return traj

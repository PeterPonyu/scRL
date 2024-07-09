import numpy as np
from joblib import Parallel, delayed
from pygam import LinearGAM, s
import torch
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

def get_state_value(gres, trainer, key):
    gres.grids[key] = trainer.agent.critic(torch.tensor(trainer.env.state_space.mean(axis=1),device=device)).detach().cpu().numpy().ravel()
    return 
    

def get_dist(a,b):
    """Pairwise distances"""
    if len(a.shape) == 1:
        return np.linalg.norm(a-b,axis=1)
    else:
        res = Parallel(n_jobs=-2)(delayed(np.linalg.norm)(a[i,:]-b,axis=1) for i in range(len(a)))
        return np.vstack(res)

    
def project_cells(gres):
    """Project results to the original embedding"""
    
    dis = get_dist(gres.embedding['embedding'],gres.grids['grids'][gres.grids['mapped_grids']])
    dis_ = (dis - dis.min(axis=1).reshape(-1,1))/(dis.max(axis=1) - dis.min(axis=1)).reshape(-1,1)
    proj = np.average(np.exp(-dis_),1,gres.qlearning['Q'].sum(axis=1).values)
    return proj


def fit_gam(proj, exp):
    """Gam model for genes"""
    
    mask = np.where(exp > 0)[0]
    gam = LinearGAM(s(0,n_splines=4,spline_order=2)).fit(proj[mask],exp[mask])
    return gam.predict(np.linspace(0,1,500))


def gam_predict(proj, mat):
    """Get gene expression along the project value"""
    
    res = Parallel(n_jobs=16)(delayed(fit_gam)(proj,mat[g]) for g in mat.columns)
    return np.vstack(res)

def moving_average(a, window_size):
    """MA function for the reinforcement learning training process"""
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))



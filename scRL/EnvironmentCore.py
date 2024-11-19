import numpy as np
import pandas as pd
import time
import random
import tqdm
from collections import namedtuple, deque
from .utils import get_dist


class deepEnv:
    """
    Environment for deep reinforcement learning on grid embedding.
    
    Parameters
    ----------
    gres
        Grids results after reward generating.
    X_pca
        The input latent space.
    max_step
        Maximum steps the agent can travel in the environment.
        (Default: 50)
    KNN
        Nearest neighbors for considering each grid points state value.
        (Default: 10)
    reward_type
        The reward type generated.
        (Default: 'c')
        Two types are included:'c','d'.
    reward_mode
        The reward mode selected when generating the reward.
        (Default: 'Decision')
        Two modes are included:'Decision','Contribution'.
    """
    def __init__(self, gres, X_pca, max_step = 50, KNN=100, reward_type='c', reward_mode='Decision', starts_probs=True):
        self.N = gres.grids['n']
        self.pseudotime = gres.grids['pseudotime']
        self.rewards_df = gres.qlearning[f'{reward_type}_{reward_mode}_rewards']
        self.mapped_grids = gres.grids['mapped_grids']
        self.starts_probs = None
        if starts_probs:
            self.starts_probs = gres.grids['starts_probs']
        
        self.starts_grids = gres.grids['starts_cluster_grids']
        self.mapped_boundary = gres.grids['mapped_boundary']
        
        self.state_space = get_state(gres, X_pca, KNN)
        self.reset_ = False
        self.max_step = max_step
    
    def reset(self):
        if self.starts_probs is not None:
            start_idx = np.random.choice(self.mapped_grids, 1, p=self.starts_probs).item()
        else:
            start_idx = random.sample(self.starts_grids, 1)[0]
        self.state_idx = start_idx
        start_point = np.where(self.mapped_grids == start_idx)[0][0]
        state = np.mean(self.state_space[start_point, :, :], axis=0)
        self.state = state
        self.time_step = 0
        self.reset_ = True 
        self.trajectory = [start_idx]
        return self.state
        
    def step(self, action):
        if not self.reset_:
            print('Warning: reset the environment first!')
            return 
        termination = False
        truncation = False
        df = self.rewards_df
        idx = self.state_idx
        i = idx%self.N
        j = idx//self.N
        direction_lut = dict(zip(df.columns,[(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1),(i,j-1),(i-1,j-1),(i-1,j),(i-1,j+1)]))
        A = direction_lut[df.columns[action]]
        next_idx = A[0] + A[1]*self.N 
        self.time_step += 1
        reward = df.loc[idx, df.columns[action]]
        
        if reward == -1:
            termination = True
            reward = -2
            return self.state, reward, termination, truncation
        self.state_idx = next_idx
        next_point = np.where(self.mapped_grids == next_idx)[0][0]
        next_state = np.mean(self.state_space[next_point, :, :], axis=0)
        self.state = next_state
        
        b_s = set(self.mapped_boundary)
        s_s = set(self.starts_grids) if self.starts_grids else set()
        
        if next_idx in b_s.difference(s_s):
            termination = True
            pseudotime_reward = self.pseudotime[next_idx] - self.pseudotime[idx]
            reward += pseudotime_reward
            return self.state, reward, termination, truncation

        if next_idx in self.trajectory:
            termination = True
            reward = -1
            return self.state, reward, termination, truncation
        else:
            self.trajectory.append(next_idx)
        
        if self.time_step > self.max_step:
            truncation = True
        return next_state, reward, termination, truncation


transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class replayBuffer(object):
    """
    Container of the sampled transitions
    """
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(transition(*args))
        
    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.memory)


def get_state(gres, X, KNN=10):
    """
    gres : Grid results container
    X : Latent space of observations
    knn : Nearest number of observations
    """
    a = gres.grids['grids'][gres.grids['mapped_grids']]
    b = gres.embedding['embedding']
    dists = get_dist(a,b)
    knn_idx = np.argsort(dists,axis=1)[:,:KNN]
    state_space = np.vstack([[X[knn_idx[i,:],:]] for i in range(len(knn_idx))])
    return state_space

    
def d_rewards(gres,
                 starts,
                 ends,
                 beta=1,
                 mode='Decision'
                ):
    """
    Function to generate lineage specific reward table for constructing environment
    As grid points may represent distinct states of differentiation,
    the fate decision process can be modeled as cell transition from one grid point to its neighboring points.
    Upon reaching the target cluster grid point, the cell is rewarded for its journey.
    
    Parameters
    ----------
    gres
        Grids results after cluster projection.
    starts
        Starting grids cluster annotation.
    ends
        Terminating grids cluster annotation.
    beta
        Decay coefficient.
        (Default: 1)
    mode
        Two modes are included:'Decision' and 'Contribution'.
        (Default: 'Decision')
    
    Returns
    ----------
    None
    """
    start_time = time.time()
    masked_grids = gres.grids['masked_grids']
    mapped_grids = gres.grids['mapped_grids']
    mapped_grids_clusters = gres.grids['mapped_grids_clusters']
    mapped_boundary = gres.grids['mapped_boundary']
    pseudotime = gres.grids['pseudotime']
    n = gres.grids['n']
    mat = np.ones((n,n)).ravel()
    mat[masked_grids] = 0
    mat = mat.reshape(n,n,order='F')
    
    starts_cluster_grids = [list(mapped_grids)[i] 
                            for i in np.hstack([np.where(np.array(mapped_grids_clusters)==c) 
                                                for c in starts])[0].tolist()]
    ends_cluster_grids = [list(mapped_grids)[i] 
                          for i in np.hstack([np.where(np.array(mapped_grids_clusters)==c) 
                                              for c in ends])[0].tolist()]
    
    lineage_time = pseudotime[ends_cluster_grids]
    scaled_time = (lineage_time - lineage_time.min()) / (lineage_time.max() - lineage_time.min())
    
    df = pd.DataFrame(index=list(mapped_grids),columns=['R','RT','T','LT','L','LB','B','RB'])
    pbar = tqdm.tqdm(total=len(df.index), desc='Reward generating')
    for idx in df.index:
        i = idx%mat.shape[0]
        j = idx//mat.shape[0]
        L = len(mat)-1
        direction_lut = dict(zip(df.columns,[(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1),(i,j-1),(i-1,j-1),(i-1,j),(i-1,j+1)]))
        for direction in df.columns:
            D = direction_lut[direction]
            if (i in [0,L]) or (j in [0,L]):
                if i == 0 and direction.endswith('B'):
                    df.loc[idx,direction] = -1
                elif i == L and direction.endswith('T'):
                    df.loc[idx,direction] = -1
                elif j == 0 and direction.startswith('L'):
                    df.loc[idx,direction] = -1
                elif j == L and direction.startswith('R'):
                    df.loc[idx,direction] = -1
                else:
                    df.loc[idx,direction] = mat[D] - 1
            else:
                df.loc[idx,direction] = mat[D] - 1
            if  -1 < D[0] < n and -1 < D[1] < n:
                next_idx = D[0] + n*D[1]
                if next_idx in ends_cluster_grids:
                    if mode == 'Decision':
                        df.loc[idx,direction] += np.exp(-beta*scaled_time[next_idx])
                    elif mode == 'Contribution':
                        df.loc[idx,direction] += 1 - np.exp(-beta*scaled_time[next_idx])
                    else:
                        raise ValueError('Mode must be one of "Decision" and "Contribution"!')
        pbar.update(1)
    pbar.close()
    gres.qlearning['reward_key'] = '.'.join(ends)
    gres.grids['ends_cluster_grids'] = ends_cluster_grids
    gres.grids['starts_cluster_grids'] = starts_cluster_grids
    gres.qlearning[f'd_{mode}_rewards'] = df
    gres.qlearning['matrix'] = mat
    end_time = time.time()
    print(f'Time used for generating rewards : {(end_time - start_time):.2f} seconds')
    return 

def c_rewards(gres,
              reward_keys,
              starts=None,
              starts_keys=None,
              punish_keys=None,
              beta=1,
              mode='Decision'
             ):
    """
    Function to generate continuous value specific reward table.
    Lineage-specific genes are upregulated during cell differentiation,
    while the early genes as well as other lineage genes are downregulated,
    reflecting the shift in gene expression patterns.
    
    Parameters
    ----------
    gres
        Grids results after gene projection
    reward_keys
        Rewarded by the specific continuous value
    starts
        Starting grids cluster annotation
    starts_keys
        Starting by sampling from the continuous value
    punish_keys
        Punished by the continuous value
    beta
        Decay coefficient
        (Default: 1)
    mode
        Two modes are included: 'Decision' and 'Contribution'.
        (Default: 'Decision')
    
    Returns
    ----------
    None
    """
    start_time = time.time()
    masked_grids = gres.grids['masked_grids']
    mapped_grids = gres.grids['mapped_grids']
    mapped_boundary = gres.grids['mapped_boundary']
    mapped_grids_clusters = gres.grids['mapped_grids_clusters']
    pseudotime = gres.grids['pseudotime']
    n = gres.grids['n']
    mat = np.ones((n,n)).ravel()
    mat[masked_grids] = 0
    mat = mat.reshape(n,n,order='F')

    reward = gres.grids['proj'][reward_keys].mean(axis=1).values
    reward_gene_time = pseudotime[reward > 0]
    reward_scaled_time = (reward_gene_time - reward_gene_time.min()) / (reward_gene_time.max() - reward_gene_time.min())
    
    if starts:
        starts_cluster_grids = [list(mapped_grids)[i] 
                            for i in np.hstack([np.where(np.array(mapped_grids_clusters)==c) 
                                                for c in starts])[0].tolist()]
        gres.grids['starts_cluster_grids'] = starts_cluster_grids

    else:
        gres.grids['starts_cluster_grids'] = None
        
    if starts_keys:
        starts_probs = gres.grids['proj'][starts_keys].mean(axis=1).values
        gres.grids['starts_probs'] = (starts_probs / starts_probs.sum()).astype(float)
    
    if punish_keys:
        punish = gres.grids['proj'][punish_keys].mean(axis=1).values
        punish_gene_time = pseudotime[punish > 0]
        punish_scaled_time = (punish_gene_time - punish_gene_time.min()) / (punish_gene_time.max() - punish_gene_time.min())
        
    df = pd.DataFrame(index=list(mapped_grids),columns=['R','RT','T','LT','L','LB','B','RB'])
    
    pbar = tqdm.tqdm(total=len(df.index), desc='Reward generating')
    for idx in df.index:
        i = idx%mat.shape[0]
        j = idx//mat.shape[0]
        L = len(mat)-1
        direction_lut = dict(zip(df.columns,[(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1),(i,j-1),(i-1,j-1),(i-1,j),(i-1,j+1)]))
        for direction in df.columns:
            D = direction_lut[direction]
            if (i in [0,L]) or (j in [0,L]):
                if i == 0 and direction.endswith('B'):
                    df.loc[idx,direction] = -1
                elif i == L and direction.endswith('T'):
                    df.loc[idx,direction] = -1
                elif j == 0 and direction.startswith('L'):
                    df.loc[idx,direction] = -1
                elif j == L and direction.startswith('R'):
                    df.loc[idx,direction] = -1
                else:
                    df.loc[idx,direction] = mat[D] - 1
            else:
                df.loc[idx,direction] = mat[D] - 1
            if mode not in ['Decision','Contribution']:
                raise ValueError('Mode must be one of "Decision" and "Contribution"!')
            if -1 < D[0] < n and -1 < D[1] < n:
                next_idx = D[0] + n*D[1]
                if punish_keys:
                    if next_idx in mapped_grids[reward > 0]:
                        if mode == 'Decision':
                            next_reward = reward[np.where(mapped_grids==next_idx)[0][0]]
                            df.loc[idx,direction] += np.exp(-beta*reward_scaled_time[next_idx]) * next_reward    
                        elif mode == 'Contribution':
                            next_reward = reward[np.where(mapped_grids==next_idx)[0][0]]
                            df.loc[idx,direction] += (1 - np.exp(-beta*reward_scaled_time[next_idx])) * next_reward        
                    if next_idx in mapped_grids[punish > 0]:
                        if mode == 'Decision':
                            next_punish = punish[np.where(mapped_grids==next_idx)[0][0]]
                            df.loc[idx,direction] -= np.exp(-beta*punish_scaled_time[next_idx]) * next_punish
                        elif mode == 'Contribution':
                            next_punish = punish[np.where(mapped_grids==next_idx)[0][0]]
                            df.loc[idx,direction] -= (1 - np.exp(-beta*punish_scaled_time[next_idx])) * next_punish
                else:
                    if next_idx in mapped_grids[reward > 0]:
                        if mode == 'Decision':
                            next_reward = reward[np.where(mapped_grids==next_idx)[0][0]]
                            df.loc[idx,direction] += np.exp(-beta*reward_scaled_time[next_idx]) * next_reward    
                        elif mode == 'Contribution':
                            next_reward = reward[np.where(mapped_grids==next_idx)[0][0]]
                            df.loc[idx,direction] += (1 - np.exp(-beta*reward_scaled_time[next_idx])) * next_reward                 
        pbar.update(1)
    pbar.close()
    gres.qlearning['reward_key'] = '.'.join(reward_keys)
    gres.qlearning[f'c_{mode}_rewards'] = df
    gres.qlearning['matrix'] = mat
    end_time = time.time()
    print(f'Time used for reward generation: {(end_time - start_time):.2f} seconds')    
    return 




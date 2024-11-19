import pandas as pd
import numpy as np
import random

class tabularEnv:
    """
    The tabularEnv class is a specialized environment designed for tabular Q-learning, tailored to navigate grid-based results generated from complex data, such as biological or gene expression grids. This environment is particularly structured to handle various types of rewards and operational modes, making it highly adaptable to different learning scenarios and data interpretations.
The environment is initialized with gres, a structured data object containing the grid results post-reward generation. This object includes multiple attributes such as qlearning for rewards, and other grid-related information like mapped_grids, starts_cluster_grids, mapped_boundary, and pseudotime. These elements are crucial for defining the spatial and temporal structure within which the learning agent operates.
The max_step parameter limits the number of steps an agent can take in a single episode, defaulting to 50, to prevent infinite loops and ensure computational efficiency. The reward_type and reward_mode parameters allow customization of the reward structure. reward_type can be either 'lineage' or 'gene', indicating the nature of the reward, while reward_mode can be 'Decision' or 'Contribution', specifying how rewards are calculated based on the agent's actions.
Upon initialization, the environment sets up the reward dataframe from gres based on the specified reward_type and reward_mode. This setup facilitates the dynamic retrieval and calculation of rewards as the agent interacts with the environment.
The reset method is crucial for starting or restarting an episode. It randomly selects a starting state from starts_grids, initializes the time_step, and sets reset_ to True, indicating that the environment is ready for a new episode. The agent's trajectory, which records the states visited during an episode, is also initialized here.
The step method is where the action takes place. If the environment has not been reset, it prompts the user to do so, ensuring that each episode begins from a valid starting point. Given an action, the environment calculates the next state using a lookup table that maps actions to grid movements. The reward for moving to this state is retrieved, and if the movement leads to a state outside the allowed grid or revisits a previous state, termination or truncation conditions are triggered. Additionally, rewards are adjusted based on the difference in pseudotime between the current and next states, integrating a temporal aspect into the reward system. This method handles termination if the agent reaches a boundary state not in starts_grids, and truncation if the agent either revisits a state or exceeds the maximum allowed steps.
    
    Parameters
    ----------
    gres
        Grids results after reward generating.
    max_step
        Maximum steps the agent can travel in the environment.
        (Default: 50)
    reward_type
        The reward type generated.
        (Default: 'c')
        Two types are included:'c','d'.
    reward_mode
        The reward mode selected when generating the reward.
        (Default: 'Decision')
        Two modes are included:'Decision','Contribution'.
    """
    def __init__(self, gres, max_step = 50, reward_type='c', reward_mode='Decision', starts_probs=True):
        self.rewards_df = gres.qlearning[f'{reward_type}_{reward_mode}_rewards']
        self.mapped_grids = gres.grids['mapped_grids']
        self.starts_probs = None
        if starts_probs:
            self.starts_probs = gres.grids['starts_probs']
        
        self.starts_grids = gres.grids['starts_cluster_grids']
        self.mapped_boundary = gres.grids['mapped_boundary']
        self.pseudotime = gres.grids['pseudotime']
        self.reset_ = False
        self.max_step = max_step
        
        
    def reset(self):
        if self.starts_probs is not None:
            state = np.random.choice(self.mapped_grids, 1, p=self.starts_probs).item()
        else:
            state = random.sample(self.starts_grids, 1)[0]
        
        self.state = state
        self.time_step = 0
        self.reset_ = True
        self.trajectory = [state]
        return self.state
        
    def step(self, action):
        
        if not self.reset_:
            print('Warning: reset the environment first!')
            return 
        termination = False
        truncation = False
        df = self.rewards_df
        idx = self.state
        i = idx%50
        j = idx//50
        direction_lut = dict(zip(df.columns,[(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1),(i,j-1),(i-1,j-1),(i-1,j),(i-1,j+1)]))
        A = direction_lut[df.columns[action]]
        
        next_state = A[0] + A[1]*50
        reward = df.loc[idx, df.columns[action]]
        if reward == -1:
            termination = True
            reward = -2
            return self.state, reward, termination, truncation
        pseudotime_reward = self.pseudotime[next_state] - self.pseudotime[idx]
        reward += pseudotime_reward
        self.state = next_state
    
        b_s = set(self.mapped_boundary)
        s_s = set(self.starts_grids)
        
        if next_state in b_s.difference(s_s):
            termination = True
            return self.state, reward, termination, truncation
        
        if next_state in self.trajectory:
            truncation = True
            reward = -1
        else:
            self.trajectory.append(next_state)
        
        self.time_step += 1
        if self.time_step > self.max_step:
            truncation = True
        return next_state, reward, termination, truncation


class tabularQ:
    def __init__(self, env, alpha=.5, gamma=.9, n_actions=8):
        self.Q = pd.DataFrame(0, index=env.mapped_grids, columns=np.arange(8))
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.steps = 0
        
    def take_action(self, state):
        eps_thres = .01 + (.9 - .01) * np.exp(-.01 * self.steps / 1000)
        if random.random() < eps_thres:
            action = random.randint(0, self.n_actions - 1)
        else:
            action_space = self.Q.loc[state,:]
            action = np.where(action_space == action_space.max())[0]
            action = action[random.randint(0, len(action) - 1)]
        self.steps += 1
        return action
     
    def max_Q_value(self, state):
        return self.Q.loc[state,:].max()
    
    def update(self, state, action, reward, next_state):
        td_err = reward + self.gamma * self.Q.loc[next_state, :].max() - self.Q.loc[state, action]
        self.Q.loc[state, action] += self.alpha * td_err 
        



    
    

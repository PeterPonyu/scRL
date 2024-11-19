import numpy as np
import tqdm
from .EnvironmentCore import deepEnv, replayBuffer
from .DDQNCore import dqn
from .ActorCriticCore import actorcritic
from .TabularQCore import tabularQ, tabularEnv
from .Simulator.Core import simulator
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class trainer:
    """
    Reinforcement learning trainer class.

    Parameter
    ----------
    algo
        Algorithm to train the agent.
        Three algorithms are included: 'TabularQ', 'ActorCritic', 'DDQN'.
    gres
        Grids results after all the preprocessings.
    reward_type
        The reward type generated.
        (Default: 'c')
        Two types are included:'c','d'.
    reward_mode
        The reward mode selected when generating the reward.
        (Default: 'Decision')
        Two modes are included:'Decision','Contribution'.
    simulating
        Whether to train a simulator together.
        (Default: False)
    X_latent
        Latent space for the input of deep environment.
        (Default: None)
    hidden_dim
        The dimensionality of the hidden layer.
        (Default: 128)
    num_episodes
        The maximum training episodes.
        (Default: 10000)
    max_step
        Maximum steps the agent can travel in the environment.
        (Default: 50)
    KNN
        Nearest neighbors for considering each grid points state value.
        (Default: 10)
    batch_size
        The batch size during training
        (Default: 64)
    capacity
        The size of replay buffer of DDQN.
        (Default: 10000)
    soft
        Whether to soft update the target net of DDQN.
        (Default: True)
    target_update
        Steps for udating target net of DDQN.
        (Default: 50)
    gamma
        Decaying factor for reinforcement learning.
        (Default: .8)
    tau
        Soft update percent for the target net of DDQN.
        (Default: .005)
    alpha
        Learning rate for tabular Q learning.
        (Default: .5)
    lr
        Learning rate for the DDQN.
        (Default: 1e-3)
    actor_lr
        The actor learning rate for ActorCritic.
        (Default: 5e-4)
    critic_lr
        The critic learning rate for ActorCritic.
        (Default: 1e-3)
    """
    def __init__(self,
                 algo,
                 gres,
                 reward_type = 'c',
                 reward_mode = 'Decision',
                 starts_prob = True,
                 X_latent = None,
                 hidden_dim = 128,
                 num_episodes=10000,
                 max_step = 50,
                 KNN = 10,
                 batch_size = 64,
                 capacity = 10000,
                 soft = True,
                 target_update = 50,
                 gamma = .9,
                 tau = .005,
                 alpha=.5,
                 lr = 1e-3,
                 actor_lr = 5e-4,
                 critic_lr = 1e-3
                ):
        
        self.algo = algo    
        self.num_episodes = num_episodes
        self.gres = gres
        if algo == 'TabularQ':
            self.env = tabularEnv(gres, max_step, reward_type, reward_mode, starts_prob)
            self.agent = tabularQ(self.env, alpha, gamma, 8)
        else:
            if X_latent is not None:
                gres.embedding['X_latent'] = X_latent
                state_dim = X_latent.shape[1]
                self.state_dim = state_dim
                self.hidden_dim = hidden_dim
                self.env = deepEnv(gres, X_latent, max_step, KNN, reward_type, reward_mode, starts_prob)
                if algo == 'DDQN':
                    self.batch_size = batch_size
                    self.capacity = capacity
                    self.agent = dqn(state_dim, hidden_dim, 8, gamma, lr, soft, target_update, tau)
                if algo == 'ActorCritic':
                    self.agent = actorcritic(state_dim, hidden_dim, 8, gamma, actor_lr, critic_lr)
            else:
                raise ValueError('Please select a latent space of the data')
        

    def train(self):
        if self.algo == 'TabularQ':
            return train_tabular(self.agent,
                            self.env,
                            self.num_episodes
                           )
        if self.algo == 'DDQN':
            self.memory = replayBuffer(self.capacity)
            return train_off_policy(self.agent,
                                    self.env,
                                    self.memory,
                                    self.num_episodes,
                                    self.batch_size,
                                    self.capacity*.1
                                   )
        if self.algo == 'ActorCritic':
            return train_on_policy(self.agent,
                                   self.env,
                                   self.num_episodes
                                  )
    def eval_fate(self,):
        key = self.gres.qlearning['reward_key']
        if self.algo in ['ActorCritic', 'DDQN']:
            self.gres.grids[f'fate_{key}'] = self.agent.critic(torch.tensor(self.env.state_space.mean(axis=1),device=device)).detach().cpu().numpy().ravel()
        else:
            self.gres.grids[f'fate_{key}'] = self.agent.Q.mean(axis=1)
    
    def train_simulator(self, num_episodes, latent_dim):
        agent = self.agent
        env = self.env
        exps = self.gres.grids['proj'].values.astype('float')
        grids = self.gres.grids['grids']
        times = self.gres.grids['pseudotime']
        mapped_grids = self.gres.grids['mapped_grids']
        sim = simulator(self.state_dim, self.hidden_dim, latent_dim, exps.shape[1])
        self.sim = sim
        return_list = []
        v_value_list = []
        v_value = 0
        for i in range(10):
            with tqdm.tqdm(total = int(num_episodes / 10), desc = 'Iteration%d'%(i+1)) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    state = env.reset()
                    done, trunc = False, False
                    episode_return =  0
                    while not done and not trunc:
                        exp = exps[np.where(mapped_grids == env.state_idx)[0][0]]
                        grid = grids[env.state_idx]
                        time = [times[env.state_idx]]
                        action = agent.take_action(state)
                        next_state, reward, done, trunc = env.step(action)
                        if not done and not trunc:
                            next_exp = exps[np.where(mapped_grids == env.state_idx)[0][0]]
                            next_grid = grids[env.state_idx]
                            next_time = [times[env.state_idx]]
                            sim.update(state, exp, grid, time, next_state, next_exp, next_grid, next_time)
                        v_value = .005 * agent.v_value(state) + .995 * v_value
                        v_value_list.append(v_value)
                        state = next_state
                        episode_return += reward
                    return_list.append(episode_return)
                    if (i_episode + 1) % 100 == 0:
                        pbar.set_postfix({'E': '%d'%(num_episodes / 10 * i + i_episode + 1)
                                          ,'R': '%.2f'%np.mean(return_list[-100:])})
                    pbar.update()
        return return_list, v_value_list


def train_on_policy(agent,
                    env,
                    num_episodes
                   ):
    return_list = []
    v_value_list = []
    v_value = 0
    for i in range(10):
        with tqdm.tqdm(total = int(num_episodes / 10), desc = 'Iteration%d'%(i+1)) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                state = env.reset()
                done, trunc = False, False
                transition_dict = {'states':[],'actions':[],'rewards':[],'next_states':[],'dones':[]}
                episode_return =  0
                while not done and not trunc:
                    action = agent.take_action(state)
                    next_state, reward, done, trunc = env.step(action)
                    v_value = .005 * agent.v_value(state) + .995 * v_value
                    v_value_list.append(v_value)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['rewards'].append(reward)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({'E': '%d'%(num_episodes / 10 * i + i_episode + 1)
                                      ,'R': '%.2f'%np.mean(return_list[-100:])})
                pbar.update()
    return return_list, v_value_list


def train_off_policy(agent,
                     env,
                     memory,
                     num_episodes,
                     batch_size,
                     minimal_size
                    ):
    return_list = []
    max_Q_list = []
    max_Q = 0
    for i in range(10):
        with tqdm.tqdm(total=int(num_episodes / 10), desc='Iteration%d'%(i+1)) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                done, trunc = False, False
                state = env.reset()
                while not done and not trunc:
                    action = agent.take_action(state)
                    max_Q = agent.max_Q_value(state) * .005 + max_Q * .995
                    max_Q_list.append(max_Q)
                    next_state, reward, done, trunc = env.step(action)
                    memory.push(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if memory.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = memory.sample(batch_size)
                        transition_dict = {'states':b_s, 'actions':b_a, 'rewards':b_r, 'next_states':b_ns, 'dones':b_d} 
                        agent.update(transition_dict)        
                return_list.append(episode_return)
                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({'E':'%d'%(num_episodes / 10 * i + i_episode + 1)
                                     ,'R':'%.2f'%np.mean(return_list[-100:])})
                pbar.update() 
    return return_list, max_Q_list


def train_tabular(agent,
                 env,
                 num_episodes
                 ):
    return_list = []
    max_Q_list = []
    max_Q = 0
    for i in range(10):
        with tqdm.tqdm(total=int(num_episodes/10), desc='Iteration:%d'%(i+1)) as pbar:
            for i_episode in range(int(num_episodes/10)):
                state = env.reset()
                episode_return = 0
                done, trunc = False, False
                while not done and not trunc:
                    action = agent.take_action(state)
                    next_state, reward, done, trunc = env.step(action)
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    episode_return += reward
                    max_Q = .005*agent.max_Q_value(state) + .995*max_Q
                    max_Q_list.append(max_Q)
                return_list.append(episode_return)
                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({'E':'%d'%(num_episodes/10*i+i_episode+1),
                                      'R':'%.2f'%np.mean(return_list[-100:])})
                pbar.update(1)
    return return_list, max_Q_list
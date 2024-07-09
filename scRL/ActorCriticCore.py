import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class PolicyNet(nn.Module):
    """
    The policy net in actor-critic architecture.
    Designed to output the probability distribution over each potential action.
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.nn = nn.Sequential(nn.Linear(state_dim, hidden_dim)
                               ,nn.ReLU()
                               ,nn.Linear(hidden_dim, hidden_dim)
                               ,nn.ReLU()
                               ,nn.Linear(hidden_dim, action_dim)
                               ,nn.Sigmoid()
                               )
        self.softmax = nn.Softmax(dim=1)
        self.apply(weight_init)
        
    def forward(self, x):
        x =  1 + (self.nn(x) * 5)
        return self.softmax(x)

class ValueNet(nn.Module):
    """
    The value net in actor-critic architechture.
    Serving as an approximator for the state value function.
    """
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.nn = nn.Sequential(nn.Linear(state_dim, hidden_dim)
                               ,nn.ReLU()
                               ,nn.Linear(hidden_dim, hidden_dim)
                               ,nn.ReLU()
                               ,nn.Linear(hidden_dim, 1))
        self.apply(weight_init)
        
    def forward(self, x):
        return self.nn(x)


class actorcritic:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, actor_lr, critic_lr):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr)
        self.gamma = gamma
        
    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        return action_dist.sample().item()
    
    def v_value(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        return self.critic(state).item()
    
    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.int64).view(-1,1).to(device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1,1).to(device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1,1).to(device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        p = self.actor(states)
        entropy_loss = torch.mean(torch.sum(p*torch.log(p) ,axis=1))
        actor_loss = torch.mean(-log_probs * td_delta.detach()) + entropy_loss * .01
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

            
            
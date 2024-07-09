import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class QValueNet(nn.Module):
    """
    The deep Q net in double-DQN architecture.
    Serving as an approximator for the state action value function
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.nn = nn.Sequential(nn.Linear(state_dim, hidden_dim)
                                ,nn.ReLU()
                                ,nn.Linear(hidden_dim, hidden_dim)
                                ,nn.ReLU()
                                ,nn.Linear(hidden_dim, action_dim)
                               )
        self.apply(weight_init)
    
    def forward(self, x):
        return self.nn(x)


class dqn:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, lr, soft, target_update, tau):
        self.policy_net = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.action_dim = action_dim
        self.gamma = gamma
        self.soft = soft
        self.target_update = target_update
        self.tau = tau
        self.counts = 0
        self.steps = 0
        
    def take_action(self, state, EPS_START=.9, EPS_END=.01, EPS_DECAY=1000):
        state = torch.tensor(state, dtype=torch.float).to(device)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-.1 * self.steps / EPS_DECAY)
        self.steps += 1
        if sample > eps_threshold:
            with torch.no_grad():
                Q = self.policy_net(state).cpu().numpy().ravel()
                return random.sample(list(np.where(Q == Q.max())[0]), 1)[0]
        else:
            return random.randint(0, self.action_dim-1)
            
    def max_Q_value(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        return self.policy_net(state).max().item()
        
    def soft_update(self, target_net, net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param + self.tau * param)
        
    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(transition_dict['actions'])).view(-1,1).to(device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1,1).to(device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1,1).to(device)
        q_values = self.policy_net(states).gather(1, actions)
        max_action = self.policy_net(next_states).max(1)[1].view(-1,1)
        max_next_q_values = self.target_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = torch.mean(F.smooth_l1_loss(q_values ,q_targets))
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        if self.soft:
            self.soft_update(self.target_net, self.policy_net)
        elif self.counts % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.counts += 1
        
        
        
        
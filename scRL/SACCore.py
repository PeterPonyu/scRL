import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


def mlp(state_dim, hidden_dim, out_dim):
    """Two hidden layers with ReLU."""
    net = nn.Sequential(nn.Linear(state_dim, hidden_dim)
                        ,nn.ReLU()
                        ,nn.Linear(hidden_dim, hidden_dim)
                        ,nn.ReLU()
                        ,nn.Linear(hidden_dim, out_dim)
                       )
    net.apply(weight_init)
    return net


class sac:
    """
    Soft actor-critic for discrete moves.

    Two Q networks with slowly following target networks, a categorical actor, and a temperature that is
    learned so that the entropy of the actor stays near ``entropy_frac * log(number of moves)``. The value
    of a state is the soft value sum_a pi(a|s) [min Q(s, a) - alpha log pi(a|s)].

    Parameters
    ----------
    state_dim
        Dimension of the state vector.
    hidden_dim
        Units of each hidden layer.
    action_dim
        Number of moves.
    gamma
        Discount factor.
    actor_lr
        Learning rate of the actor.
    critic_lr
        Learning rate of the Q networks.
    tau
        Fraction by which the target networks follow the Q networks after each update.
        (Default: .005)
    alpha_lr
        Learning rate of the temperature.
        (Default: 1e-3)
    entropy_frac
        Target entropy as a fraction of the largest possible entropy, 0 < entropy_frac <= 1.
        (Default: .3)
    """
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, actor_lr, critic_lr, tau=.005, alpha_lr=1e-3,
                 entropy_frac=.3):
        if not 0 < entropy_frac <= 1:
            raise ValueError(f'entropy_frac must satisfy 0 < entropy_frac <= 1, got {entropy_frac}.')
        self.actor = nn.Sequential(mlp(state_dim, hidden_dim, action_dim), nn.Softmax(dim=-1)).to(device)
        self.q1 = mlp(state_dim, hidden_dim, action_dim).to(device)
        self.q2 = mlp(state_dim, hidden_dim, action_dim).to(device)
        self.q1_t = mlp(state_dim, hidden_dim, action_dim).to(device)
        self.q2_t = mlp(state_dim, hidden_dim, action_dim).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float, device=device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = entropy_frac * np.log(action_dim)
        self.gamma = gamma
        self.tau = tau

    def take_action(self, state):
        with torch.no_grad():
            p = self.actor(torch.tensor(np.array([state]), dtype=torch.float, device=device))
        return torch.distributions.Categorical(p).sample().item()

    def value(self, states):
        """Soft value of a batch of states (no gradient)."""
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(states), dtype=torch.float, device=device)
            p = self.actor(s)
            q = torch.min(self.q1(s), self.q2(s))
            return (p * (q - self.log_alpha.exp() * torch.log(p + 1e-8))).sum(-1)

    def v_value(self, state):
        """Soft value of one state."""
        return self.value(np.asarray(state)[None]).item()

    def max_Q_value(self, state):
        with torch.no_grad():
            s = torch.tensor(np.array(state), dtype=torch.float, device=device)
            return torch.min(self.q1(s), self.q2(s)).max().item()

    def update(self, transition_dict):
        td = transition_dict
        s = torch.tensor(np.array(td['states']), dtype=torch.float, device=device)
        a = torch.tensor(np.array(td['actions']), dtype=torch.int64, device=device).view(-1, 1)
        r = torch.tensor(np.array(td['rewards']), dtype=torch.float, device=device).view(-1, 1)
        s2 = torch.tensor(np.array(td['next_states']), dtype=torch.float, device=device)
        d = torch.tensor(np.array(td['dones']), dtype=torch.float, device=device).view(-1, 1)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            p2 = self.actor(s2)
            v2 = (p2 * (torch.min(self.q1_t(s2), self.q2_t(s2)) - alpha * torch.log(p2 + 1e-8))).sum(1, keepdim=True)
            y = r + self.gamma * (1 - d) * v2
        q_loss = F.mse_loss(self.q1(s).gather(1, a), y) + F.mse_loss(self.q2(s).gather(1, a), y)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        p = self.actor(s)
        logp = torch.log(p + 1e-8)
        q = torch.min(self.q1(s), self.q2(s)).detach()
        actor_loss = (p * (alpha * logp - q)).sum(1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        entropy = -(p.detach() * logp.detach()).sum(1).mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        for target, net in ((self.q1_t, self.q1), (self.q2_t, self.q2)):
            for tp, p_ in zip(target.parameters(), net.parameters()):
                tp.data.copy_((1 - self.tau) * tp.data + self.tau * p_.data)

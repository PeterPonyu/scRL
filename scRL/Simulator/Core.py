import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class encoder(nn.Module):
    """
    The encoder for simulator
    """
    def __init__(self, state_dim, hidden_dim, latent_dim, gs):
        super().__init__()
        self.fc1 = nn.Linear(state_dim+gs+3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.apply(weight_init)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)          
    
class decoder(nn.Module):
    """
    The decoder for simulator
    """
    def __init__(self, latent_dim, hidden_dim, state_dim, gs):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, state_dim+gs+3)
        self.fc_sigma = nn.Linear(hidden_dim, state_dim+gs+3)
        self.apply(weight_init)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x) + 1e-8  
        sigma = F.softplus(self.fc_sigma(x)) + 1e-8
        dist = torch.distributions.Normal(mu, sigma)
        return dist.rsample()   
    
class simulator:
    """
    The simulator architecture
    """
    def __init__(self, state_dim, hidden_dim, latent_dim, gs):
        self.encoder = encoder(state_dim, hidden_dim, latent_dim, gs).to(device)
        self.decoder = decoder(latent_dim, hidden_dim, state_dim, gs).to(device)
        self.opt = optim.Adam([{'params':self.encoder.parameters()}
                               ,{'params':self.decoder.parameters()}])
        self.loss_ls = []
        
    def sim(self, x, exp, grid, time):
        x = torch.tensor(x, dtype=torch.float, device=device)
        exp = torch.tensor(exp, dtype=torch.float, device=device)
        grid = torch.tensor(grid, dtype=torch.float, device=device)
        time = torch.tensor(time, dtype=torch.float, device=device)
        latent = self.encoder(torch.cat([x, exp, grid, time]))
        return self.decoder(latent).cpu().detach().numpy()
    
    def update(self, x, exp, grid, time, next_x, next_exp, next_grid, next_time):
        x = torch.tensor(x, dtype=torch.float, device=device)
        exp = torch.tensor(exp, dtype=torch.float, device=device)
        grid = torch.tensor(grid, dtype=torch.float, device=device)
        time = torch.tensor(time, dtype=torch.float, device=device)
        next_x = torch.tensor(next_x, dtype=torch.float, device=device)
        next_exp = torch.tensor(next_exp, dtype=torch.float, device=device)
        next_grid = torch.tensor(next_grid, dtype=torch.float, device=device)
        next_time = torch.tensor(next_time, dtype=torch.float, device=device)
        latent = self.encoder(torch.cat([x, exp, grid, time]))
        pred = self.decoder(latent) 
        pred1 = pred[:-3-len(exp)]
        pred2 = pred[-3-len(exp):-3]
        pred3 = pred[-3:-1]
        pred4 = pred[-1]
        loss = F.smooth_l1_loss(next_x, x+pred1) + F.smooth_l1_loss(next_exp, exp+pred2) + F.smooth_l1_loss(next_grid, grid+pred3) + F.smooth_l1_loss(next_time, time+pred4)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.loss_ls.append(loss.item())
        
        
def get_sim_df(gres, trainer, lineages=None, steps=50, prefix=None):
    """
    Integrating simulated data for down stream analysis
    
    Parameters
    ----------
    gres
        Grid embedding results
    trainer
        A trainer after training in simulating mode
    lineage
        Lineage types for predicting trajectories cell types
        (Default: None)
    steps
        Simulating steps each trajectory
        (Default: 50)
    prefix
        String for identifying multiple lineages
        (Default: None)
        
    Returns
    ----------
    None
    """
    gres.simulating[f'{prefix}_steps'] = steps
    ls_pc = []
    ls_exp = []
    ls_coord = []
    ls_time = []
    for s in gres.grids['starts_cluster_grids']:
        pc, exp, coord, time = get_sim(trainer, gres,[s], steps)
        ls_pc.append(pc)
        ls_exp.append(exp)
        ls_coord.append(coord)
        ls_time.append(time)
    ls = []
    for i, pc in enumerate(ls_pc):
        df = pd.DataFrame(pc)
        ls.append(df)

    df_pc = pd.concat(ls)
    df_exp = pd.concat([pd.DataFrame(le, columns=gres.grids['gene_exp'].columns.values) for le in ls_exp])
    df_coord = pd.concat([pd.DataFrame({'c1':c[:,0],'c2':c[:,1]}) for i, c in enumerate(ls_coord)])
    df_time = pd.concat([pd.DataFrame({'time':t.ravel()}) for i ,t in enumerate(ls_time)])
    
    gres.simulating[f'{prefix}_N'] = df_pc.shape[0]
    gres.simulating[f'{prefix}_df_pc'] = df_pc
    gres.simulating[f'{prefix}_df_exp'] = df_exp
    gres.simulating[f'{prefix}_df_coord'] = df_coord
    gres.simulating[f'{prefix}_df_time'] = df_time
    
    if lineages is not None:
        from sklearn.linear_model import LogisticRegression
        idx = pd.Series(gres.grids['mapped_grids_clusters']).isin(lineages)
        states_train = trainer.env.state_space.mean(axis=1)[idx]
        type_train = gres.grids['mapped_grids_clusters'][idx]
        lr = LogisticRegression().fit(states_train, type_train)
        df_type = pd.DataFrame({'Type':lr.predict(df_pc)},index=df_pc.index)
        gres.simulating[f'{prefix}_df_type'] = df_type
        gres.simulating[f'{prefix}_lineages'] = lineages
    return 

def get_sim(t, gres, starts, n=50):
    """
    Function for simulating with trainer 
    
    Parameters
    ----------
    t
        Trainer class
    gres
        Grid embedding results
    starts
        Sampling starting points
    n
        Simulating steps each trajectory
        (Default: 50)
        
    Returns
    ----------
    Lists for simulated spaces, expressions, embeddings and pseudotime 
    """
    exps = t.gres.grids['gene_exp'].values.astype('float')
    grids = t.gres.grids['grids']
    times = t.gres.grids['pseudotime']
    mapped_grids = t.gres.grids['mapped_grids']
    env = t.env
    
    idx = np.random.choice(starts)
    idx1 = np.where(mapped_grids == idx)[0][0]
    
    state = t.env.state_space.mean(axis=1)[idx1]
    expression = exps[idx1]
    embed = grids[idx]
    time = np.array([times[idx]])
    
    ls_state = [state]
    ls_exp = [expression]
    ls_embed = [embed]
    ls_time = [time]
    
    for i in range(n):
        diff = t.sim.sim(state, expression, embed, time)
        
        next_state = state + diff[:-3-len(expression)]
        next_expression = expression + diff[-3-len(expression):-3]
        next_embed = embed + diff[-3:-1]
        next_time = time + diff[-1]
        
        ls_state.append(next_state)
        ls_exp.append(next_expression)
        ls_embed.append(next_embed)
        ls_time.append(next_time)
        
        state = next_state
        expression = next_expression
        embed = next_embed
        time = next_time
    return np.vstack(ls_state), np.vstack(ls_exp), np.vstack(ls_embed), np.vstack(ls_time)

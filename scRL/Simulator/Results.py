import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import KBinsDiscretizer

def sim_results(gres, n_sample=3000, traj=None, sim_traj=False, prefix=None):
    """
    Plotting function for visualizing the simulating process.
    With an increasing step pattern to illustrate the differentiation.
    
    Parameters
    ----------
    gres
        Grids results after simulating.
    n_sample
        Number of points for efficiency visualization.
        (Default: 3000)
    traj
        The original learned trajectory.
        (Default: None)
    sim_traj
        The simulated points trajectory.
        (Default: False)
    prefix
        String for identifying multiple lineages
        (Default: None)
        
    Returns
    ----------
    None
    """
    gres.simulating[f'{prefix}_n_sample'] = n_sample
    idx = np.random.randint(0, gres.simulating[f'{prefix}_N'], n_sample)
    gres.simulating[f'{prefix}_sample_idx'] = idx
    df_coord1 = gres.simulating[f'{prefix}_df_coord'].iloc[idx, :]
    df_time1 = gres.simulating[f'{prefix}_df_time'].iloc[idx, :]
    df_type1 = gres.simulating[f'{prefix}_df_type'].iloc[idx, :]
    
    with sns.axes_style('white'):
        import os
        if not os.path.exists('sim_output'):
            os.mkdir('sim_output')
        
        lut = dict(zip(gres.embedding['clusters'], gres.embedding['cluster_colors']))
        mask = gres.embedding['clusters'].isin(gres.simulating[f'{prefix}_lineages'])
        fig = plt.figure(figsize=(5,5))
        if traj:
            from ..Trajectory.Results import traj_results
            traj_results(gres, gres.simulating[f'{prefix}_lineages'], traj)
        else:
            sns.scatterplot(x=gres.embedding['embedding'][:,0], y=gres.embedding['embedding'][:,1]
                            , color='lightgrey', linewidth=0)
            sns.scatterplot(x=gres.embedding['embedding'][mask,0], y=gres.embedding['embedding'][mask,1]
                            , c=gres.embedding['cluster_colors'][mask], linewidth=0)
            ax = plt.gca()
            ax.set_frame_on(False)
            ax.tick_params(labelleft=False,labelbottom=False)
            ax.set_xlabel('')
            ax.set_ylabel('')
        plt.savefig('sim_output/lineages.png',dpi=600,bbox_inches='tight')
        
        fig = plt.figure(figsize=(5,5))
        sns.scatterplot(x=gres.embedding['embedding'][:,0], y=gres.embedding['embedding'][:,1]
                            , color='lightgrey', linewidth=0)
        sns.scatterplot(x=gres.embedding['embedding'][mask,0], y=gres.embedding['embedding'][mask,1]
                        ,c=gres.embedding['pseudotime'][mask], cmap='viridis', linewidth=0)
        ax = plt.gca()
        ax.set_frame_on(False)
        ax.tick_params(labelleft=False,labelbottom=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.savefig('sim_output/pseudotime.png',dpi=600,bbox_inches='tight')
        steps = gres.simulating[f'{prefix}_steps']
        
        fig = plt.figure()
        sns.scatterplot(data=df_coord1, x='c1', y='c2')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_visible(False)
        
        for t in [steps*.2, steps*.4, steps*.6, steps*.8, steps]:
            #Differentiation trajectory for each time stage
            idx1 = df_time1.index < t
            fig = plt.figure(figsize=(5,5))
            sns.scatterplot(data=df_coord1.loc[idx1, :], x='c1', y='c2', c=[lut[c] for c in df_type1['Type'][idx1]], linewidth=0)
            ax = plt.gca()
            ax.set_frame_on(False)
            ax.tick_params(labelleft=False,labelbottom=False)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if sim_traj:
                df_coord = gres.simulating[f'{prefix}_df_coord'].copy()
                df_time = gres.simulating[f'{prefix}_df_time'].copy()
                df_coord['bins'] = KBinsDiscretizer(df_coord.index.max(), encode='ordinal', strategy='uniform').fit_transform(df_time.values).ravel()
                df_coord['steps'] = df_coord.index
                start_num = len(gres.grids['starts_cluster_grids'])
                sim_traj_df = df_coord[df_coord['bins'] <= df_coord['steps']].groupby('bins').apply(lambda x : x.mean() if x.shape[0] > .2 * start_num else None).dropna()
                gres.simulating[f'{prefix}_df_traj'] = sim_traj_df
                traj_num = len(sim_traj_df)
                for i in range(traj_num-1):
                    c = mpl.patches.ConnectionPatch(sim_traj_df[['c1','c2']].iloc[i,:], sim_traj_df[['c1','c2']].iloc[i+1,:]
                                                    , ax.transData ,arrowstyle='->'
                                                    , color=mpl.cm.rainbow(np.linspace(0,1,traj_num))[i]
                                                    , lw=2, mutation_scale=20, capstyle='round')
                    ax.add_patch(c)
            plt.savefig(f'sim_output/types_{t}.png',dpi=600,bbox_inches='tight')
            #Pseudotime for simulated cells
            fig = plt.figure(figsize=(5,5))
            sns.scatterplot(data=df_coord1.loc[idx1, :], x='c1', y='c2', c=df_time1['time'][idx1], cmap='viridis', vmax=1, linewidth=0)
            ax = plt.gca()
            ax.set_frame_on(False)
            ax.tick_params(labelleft=False,labelbottom=False)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if sim_traj:
                for i in range(traj_num-1):
                    c = mpl.patches.ConnectionPatch(sim_traj_df[['c1','c2']].iloc[i,:], sim_traj_df[['c1','c2']].iloc[i+1,:]
                                                    , ax.transData ,arrowstyle='->'
                                                    , color=mpl.cm.rainbow(np.linspace(0,1,traj_num))[i], lw=2, mutation_scale=20, capstyle='round')
                    ax.add_patch(c)
            plt.savefig(f'sim_output/time_{t}.png',dpi=600,bbox_inches='tight')
            #Pie plots for simulated cell types proportion
            types = np.unique(df_type1['Type'][idx1], return_counts=True)[0]
            nums = np.unique(df_type1['Type'][idx1], return_counts=True)[1]
            lut1 = dict(zip(types, nums))
            fig = plt.figure(figsize=(5,5))
            plt.pie([lut1[c] for c in types], startangle=90
                    ,colors=[lut[c] for c in types], wedgeprops={'lw':0})
            plt.savefig(f'sim_output/pie_{t}.png',dpi=600,bbox_inches='tight')
    return


def sim_results2(gres, lin1, lin2, prefix=None):
    """
    Plotting funtion for comparing the two lineages correlation.
    A stem lineage and a mature lineage are preferred for distinctive comparison.
    
    Parameters
    ----------
    gres
        Grids results after simulating mode training.
    lin1
        Lineage 1 for correlation comparision.
    lin2
        Lineage 2 for correlation comparision.
    prefix
        String for identifying multiple lineages
        (Default: None)
        
    Returns
    ----------
    None
    """
    #Save the comparision information 
    gres.simulating[f'{prefix}_lin1'] = lin1
    gres.simulating[f'{prefix}_lin2'] = lin2
    #Embedding pre-setttings
    X_pca = gres.embedding['X_latent']
    exp = gres.embedding['gene_exp']
    pseudotime = gres.embedding['pseudotime']
    clusters = gres.embedding['clusters']
    cluster_colors = gres.embedding['cluster_colors']
    lut = dict(zip(clusters, cluster_colors))
    #Simulating pre-settings
    N = gres.simulating[f'{prefix}_N']
    df_pc = gres.simulating[f'{prefix}_df_pc']
    df_exp = gres.simulating[f'{prefix}_df_exp']
    df_time = gres.simulating[f'{prefix}_df_time']
    df_type = gres.simulating[f'{prefix}_df_type']
    sample_idx = gres.simulating[f'{prefix}_sample_idx']
    #Plot settings
    def trend_plot(df, key, mode):
        if key == 'steps':
            n = sorted(df['steps'].unique())
        else:
            n = sorted(df['bins'].unique())
        import os
        if not os.path.exists('sim_output'):
            os.mkdir('sim_output')
        mu1 = df.groupby(key).mean()['lin1']
        mu2 = df.groupby(key).mean()['lin2']
        sigma1 = df.groupby(key).std()['lin1']
        sigma2 = df.groupby(key).std()['lin2']
        #Correlation trend for each lineage along simulated steps
        with sns.axes_style('darkgrid'):
            fig = plt.figure(figsize=(5,5))
            plt.plot(mu1, lw=5,color=lut[lin1[0]])
            plt.fill_between(n, mu1-sigma1, mu1+sigma1,color=lut[lin1[0]], alpha=.25, lw=0)
            plt.plot(mu2, lw=5,color=lut[lin2[0]])
            plt.fill_between(n, mu2-sigma2, mu2+sigma2,color=lut[lin2[0]], alpha=.25, lw=0)
            ax = plt.gca()
            ax.grid(axis='x',visible=False)
            plt.savefig(f'sim_output/{mode}_{key}_corr.png',dpi=600,bbox_inches='tight')
        return 
    
    #For latent space correlation 
    lin1_state = pd.DataFrame(X_pca[clusters.isin(lin1)])
    lin2_state = pd.DataFrame(X_pca[clusters.isin(lin2)])
    time = df_time.values
    bins = KBinsDiscretizer(50, encode='ordinal').fit_transform(time).ravel()
    lin1_corr = np.corrcoef(df_pc, lin1_state)[:N, N:].mean(axis=1)
    lin2_corr = np.corrcoef(df_pc, lin2_state)[:N, N:].mean(axis=1)
    df_pc_corr = pd.DataFrame({'lin1':lin1_corr, 'lin2':lin2_corr,'steps':df_time.index, 'bins':bins, 'time':time.ravel()})
    gres.simulating[f'{prefix}_df_pc_corr'] = df_pc_corr
    trend_plot(df_pc_corr, 'steps', 'pc')
    trend_plot(df_pc_corr, 'bins', 'pc')
    
    #For gene expression correlation
    exp_corr = np.corrcoef(df_exp, exp)
    exp_corr1 = exp_corr[:N, N:][:,clusters.isin(lin1)].mean(axis=1)
    exp_corr2 = exp_corr[:N, N:][:,clusters.isin(lin2)].mean(axis=1)
    df_exp_corr = pd.DataFrame({'lin1':exp_corr1, 'lin2':exp_corr2,'steps':df_time.index, 'bins':bins, 'time':time.ravel()})
    gres.simulating[f'{prefix}_df_exp_corr'] = df_exp_corr
    trend_plot(df_exp_corr, 'steps', 'exp')
    trend_plot(df_exp_corr, 'bins', 'exp')
    
    #For oringinal latent space correlation
    ori_pc_corr = np.corrcoef(X_pca)
    ori_pc_corr1 = ori_pc_corr[clusters.isin(lin1+lin2),:][:,clusters.isin(lin1)].mean(axis=1)
    ori_pc_corr2 = ori_pc_corr[clusters.isin(lin1+lin2),:][:,clusters.isin(lin2)].mean(axis=1)
    ori_pc_df = pd.DataFrame({'lin1':ori_pc_corr1,'lin2':ori_pc_corr2, 'time':pseudotime[clusters.isin(lin1+lin2)]})
    ori_pc_df['bins'] = KBinsDiscretizer(50, encode='ordinal').fit_transform(ori_pc_df['time'].values.reshape(-1,1)).ravel()
    gres.embedding['ori_pc_df'] = ori_pc_df
    trend_plot(ori_pc_df, 'bins', 'ori_pc')
    
    #For oringinal expression space correlation
    ori_exp_corr = np.corrcoef(exp)
    ori_exp_corr1 = ori_exp_corr[clusters.isin(lin1+lin2),:][:,clusters.isin(lin1)].mean(axis=1)
    ori_exp_corr2 = ori_exp_corr[clusters.isin(lin1+lin2),:][:,clusters.isin(lin2)].mean(axis=1)
    ori_exp_df = pd.DataFrame({'lin1':ori_exp_corr1,'lin2':ori_exp_corr2, 'time':pseudotime[clusters.isin(lin1+lin2)]})
    ori_exp_df['bins'] = KBinsDiscretizer(50, encode='ordinal').fit_transform(ori_exp_df['time'].values.reshape(-1,1)).ravel()
    gres.embedding['ori_exp_df'] = ori_exp_df
    trend_plot(ori_exp_df, 'bins', 'ori_exp')
    
    cols_sim = np.array([lut[t] for t in df_type['Type']])[sample_idx]
    cols = cluster_colors[clusters.isin(lin1+lin2)]
    
    pc_corr_sim_self = np.corrcoef(df_pc)
    pc_corr_sim_self1 = pc_corr_sim_self[sample_idx, :][:,sample_idx]
    sns.clustermap(pc_corr_sim_self1, xticklabels=False,yticklabels=False, col_colors=cols_sim, row_colors=cols_sim
                   , figsize=(5,5), cmap='rainbow', center=0)
    plt.savefig('sim_output/sim_pc_clustermap.png',dpi=600,bbox_inches='tight')
    
    pc_corr_self = np.corrcoef(X_pca)
    pc_corr_self1 = pc_corr_self[clusters.isin(lin1+lin2), :][:,clusters.isin(lin1+lin2)]
    sns.clustermap(pc_corr_self1, xticklabels=False,yticklabels=False, col_colors=cols, row_colors=cols
                   , figsize=(5,5), cmap='rainbow', center=0)
    plt.savefig('sim_output/pc_clustermap.png',dpi=600,bbox_inches='tight')
    
    exp_corr_sim_self = np.corrcoef(df_exp)
    exp_corr_sim_self1 = exp_corr_sim_self[sample_idx, :][:,sample_idx]
    sns.clustermap(exp_corr_sim_self1, xticklabels=False,yticklabels=False, col_colors=cols_sim, row_colors=cols_sim
                   , figsize=(5,5), cmap='rainbow', center=0)
    plt.savefig('sim_output/sim_exp_clustermap.png',dpi=600,bbox_inches='tight')
    
    exp_corr_self = np.corrcoef(exp)
    exp_corr_self1 = exp_corr_self[clusters.isin(lin1+lin2),:][:,clusters.isin(lin1+lin2)]
    sns.clustermap(exp_corr_self1, xticklabels=False,yticklabels=False, col_colors=cols, row_colors=cols
                   , figsize=(5,5), cmap='rainbow', center=0)
    plt.savefig('sim_output/exp_clustermap.png',dpi=600,bbox_inches='tight') 
    
    return



def sim_results3(gres, top = 10, ma = 11, prefix=None):
    """
    Identifying the most and least time related genes in simulating.
    Plotting the correlation score and expression trend for both simulated and original genes.
    
    Parameters
    ----------
    gres
        Grid results after get simulating data.
    top
        Top genes for time correlation identifying.
        (Default: 10)
    ma
        Moving average window size for original gene expression along pseudotime.
        (Default: 11)
    prefix
        String for identifying multiple lineages.
        (Default: None)
        
    Returns
    ----------
    None
    """
    #Pre-setttings for plotting
    df_exp = gres.simulating[f'{prefix}_df_exp'].copy()
    df_time = gres.simulating[f'{prefix}_df_time'].copy()
    exp = gres.embedding['gene_exp']
    time = gres.embedding['pseudotime']
    clusters = gres.embedding['clusters']
    lineages = gres.simulating[f'{prefix}_lineages']
    
    #Calculating the simulated gene correlation with pseudotime
    df_gene_corr = pd.DataFrame({'time_corr':np.corrcoef(df_exp.T, df_time.T)[:-1,-1]}, index=df_exp.columns).sort_values('time_corr', ascending=False)
    gres.simulating[f'{prefix}_df_gene_corr'] = df_gene_corr
    df_bar = pd.concat([df_gene_corr.head(top), df_gene_corr.tail(top)], axis=0)
    g = df_bar.index
    
    import os
    if not os.path.exists('sim_output'):
        os.mkdir('sim_output')
    #Plotting the simulated genes correlation bar
    pos_col = sns.color_palette('RdBu',8)[0]
    neg_col = sns.color_palette('RdBu',8)[-1]
    fig = plt.figure(figsize=(3,5))
    sns.barplot(x=df_bar['time_corr'], y=g, orient='horizonal',palette='RdBu')
    ax = plt.gca()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(-1,1)
    ax.tick_params(left=False,labelleft=False)
    ax.spines[['top','right','left']].set_visible(False)
    ax.axvline(0,0,1,color='k')
    for i, t in enumerate(g):
        ax.text(0, i,t, fontdict={'fontsize':14,'color':pos_col if i < 10 else neg_col
                                  ,'ha':'right' if i < 10 else 'left','va':'center'})
    plt.savefig('sim_output/sim_genes_corr_bar.png',dpi=600,bbox_inches='tight')
    #Plotting the original genes correlation bar
    exp1 = exp.loc[clusters.isin(lineages)].copy()
    time1 = time[clusters.isin(lineages)]
    fig = plt.figure(figsize=(3,5))
    sns.barplot(x=np.corrcoef(exp1[g].T, time1.T)[1:,0].astype(np.float16),y=g, orient='horizonal',palette='RdBu')
    ax = plt.gca()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(-1,1)
    ax.tick_params(left=False,labelleft=False)
    ax.spines[['top','right','left']].set_visible(False)
    ax.axvline(0,0,1,color='k')
    for i, t in enumerate(g):
        ax.text(0, i,t, fontdict={'fontsize':14,'color':pos_col if i < 10 else neg_col
                                  ,'ha':'right' if i < 10 else 'left','va':'center'})
    plt.savefig('sim_output/genes_corr_bar.png',dpi=600,bbox_inches='tight')
    df_exp['steps'] = df_exp.index
    df_exp1 = df_exp.groupby('steps').mean()
    mu1 = df_exp1[g[:10]].T.melt().groupby('steps').mean()['value'].to_numpy()
    sigma1 = df_exp1[g[:10]].T.melt().groupby('steps').std()['value'].to_numpy()
    mu2 = df_exp1[g[10:]].T.melt().groupby('steps').mean()['value'].to_numpy()
    sigma2 = df_exp1[g[10:]].T.melt().groupby('steps').std()['value'].to_numpy()
    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=(5, 5))
        plt.plot(mu1,color=pos_col,lw=5)
        plt.fill_between(np.arange(df_exp.index.max()+1), mu1-sigma1, mu1+sigma1, color=pos_col,lw=0, alpha=.25)
        plt.plot(mu2,color=neg_col,lw=5)
        plt.fill_between(np.arange(df_exp.index.max()+1), mu2-sigma2, mu2+sigma2, color=neg_col,lw=0, alpha=.25)
        ax = plt.gca()
        ax.grid(axis='x',visible=False)
        plt.savefig('sim_output/simulated_genes_trend.png',dpi=600,bbox_inches='tight')
    exp1['bins'] = KBinsDiscretizer(df_exp.index.max(), encode='ordinal', strategy='uniform').fit_transform(time1.reshape(-1,1))
    exp1 = exp1.groupby('bins').mean()
    
    mu1 = exp1[g[:10]].T.melt().groupby('bins').mean()['value'].to_numpy()
    sigma1 = exp1[g[:10]].T.melt().groupby('bins').std()['value'].to_numpy()
    mu2 = exp1[g[10:]].T.melt().groupby('bins').mean()['value'].to_numpy()
    sigma2 = exp1[g[10:]].T.melt().groupby('bins').std()['value'].to_numpy()
    if ma:
        from ..utils import moving_average
        mu1 = moving_average(mu1, ma)
        sigma1 = moving_average(sigma1, ma)
        mu2 = moving_average(mu2, ma)
        sigma2 = moving_average(sigma1, ma)
    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=(5, 5))
        plt.plot(mu1,color=pos_col,lw=5)
        plt.fill_between(np.arange(exp1.index.unique().shape[0]), mu1-sigma1, mu1+sigma1, color=pos_col,lw=0, alpha=.25)
        plt.plot(mu2,color=neg_col,lw=5)
        plt.fill_between(np.arange(exp1.index.unique().shape[0]), mu2-sigma2, mu2+sigma2, color=neg_col,lw=0, alpha=.25)
        ax = plt.gca()
        ax.grid(axis='x',visible=False)
        plt.savefig('sim_output/genes_trend.png',dpi=600,bbox_inches='tight')
    return 


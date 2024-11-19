import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import time
import tqdm
from joblib import Parallel, delayed
from .utils import get_dist

class Grids_Results():
    """Container for the grid results
    """
    def __init__(self):
        self._embedding = {}
        self._grids = {}
        self._qlearning = {}
        self._simulating = {}
        self._trajectory = {}
    @property
    def embedding(self):
        return self._embedding
    @property
    def grids(self):
        return self._grids
    @property
    def qlearning(self):
        return self._qlearning
    @property
    def simulating(self):
        return self._simulating
    @property
    def trajectory(self):
        return self._trajectory

def grids_from_embedding(X,
                          n=50,
                          j=3,
                          n_jobs=8
                         ):
    """
    Function for generating grids embedding
    Assuming that most of the two-dimensional embeddings of large scale single-cell data can represent 
    the generation process and inherent dynamics of the data to some extent, a derivative grids representaion 
    of the embedding therefore can provide us with a more simplified and comprehensive perspective of the data space.
    
    Parameters
    ----------
    X
        A 2-D embedding space
    n
        The grids number for boundary generation
        (Default: 50)
    j
        The observer number for mask generation
        (Default: 3)
    n_jobs
        Number of cores to use
        (Default: 8)
        
    Returns
    ----------
    Grids results with mapped information
    """
    start = time.time()
    def generate_grids(X=X,n=n):
        right = X[np.argmax(X[:,0]),:]
        left = X[np.argmin(X[:,0]),:]
        top = X[np.argmax(X[:,1]),:]
        bottom = X[np.argmin(X[:,1]),:]
        x = np.linspace(left[0],right[0],num=n)
        y = np.linspace(bottom[1],top[1],num=n)
        xv, yv = np.meshgrid(x,y,indexing='ij')
        grids = np.vstack([xv.ravel(),yv.ravel()]).T
        spines = []
        bottom_spine = np.vstack([xv[:,0],yv[:,0]]).T
        top_spine = np.vstack([xv[:,-1],yv[:,-1]]).T
        left_spine = np.vstack([xv[0,1:-1],yv[0,1:-1]]).T
        right_spine = np.vstack([xv[-1,1:-1],yv[-1,1:-1]]).T
        spines = np.vstack([bottom_spine, top_spine, left_spine, right_spine])
        return grids, spines
    grids, spines = generate_grids(X,n)
    
    def get_arc_dist(X1, X2, i, decimals=1):
        G = np.around(np.arctan((X1[i,0] - X2[:,0] + 1e-6) / (X1[i,1] - X2[:,1] + 1e-6))
                      ,decimals=decimals)
        D = get_dist(X1[i,:].reshape(1,-1), X2)
        return pd.DataFrame({'G':G,'D':D[0]})
    
    def get_boundary(i):
        arc_dist = get_arc_dist(spines, X, i)
        B = arc_dist.groupby('G').apply(lambda x : x['D'].idxmax()).values
        return B
    
    boundaries = Parallel(n_jobs=n_jobs)(delayed(get_boundary)(i)
                                         for i in tqdm.tqdm(range(len(spines))
                                                            ,desc='Boundary generating'
                                                            ))    
    boundaries = np.unique(np.hstack(boundaries))
    B_on_grids = np.argmin(get_dist(X[boundaries,:],grids),axis=1)
    if j == 1:
        right = X[np.argmax(X[:,0]),0]
        left = X[np.argmin(X[:,0]),0]
        top = X[np.argmax(X[:,1]),1]
        bottom = X[np.argmin(X[:,1]),1]
        xmid = left + (right-left) / 2
        ymid = bottom + (top-bottom) / 2
        spines2 = np.array([[xmid,bottom],[xmid,top],[left,ymid],[right,ymid]])
    else:
        _, spines2 = generate_grids(X,j)
    
    
    def get_mask(i):
        arc_dist = get_arc_dist(spines2, grids, i)
        tmp = arc_dist.groupby('G').apply(lambda x : x['D'].sort_values())
        grids_edges = []
        for idx in np.unique(tmp.index.get_level_values('G')):
            M = [np.where(tmp[idx,:].index == g)[0][0] 
                 for g in B_on_grids if g in tmp[idx,:].index]
            if len(M) == 0:
                continue
            grids_edges.append(tmp[idx].index[max(M)+j:].values)
        if len(grids_edges) == 0:
            return 
        grids_edges = np.hstack(grids_edges)   
        return grids_edges
    
    masked_grids = Parallel(n_jobs=n_jobs)(delayed(get_mask)(i) 
                                           for i in tqdm.tqdm(range(len(spines2))
                                                              ,desc='Mask testing'
                                                              ))
    masked_grids = np.unique(np.hstack(masked_grids))
    gres = Grids_Results()
    gres.embedding['embedding'] = X
    gres.embedding['boundaries'] = boundaries
    gres.grids['n'] = n
    gres.grids['grids'] = grids
    mapped_grids = np.array(list(set(i for i in range(len(grids))).difference(masked_grids)))
    
    def get_adjacent(n,masked_grids,mapped_grids):
        mat = np.ones((n,n)).ravel()
        mat[masked_grids] = 0
        mat = mat.reshape(n,n,order='F')
        adj = pd.DataFrame(0,index=list(mapped_grids),columns=list(mapped_grids))
        pbar = tqdm.tqdm(total=len(mapped_grids), desc='Adjacent generating')
        for idx in mapped_grids:
            i = idx%mat.shape[0]
            j = idx//mat.shape[0]
            L = len(mat)-1
            for D in [(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1),(i,j-1),(i-1,j-1),(i-1,j),(i-1,j+1)]:
                if  -1 < D[0] < n and -1 < D[1] < n:
                    if (D[0]+D[1]*n) in adj.index: 
                        adj.loc[D[0]+D[1]*n, idx] = mat[D]
            pbar.update(1)
        pbar.close()
        return adj
    
    adj = get_adjacent(n,masked_grids,mapped_grids)
    mapped_boundary = mapped_grids[np.where(adj.sum(axis=1) < 8)[0]]
    
    def check_boundary(n,mapped_grids,mapped_boundary,adj):
        mapped_grids = set(mapped_grids)
        pbar = tqdm.tqdm(total=len(mapped_boundary), desc='Boundary pruning')
        for b in mapped_boundary:
            adjacent = adj.index[adj[b] == 1]
            if all([idx in mapped_boundary for idx in adjacent]):
                for idx in adjacent:
                    i = idx%n
                    j = idx//n
                    for D in [(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1),(i,j-1),(i-1,j-1),(i-1,j),(i-1,j+1)]:
                        if  -1 < D[0] < n and -1 < D[1] < n:
                            idx_around = D[0]+D[1]*n
                            if idx_around in np.arange(n*n):
                                mapped_grids.add(idx_around)
            pbar.update(1)
        pbar.close()
        mapped_grids = np.array(list(mapped_grids))
        return mapped_grids
    
    mapped_grids = check_boundary(n,mapped_grids,mapped_boundary,adj)
    masked_grids = np.array(list(set(i for i in range(len(grids))).difference(mapped_grids)))
    
    adj = get_adjacent(n, masked_grids, mapped_grids)
    G = nx.from_numpy_array(adj.values)
    components = len([c for c in nx.connected_components(G)])
    if components > 1:
        print(f'Warning: There are {components} components in the graph, please consider to reduce the parameter "n" or "j".')
    gres.grids['masked_grids'] = masked_grids
    gres.grids['mapped_grids'] = mapped_grids
    gres.grids['mapped_adj'] = adj
    gres.grids['mapped_boundary'] = mapped_grids[np.where(adj.sum(axis=1) < 8)[0]]
    gres.grids['mapped_travel'] = mapped_grids[np.where(adj.sum(axis=1) == 8)[0]]
    end = time.time()
    print(f'Time used for mapping grids: {(end - start):.2f} seconds')
    return gres


def project_cluster(gres,
                     clusters=None,
                     cluster_colors=None
                    ):
    """
    Function for projecting annotations
    When analyzing single-cell data, it is necessary to assemble clusters of similar cells,
    typically based on a network of neighbors extracted from any low-dimensional embedding.
    As the identification of neighbors frequently relies on the Euclidean distance as a measure of closeness,
    we can therefore use this metric to align cluster labels to the gridded representation.
    
    Parameters
    ----------
    gres
        Grids results
    clusters
        Annotations for cells.
        (Default: None)
    cluster_colors
        Colors for categorical annotation groups.
        (Default: None)
        
    Returns
    ----------
    None
    """
    start = time.time()
    X = gres.embedding['embedding']
    grids = gres.grids['grids'] 
    mapped_boundary = gres.grids['mapped_boundary']
    mapped_grids = gres.grids['mapped_grids']
    if clusters is not None:
        gres.embedding['clusters'] = clusters
        
        cluster_lut = dict(zip(range(0,len(X)),clusters))
        mapped_grids_clusters = np.array([cluster_lut[i] 
                                          for i in np.argmin(get_dist(grids[mapped_grids,:],X)
                                                             ,axis=1)])
        if cluster_colors is None:
            cluster_colors = sns.blend_palette([(0.074, 0.403, 0.619),
                                                 (0.670, 0.227, 0.161),
                                                 (0.815, 0.498, 0.172),
                                                 (0.118, 0.486, 0.291),
                                                 (0.435, 0.427, 0.631),
                                                 (0.412, 0.565, 0.635)]
                                              ,n_colors=len(np.unique(clusters)))
        color_lut = dict(zip(clusters.cat.categories, cluster_colors))
        gres.embedding['cluster_colors'] = np.array([color_lut[i] for i in clusters])
        mapped_grids_colors = np.array([color_lut[i] for i in mapped_grids_clusters])
        gres.grids['mapped_grids_clusters'] = mapped_grids_clusters
        gres.grids['mapped_grids_colors'] = mapped_grids_colors
        end = time.time()
        print(f'Time used for projecting annotation : {(end - start):.2f} seconds')
    else:
        print('Warning: No annotation is provided!')
    return


def align_pseudotime(gres,
                      early_cluster,
                      n_sample_cells=10,
                      key_add='pseudotime',
                      boundary=True,
                      early_cell=None,
                      n_jobs=8
                     ):
    """
    This function aligns pseudotime across a grid-based representation of data.
    It can start from a predefined early cell or a cluster, optionally restricting start points to grid boundaries.
    Pseudotime is calculated using Dijkstra's algorithm for shortest paths in a graph constructed from the grid adjacency matrix.
    The function handles multiple graph components by connecting them and adjusting pseudotime accordingly.
    The result is normalized and stored.
    
    Parameters
    ----------
    gres
        Grids results
    early_cluster
        User defined early cluster which must be matched to the grids cluster name, ignored if early cell is set.
    n_sample_cells
        Number of cells to sample in early cluster
        (Default: 10)
    key_add
        The added key for pseudo-time
        (Default: 'pseudotime')
    boundary
        Whether restrict the starting points to the grid boundary
        (Default: True)
    early_cell
        User defined starting cell
        (Default: None)
    n_jobs
        Number of cores to use
        (Default: 8)
    
    Returns
    ----------
    None
    """
    start = time.time()
    grids = gres.grids['grids']
    mapped_grids = gres.grids['mapped_grids']
    mapped_grids_clusters = gres.grids['mapped_grids_clusters']
    if early_cell:
        X = gres.embedding['embedding'][early_cell]
        D = np.linalg.norm(X-grids[mapped_grids],axis=1)
        start_point = [mapped_grids[np.where(D==D.min())[0][0]]]
    else:
        if boundary:
            mapped_boundary = gres.grids['mapped_boundary']
            certain_clutser = mapped_grids[np.where(mapped_grids_clusters==early_cluster)[0]]
            certain_boundary = list(set(certain_clutser)&set(mapped_boundary))
            start_point = np.random.choice(certain_boundary, n_sample_cells)
        else:
            certain_clutser = mapped_grids[np.where(mapped_grids_clusters==early_cluster)[0]]
            start_point = np.random.choice(certain_clutser, n_sample_cells)
    adj = gres.grids['mapped_adj']
    G = nx.from_numpy_array(adj.values)
    G = nx.relabel.relabel_nodes(G, dict(enumerate(adj.columns)), copy=True)
    components = [g for g in nx.connected_components(G)]
    
    def Dijkstra(G,i):
        return nx.single_source_dijkstra_path_length(G, i)
    if len(components) > 1:
        print('Warning: The largest component of the graph is used for pseudo-time aligning')
        
        pointer = [sum([s in c for s in start_point]) for c in components]
        main_idx = pointer.index(max(pointer))
        main_c = components[main_idx]
        main_G = G.subgraph(main_c)
        
        start_point = set(start_point) & set(main_G.nodes)
        sampled_time = Parallel(n_jobs=n_jobs)(delayed(Dijkstra)(main_G,i) for i in start_point)
        mean_time = pd.DataFrame(sampled_time).mean()
        con_ls = []
        comps = components[:main_idx] + components[main_idx+1:]
        for c in comps:
            con_D = get_dist(grids[list(main_c)], grids[list(c)])
            con_grid = np.where(con_D == con_D.min())
            try:
                con_start = mean_time[list(main_c)[con_grid[0][0]]]
            except Exception as e:
                con_start = mean_time.max()
                print(f'An eception occurred: {e} and the maximal time is used.')
            con_ls.append(pd.Series(Dijkstra(G.subgraph(c),list(c)[con_grid[1][0]])) + con_start)
        con_time = pd.concat(con_ls)
        all_time = pd.concat([mean_time, con_time])[mapped_grids]
        gres.grids[key_add] = (all_time - all_time.min()) / (all_time.max() - all_time.min())
    else:
        sampled_time = Parallel(n_jobs=n_jobs)(delayed(Dijkstra)(G,i) for i in start_point)
        mean_time = pd.DataFrame(sampled_time).mean()[mapped_grids]
        gres.grids[key_add] = (mean_time - mean_time.min()) / (mean_time.max() - mean_time.min())
    end = time.time()
    print(f'Time used for aligning pseudo-time : {(end - start):.2f} seconds')
    return


def project_back(gres,
                  key,
                  neighbors=15,
                  w=None,
                  negative=False
                 ):
    """
    This function projects grid-based data back to the original points.
    It uses a Gaussian kernel to weight the influence of the nearest grids on each point, optionally adjusted by a user-defined weighting factor.
    The results can be scaled to positive values and are stored back in the original data structure.
    
    Parameters
    ----------
    gres
        Grids results
    key
        Key in the grids dict which must be a continuous variable
    neighbors
        Nearest grid number to be considered
        (Default: 15)
    w
        Annotation to weight the projected pseudotime at cell level
        (Default: None)
    negative
        Whether the value is non zero or not
        (Default: False)
    
    Returns
    ----------
    None
    """
    X = gres.embedding['embedding']
    grids = gres.grids['grids']
    mapped_grids = gres.grids['mapped_grids']
    grids = grids[mapped_grids]
    D = get_dist(grids,X)
    min_idx = np.argsort(D,axis=0)[:neighbors]
    val = []
    for col in range(min_idx.shape[1]):
        sigma = np.std(D[min_idx[:,col],col])
        D_min = D[min_idx[:,col],col]
        weight = np.exp(-D_min**2 / (2*sigma**2))
        weight = weight / weight.sum()
        t = (weight * np.array(gres.grids[key])[min_idx[:,col]]).sum()
        val.append(t)
    val = np.array(val)
    if w:
        val = val * np.log(w)
    if negative:
        gres.embedding[key] = val
    else:
        gres.embedding[key] = (val - val.min()) / (val.max() - val.min())
    return

def project(gres,
            data,
            neighbors=15
           ):
    """
    This function projects data from cells onto a grid-based representation.
    It calculates the weighted sum of the nearest cell data for each grid, using a Gaussian kernel to determine the weights.
    The projected data is then stored for further analysis.
    
    Parameters
    ----------
    gres
        Grids results
    Value
        Dataframe [Cell X Data]
    neighbors
        Nearest cell number to be considered
        (Default: 15)
        
    Returns
    ----------
    None
    """
    gres.embedding['data'] = data
    X = gres.embedding['embedding']
    grids = gres.grids['grids']
    mapped_grids = gres.grids['mapped_grids']
    
    data = data.reset_index(drop=True)
    grids = grids[mapped_grids]
    D = get_dist(X, grids)
    idx = data.index
    min_idx = np.argsort(D, axis=0)[:neighbors]
    exp = pd.DataFrame(columns=data.columns)
    for g in data.columns:
        for col in range(min_idx.shape[1]):
            sigma = np.std(D[min_idx[:,col],col])
            D_min = D[min_idx[:,col],col]
            weight = np.exp(-D_min**2 / (2*sigma**2))
            if weight.sum() == 0:
                exp.loc[col,g] = 0
            else:
                weight = weight / weight.sum()
                exp.loc[col,g] = (weight * np.array(data.loc[min_idx[:,col],g])).sum()
    gres.grids['proj'] = exp
    return

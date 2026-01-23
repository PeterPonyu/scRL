from .GridCore import grids_from_embedding, align_pseudotime, project_cluster, project_back, project
from .EnvironmentCore import d_rewards, c_rewards
from .Trainer import trainer
from .Simulator.Core import get_sim_df
from .Simulator.Results import sim_results, sim_results2, sim_results3
from .Trajectory.Core import get_traj_df
from .Trajectory.Results import traj_results
from .utils import get_state_value

# Backward compatibility aliases for deprecated function names
lineage_rewards = d_rewards  # Renamed: lineage_rewards -> d_rewards
gene_rewards = c_rewards     # Renamed: gene_rewards -> c_rewards  
project_expression = project  # Renamed: project_expression -> project

__all__ = ['grids_from_embedding', 'align_pseudotime', 'project_cluster', 'project_back', 'project'
           , 'd_rewards', 'c_rewards', 'trainer'
           , 'get_sim_df', 'sim_results', 'sim_results2', 'sim_results3'
           , 'get_traj_df', 'traj_results'
           , 'get_state_value'
           # Backward compatibility
           , 'lineage_rewards', 'gene_rewards', 'project_expression']

__version__ = '0.0.6'

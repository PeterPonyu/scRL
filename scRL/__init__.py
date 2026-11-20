from .GridCore import grids_from_embedding, align_pseudotime, project_cluster, project_back, project
from .EnvironmentCore import d_rewards, c_rewards
from .Trainer import trainer
from .Simulator.Core import get_sim_df
from .Simulator.Results import sim_results, sim_results2, sim_results3
from .Trajectory.Core import get_traj_df
from .Trajectory.Results import traj_results
from .utils import get_state_value
from .ExactCore import exact_values, exact_trajectories
from .FateCore import fate_gate, fate_probabilities, fate_over_runs
from .EnvironmentNDCore import reachability
from .CorridorCore import fate_corridors, corridor_profile, plot_corridors
from .ConditionCore import (project_conditions, compare_abundance, condition_grid, compare_values,
                            compare_fates, shares_by_sample)
from .PlotCore import (plot_grids, plot_slices, plot_views, plot_trend, plot_clusters, plot_policy,
                       plot_training, plot_projections, plot_report, save_rotation, plot_style, plot_fates,
                       plot_conditions, plot_mesh, plot_trajectories, plot_voxels)

# Backward compatibility aliases for deprecated function names
lineage_rewards = d_rewards  # Renamed: lineage_rewards -> d_rewards
gene_rewards = c_rewards     # Renamed: gene_rewards -> c_rewards  
project_expression = project  # Renamed: project_expression -> project

__all__ = ['grids_from_embedding', 'align_pseudotime', 'project_cluster', 'project_back', 'project'
           , 'd_rewards', 'c_rewards', 'trainer'
           , 'get_sim_df', 'sim_results', 'sim_results2', 'sim_results3'
           , 'get_traj_df', 'traj_results'
           , 'get_state_value', 'plot_grids', 'plot_slices', 'plot_views', 'plot_trend'
           , 'plot_clusters', 'plot_policy', 'plot_training'
           , 'plot_projections', 'plot_report', 'save_rotation', 'plot_style'
           , 'exact_values', 'exact_trajectories', 'fate_gate', 'fate_probabilities', 'fate_over_runs', 'reachability'
           , 'project_conditions', 'compare_abundance', 'condition_grid', 'compare_values', 'compare_fates', 'shares_by_sample'
           , 'plot_fates', 'plot_conditions', 'plot_mesh', 'plot_trajectories', 'plot_voxels'
           , 'fate_corridors', 'corridor_profile', 'plot_corridors'
           # Backward compatibility
           , 'lineage_rewards', 'gene_rewards', 'project_expression']

__version__ = '0.0.9'

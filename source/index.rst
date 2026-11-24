scRL: deep reinforcement learning in single cell data analysis for fate decision evaluation
============================================================================================

scRL places a regular grid over an embedding of single cells and treats differentiation as a sequence of moves on that grid. With lineage- or gene-based rewards, an actor–critic agent learns a Decision or a Contribution value for every grid point, and its moves trace routes between cell states. Version 2.0 adds grids in three dimensions, an exact solution of the same decision process that needs no training, fate probabilities from a walk on the grid, reading new cells onto a stored grid, and tools for comparing conditions, including a protocol for testing a per-cell readout with the sample as the unit. The original study (Fu et al., 2025; see Citation) describes the framework and its applications.

.. image:: _static/Pattern.png
    :width: 600
    :align: center

Model Architecture and Reinforcement Learning Environment
------------------------------------------------------------

The scRL model architecture is grounded in the Actor-Critic reinforcement learning framework, which is specifically tailored for the unique environment created by single-cell data. This environment is meticulously constructed through a comprehensive data preprocessing pipeline, providing a structured playground where the reinforcement learning agent can explore and learn effectively.

At the heart of scRL's model is its capability to transform high-dimensional single-cell data into a grid embedding. This embedding, generated using a boundary scanning algorithm, serves as a critical interface between the raw data and the reinforcement learning agent. It not only preserves the essential topological features of the original data but also facilitates efficient exploration and learning by the agent.

Within this environment, the reinforcement learning agent, guided by the Actor-Critic architecture, interacts with the grid embedding to gather information and make informed decisions. The agent receives rewards based on predefined criteria, such as gene expression patterns or lineage commitments, which are designed to encourage the learning of optimal policies for understanding cellular fate determination and differentiation.

Workflow Overview
------------------

1. **Data Preprocessing and Grid Embedding Generation**:

   - scRL begins by preprocessing the single-cell data, employing techniques like PCA, UMAP, t-SNE for dimensionality reduction, and clustering algorithms like Leiden or Louvain for subpopulation identification.
   - A specialized grid embedding is then created using a boundary scanning algorithm, mapping the single-cell data into a manageable 2D space while preserving crucial topological relationships.

2. **Projection and Pseudotime Calculation**:

   - Both subpopulation information and gene expression data are projected onto this grid embedding, providing a rich context for the reinforcement learning agent.
   - A starting subpopulation, often representing stem or progenitor cells, is selected, and the Dijkstra shortest path algorithm is employed to compute the pseudotime—a measure of cellular progression—from every grid point to this starting point.

3. **Environmental Reward Generation**:

   - Leveraging pseudotime information, scRL crafts environmental rewards that guide the reinforcement learning agent.
   - Lineage-specific rewards are designed based on target lineages' grid regions, while gene-specific rewards are formulated using projected gene expression data.
   - Two distinct reward patterns—contributory (increasing with pseudotime) and fate-determining (decreasing with pseudotime)—offer flexibility in exploring different cellular behaviors.

4. **Reinforcement Learning Training and Interaction**:

   - With a fully established environment and reward system, the Actor-Critic reinforcement learning architecture engages in interactive learning.
   - The agent explores the grid-embedded environment, making decisions based on policies learned through interactions and feedback from the environment.
   - The goal is to maximize cumulative rewards, indicating a deeper understanding of cellular fate determination and differentiation processes.

Functional Modules
-------------------

scRL's functionality extends beyond basic reinforcement learning, offering three specialized modules:

1. **Gene Functional Module**:

   - Explores gene expression patterns and potentials, providing insights into genes' roles in cellular differentiation and fate determination.
   - Outputs gene-specific valuations based on the chosen reward mode, shedding light on their contributory or fate-determining effects.

2. **Lineage Functional Module**:

   - Delves into lineage commitments and potentials, revealing the trajectories and decision-making processes involved in cellular differentiation.
   - Offers lineage-specific valuations, highlighting key lineages and their associated fate determination strengths.

3. **Trajectory Functional Module**:

   - Focuses on differentiation trajectories, sampling and visualizing the paths cells take during their developmental journey.
   - Utilizes an advanced autoencoder, trained in parallel with the Actor-Critic architecture, to simulate and predict future differentiation steps.
   - Provides a parameterized probability distribution for upcoming changes in cellular states, gene expressions, and trajectory coordinates, offering unparalleled insights into the dynamics of cellular differentiation.

Installation
------------

.. code-block:: bash

    pip install scrl-fatedecision

Citation
--------

Fu, Z., Chen, C., Wang, S., Wang, J., & Chen, S. (2025).
scRL: Utilizing Reinforcement Learning to Evaluate Fate Decisions in Single-Cell Data.
*Biology*, 14(6), 679.
`doi:10.3390/biology14060679 <https://doi.org/10.3390/biology14060679>`_

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: TUTORIALS

   notebook/Basics_on_scRL
   notebook/Contribution_vs_Decision_Example
   notebook/Application1
   notebook/Application2
   grid3d
   exact
   conditions
   readout_protocol
   testdata

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API
   
   GridCore
   environment
   trainer
   simulator
   trajectory


# **scRL: deep reinforcement learning in single cell data analysis for fate decision evaluation**

Single-cell sequencing effectively delineates the heterogeneous states within cell populations, and pseudotime analysis reconstructs these states' sequential transitions. However, pseudotime often fails to adequately represent intermediate states where critical cell fate decisions are made. To overcome this limitation, we introduce single-cell Reinforcement Learning (scRL), which integrates single-cell grid embedding—an extension of UMAP—with a robust reinforcement learning framework using an actor-critic architecture. This novel approach dynamically simulates differentiation pathways, enhancing our understanding of cell fate decision timing and progression. Using lineage or genetic information-based rewards, the actor network samples potential differentiation trajectories, while the critic network assesses the decision strength. Our results reveal that the strength of fate decisions typically peaks before lineage potential and gene decision strength precedes gene expression, identifying key transitional phases and pre-expression states. Validated through an irradiation-perturbed atlas and a gene perturbation atlas of hematopoietic lineages, scRL offers a nuanced understanding of cellular differentiation, improving our ability to decode and predict cell fate decisions and developmental outcomes.


<img src="docs/_static/Pattern.png" width="600" align="middle"/>


## **Model Architecture and Reinforcement Learning Environment**

The scRL model architecture is grounded in the Actor-Critic reinforcement learning framework, which is specifically tailored for the unique environment created by single-cell data. This environment is meticulously constructed through a comprehensive data preprocessing pipeline, providing a structured playground where the reinforcement learning agent can explore and learn effectively.

At the heart of scRL's model is its capability to transform high-dimensional single-cell data into a grid embedding. This embedding, generated using a boundary scanning algorithm, serves as a critical interface between the raw data and the reinforcement learning agent. It not only preserves the essential topological features of the original data but also facilitates efficient exploration and learning by the agent.

Within this environment, the reinforcement learning agent, guided by the Actor-Critic architecture, interacts with the grid embedding to gather information and make informed decisions. The agent receives rewards based on predefined criteria, such as gene expression patterns or lineage commitments, which are designed to encourage the learning of optimal policies for understanding cellular fate determination and differentiation.

## **Workflow Overview**

### 1. **Data Preprocessing and Grid Embedding Generation**:

-   **Data Preprocessing**: scRL begins by preprocessing the single-cell data using dimensionality reduction techniques (PCA, UMAP, t-SNE) and clustering algorithms (Leiden, Louvain) to identify subpopulations within the cells.
-   **Grid Embedding Creation**: A specialized grid embedding is then generated using a boundary scanning algorithm. This maps the single-cell data into a manageable 2D space while preserving crucial topological relationships.

### 2. **Projection and Pseudotime Calculation**:

-   **Projection**: Both subpopulation information and gene expression data are projected onto this grid embedding, providing a comprehensive context for the reinforcement learning agent.
-   **Pseudotime**: A starting subpopulation, often representing stem or progenitor cells, is selected. The Dijkstra shortest path algorithm is then used to compute pseudotime—a measure of cellular progression—from every grid point to this starting point.

### 3. **Environmental Reward Generation**:

-   **Reward Crafting**: Using pseudotime information, scRL designs environmental rewards that guide the reinforcement learning agent. Lineage-specific rewards are tailored based on target lineages' grid regions, while gene-specific rewards utilize projected gene expression data.
-   **Reward Patterns**: Two distinct reward patterns are employed—contributory (increasing with pseudotime) and fate-determining (decreasing with pseudotime)—to provide flexibility in exploring different cellular behaviors.

### 4. **Reinforcement Learning Training and Interaction**:

-   **Training**: With a fully established environment and reward system, the Actor-Critic reinforcement learning architecture engages in interactive learning.
-   **Interaction**: The agent explores the grid-embedded environment, making decisions based on policies learned through interactions and feedback from the environment.
-   **Goal**: The primary goal is to maximize cumulative rewards, which indicates a deeper understanding of cellular fate determination and differentiation processes.

## **Functional Modules**

scRL's functionality extends beyond basic reinforcement learning, offering three specialized modules:

### 1. **Gene Functional Module**:

-   **Investigation**: Investigates gene expression patterns and potentials, providing insights into genes' roles in cellular differentiation and fate determination.
-   **Valuation**: Outputs gene-specific valuations based on the chosen reward mode, shedding light on their contributory or fate-determining effects.

### 2. **Lineage Functional Module**:

-   **Investigation**: Delves into lineage commitments and potentials, revealing the trajectories and decision-making processes involved in cellular differentiation.
-   **Valuation**: Offers lineage-specific valuations, highlighting key lineages and their associated fate determination strengths.

### 3. **Trajectory Functional Module**:

-   **Focus**: Concentrates on differentiation trajectories, sampling, and visualizing the paths cells take during their developmental journey.
-   **Simulation**: Utilizes an advanced autoencoder, trained in parallel with the Actor-Critic architecture, to simulate and predict future differentiation steps.
-   **Insights**: Provides a parameterized probability distribution for upcoming changes in cellular states, gene expressions, and trajectory coordinates, offering unparalleled insights into the dynamics of cellular differentiation.


## **Documentation**

[![Documentation Status](https://readthedocs.org/projects/scrl/badge/?version=latest)](https://scrl.readthedocs.io/en/latest/?badge=latest)

[documentation](https://scrl.readthedocs.io/en/latest/)

## **Installation**

[![PyPI](https://img.shields.io/pypi/v/singlecellrl.svg?color=brightgreen&style=flat)](https://pypi.org/project/singlecellrl/)

``` bash
pip install singlecellRL
```

## **Reference**

[Zeyu Fu et al. Reinforcement learning guides single-cell sequencing in decoding lineage and cell fate decisions](https://doi.org/10.1101/2024.07.04.602019)

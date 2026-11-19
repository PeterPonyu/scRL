Grids in three dimensions
=========================

scRL lays a regular grid over an embedding of the cells, lets an agent move over the grid points and reads
the critic's value at each grid point as the strength of a fate decision. The grid can be laid over a 2-D
embedding, as in the tutorials, or over a 3-D embedding, for example a three-component UMAP:

.. code-block:: python

    sc.tl.umap(adata, n_components=3)
    X3 = adata.obsm['X_umap']

    gres = scRL.grids_from_embedding(X3)
    scRL.project_cluster(gres, clusters)
    scRL.align_pseudotime(gres, early_cluster='4')
    scRL.d_rewards(gres, starts=['4'], ends=['0', '3', '5'], mode='Decision')
    trainer = scRL.trainer('ActorCritic', gres, reward_type='d', reward_mode='Decision',
                           starts_prob=False, X_latent=X_pca, num_episodes=5000)
    returns, values = trainer.train()
    scRL.get_state_value(gres, trainer, key='decision_value')
    scRL.project_back(gres, 'decision_value')
    scRL.plot_grids(gres, key='decision_value')

Every step after ``grids_from_embedding`` is called exactly as in 2-D.

How the grid is built
---------------------

The grid has the same spacing along every axis, so one grid step is the same distance in every direction.
It is built the same way for any 3-D embedding: a grid point is kept when cells lie close to it, and one
further layer of grid points around the kept points is added. That outer layer is the boundary of the grid;
the points inside it, where the cells are, are the points the agent travels through. Each grid point takes
the cluster label of its nearest cell, and pseudotime is the number of grid steps from the start cluster,
as in 2-D.

Moves
-----

In 3-D the agent moves one grid step along one axis at a time: left or right, down or up, back or front,
six moves in all. The 2-D grid keeps its eight moves, diagonals included. The stop rules are the same as in
2-D: a move off the grid, a step onto the boundary outside the start cluster, a return to a grid point
already visited, or the step limit ends the episode.

Parameters
----------

``n``
    The number of grid points across the widest axis of the data. The default is 30 in 3-D (50 in 2-D).
    A larger ``n`` gives a finer grid with more grid points to learn.
``radius``
    How far from the cells the grid extends, in grid steps. The default is 1 in 3-D. A larger radius gives
    a thicker grid around the cells.
``moves``
    ``'axis'`` (the six moves above, the default in 3-D) or ``'full'`` (diagonal steps as well, 26 moves).
``method``
    ``'scan'`` is the outline scan used for 2-D embeddings; ``'occupancy'`` is the grid described here and
    is used for 3-D embeddings. ``'auto'`` (the default) picks the scan for 2-D input and the occupancy grid
    otherwise; ``method='occupancy'`` can also be used on a 2-D embedding.

Viewing the grid
----------------

``scRL.plot_grids(gres)`` shows the grid structure, with the travel points dark and the boundary light.
``scRL.plot_grids(gres, key='pseudotime')`` colours the grid points by a stored value, and
``cells=True`` colours the cells instead (for values brought back to the cells with ``project_back``).
``view=(elevation, azimuth)`` turns the 3-D view, and ``interactive=True`` returns a figure that can be
rotated with the mouse (this needs plotly).

Checking the results
--------------------

As in 2-D, the outcome of training varies between random seeds. Look at the return curve returned by
``trainer.train()`` and repeat the training with a few seeds before reading the decision values.

Four and five dimensions
------------------------

4-D and 5-D embeddings are accepted with the same calls (8 and 10 moves along the axes). This is
experimental and not yet meant for results.

A complete script on the paul15 data is in ``examples/grid3d_paul15.py`` in the repository.

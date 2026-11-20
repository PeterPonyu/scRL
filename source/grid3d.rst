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
    scRL.align_pseudotime(gres, early_cluster='4', seed=0)
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
as in 2-D. The start points are sampled from the start cluster; ``align_pseudotime(gres, early_cluster,
seed=0)`` makes the draw, and so pseudotime, reproducible without touching numpy's global generator.
Once pseudotime is fixed, the exact solution, the lineage gate and the lineage shares are deterministic.

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

Viewing results
---------------

``scRL.plot_grids(gres, key=...)`` draws the grid points (or, with ``cells=True``, the cells) coloured by a
stored value in one 3-D view; ``key='clusters'`` colours by cluster, and ``interactive=True`` gives a figure
that can be rotated with the mouse (this needs plotly).

``scRL.plot_views`` shows the same 3-D view from several angles side by side, with one colour scale.

``scRL.plot_mesh`` draws the grid as a mesh: a line for every move between two neighbouring kept grid points,
so the drawing shows the space an episode or a walk can travel; with ``key=...`` the grid points (and, with
``color_edges=True``, the lines) are coloured by a stored value.

``scRL.plot_voxels`` draws a 3-D grid as cubes, one per kept grid point, coloured by a stored value;
``cut=(1, .5)`` removes the grid points beyond half of Dim 2 and opens a section through the interior.

``scRL.plot_trajectories`` draws the trajectories of one or several lineages over the grid (or the cells):
each walk stored by ``exact_trajectories`` or ``get_traj_df`` as a thin line and their mean path as a thick
line ending in an arrow, one colour per lineage. The mean path is a route only when the walks of one key
follow one route; walks that part towards different end clusters are better drawn with ``mean=False`` or
under separate keys. With Contribution rewards the walks follow each lineage to its far end, which shows
where lineages part:

.. code-block:: python

    for name, ends in {'Ery': ['1Ery', '2Ery'], 'Mo': ['14Mo', '15Mo']}.items():
        scRL.d_rewards(gres, starts=['7MEP'], ends=ends, mode='Contribution')
        scRL.exact_values(gres, reward_mode='Contribution', key_add=f'{name}_value')
        scRL.exact_trajectories(gres, key=f'{name}_value', key_add=name)
    scRL.plot_trajectories(gres, ['Ery', 'Mo'], context='cells')

``scRL.plot_slices`` cuts the grid into slabs along one axis and draws each slab as a flat section, so that the
inside of the grid can be read; with ``policy=trainer`` it also marks the direction the agent tends to move.

``scRL.plot_projections`` collapses the grid along each axis onto the plane of the other two and shows the
largest (or mean) value along each line of sight.

``scRL.plot_clusters`` summarises a value per cluster as box plots, highest median first.

``scRL.plot_trend`` follows a value along pseudotime as the median and inter-quartile range in bins;
``clusters=[...]`` restricts it to one lineage.

``scRL.plot_policy`` draws arrows for the direction the agent tends to move from the grid points.

``scRL.plot_training`` shows the return over the training episodes, and the value estimate when given.

``scRL.plot_report`` puts the 3-D view, the projections, six sections, the per-cluster summary and the trend
of one value on a single page.

``scRL.save_rotation`` writes a turning view of the grid as a GIF (or an MP4 when ffmpeg is installed).

``with scRL.plot_style():`` applies publication settings (font, font size, embedded fonts, line widths,
300 dpi) to the figures drawn inside the block only.

Checking the results
--------------------

As in 2-D, the outcome of training varies between random seeds. Look at the return curve returned by
``trainer.train()`` and repeat the training with a few seeds before reading the decision values.

Four and five dimensions
------------------------

4-D and 5-D embeddings are accepted with the same calls (8 and 10 moves along the axes). This is
experimental and not yet meant for results.

A complete script on the paul15 data is in ``examples/grid3d_paul15.py`` in the repository.

``examples/grid3d_walkthrough.py`` goes through every mode on a 3-D grid of the same data: lineage rewards in
both modes, gene rewards, the exact solution next to a soft actor-critic, a fate map over three lineages,
trajectories and the simulator, two conditions compared on one grid, and a rotating and an interactive view.
It runs from the repository root in a few minutes and writes its figures to ``grid3d_walkthrough_output/``.

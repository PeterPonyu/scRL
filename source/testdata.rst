Test data and expected results
==============================

The worked example ``examples/grid3d_walkthrough.py`` doubles as a test data set with expected results.

Data
----

The test data are the mouse myeloid progenitor cells of Paul and colleagues (2,730 cells, 19 clusters from
the megakaryocyte-erythrocyte progenitors to the erythroid, monocyte and neutrophil lineages), the small
public data set that scanpy loads with ``scanpy.datasets.paul15``; a copy ships with the tutorials in
``source/notebook/data/paul15``. The walkthrough normalises the counts, takes 50 principal components and a
3-D UMAP, lays a regular 3-D grid over the UMAP and goes through every mode of scRL on it.

UMAP coordinates depend on the numerical libraries and the processor, so they are not guaranteed to be
the same on another machine. The 3-D coordinates of the run behind the expected results therefore ship
with the repository (in ``examples/expected/``; they are not part of the wheel installed by pip), together with the cluster labels and the six log-normalised marker genes the walkthrough
uses (``examples/expected/grid3d_walkthrough_inputs.npz``, about 60 kB). The check starts from these
coordinates, so it does not depend on UMAP.

Command
-------

From the repository root:

.. code-block:: bash

    python examples/grid3d_walkthrough.py --check

It needs neither scanpy nor a GPU, prints ``check passed`` with the number of values compared, or lists
every value that differs, and exits with status 1 when any value differs. On a laptop with a 24-core
processor it takes about 4 seconds; the full walkthrough (``python examples/grid3d_walkthrough.py``, which
also trains a soft actor-critic and the simulator and writes all figures) takes about nine minutes on the same
laptop with its GPU. The same comparison runs in the test suite (``tests/test_walkthrough_expected.py``).

What is compared
----------------

The check reruns the deterministic part of the walkthrough with fixed seeds: the grid, the exact solutions,
the lineage gate and the lineage shares over three lineages, and the comparison of two conditions made from
the public data (cells go to 'treated' with probability 0.5, neutrophil-lineage cells with probability 0.15).
It compares with ``examples/expected/grid3d_walkthrough_results.json``:

- counts, which must match exactly: cells, kept grid points, travel and boundary points, rewarded and
  reachable rewarded grid points, grid points per heaviest lineage (from the lineage shares and from the
  gate), cells per condition and grid points per abundance call;
- the exact erythroid Decision value (minimum, maximum, mean and its value at ten fixed grid points) and the
  means of the Contribution and gene-reward values, within 1e-8;
- the column means of the lineage shares, within 1e-8;
- the shares, differences and permutation p-values of the condition comparison of heaviest lineages,
  within 1e-12.

The exact values and the lineage shares are double-precision linear algebra on the shipped coordinates, so
they agree to rounding on any machine; the tolerances leave several orders of magnitude for differences in
summation order. The soft actor-critic, the simulator and the figures are not compared: they depend on the
GPU and on the random number streams of the deep-learning library.

``python examples/grid3d_walkthrough.py --write-expected`` recomputes the UMAP from the data and writes the
shipped inputs and expected results anew.

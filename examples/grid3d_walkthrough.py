"""Every mode of scRL on a 3-D grid, on the paul15 haematopoiesis data.

Needs scanpy (plotly for the interactive figure). Run from the repository root:

    python examples/grid3d_walkthrough.py

It takes about nine minutes on a laptop GPU and writes its figures to ./grid3d_walkthrough_output/. The short example
examples/grid3d_paul15.py is the quick start; this script goes through lineage rewards in both modes, gene
rewards, the exact solution next to a soft actor-critic, a fate map over three lineages, trajectories and the
simulator, two conditions compared on one grid, and a rotating and an interactive view.

    python examples/grid3d_walkthrough.py --check

reruns the deterministic part of the walkthrough (grid, exact values, fate map, condition comparison) on the
3-D coordinates shipped in examples/expected/ and compares the results with examples/expected/; it needs
neither scanpy nor a GPU and takes well under a minute. ``--write-expected`` recomputes the 3-D UMAP from the
data, writes the shipped inputs and the expected results anew.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # use the scRL package of this repository
import scRL

EXPECTED = HERE / 'expected'
INPUTS = EXPECTED / 'grid3d_walkthrough_inputs.npz'
RESULTS = EXPECTED / 'grid3d_walkthrough_results.json'

ERY = ['7MEP', '6Ery', '5Ery', '4Ery', '3Ery', '2Ery', '1Ery']
NEU = ['7MEP', '9GMP', '10GMP', '16Neu', '17Neu']
MARKERS = ['Klf1', 'Gata1', 'Irf8', 'Csf1r', 'Elane', 'Mpo']
LINEAGES = {'erythroid': ['1Ery', '2Ery'], 'monocyte': ['14Mo', '15Mo'], 'neutrophil': ['16Neu', '17Neu']}

# Tolerances of --check. Counts must match exactly. The exact solution and the walk shares are float64
# linear algebra on the shipped coordinates, so they agree to rounding; the shares of the condition
# comparison are ratios of counts with a seeded permutation test.
TOLERANCE = {'exact': 1e-8, 'shares': 1e-8, 'comparison': 1e-12}


def seed(s=0):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except ImportError:
        pass


def prepare():
    """3-D UMAP, PCA, clusters and log-normalised marker values of paul15 (needs scanpy)."""
    import scanpy as sc
    data_dir = HERE.parent / 'source' / 'notebook' / 'data'
    if data_dir.is_dir():
        sc.settings.datasetdir = data_dir      # reuse the copy of paul15 shipped with the tutorials
    adata = sc.datasets.paul15()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    markers = pd.DataFrame(adata[:, MARKERS].X, columns=MARKERS)    # log-normalised, not scaled
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50, random_state=0)
    sc.pp.neighbors(adata, random_state=0)
    sc.tl.umap(adata, n_components=3, random_state=0)
    clusters = adata.obs['paul15_clusters'].reset_index(drop=True)
    return adata.obsm['X_umap'], adata.obsm['X_pca'], clusters, markers


def save_inputs(X3, clusters, markers):
    EXPECTED.mkdir(exist_ok=True)
    np.savez_compressed(INPUTS, umap3=np.asarray(X3, dtype=np.float32), codes=clusters.cat.codes.to_numpy(np.int8),
                        categories=np.asarray(clusters.cat.categories, dtype=str),
                        markers=markers.to_numpy(np.float32), marker_names=np.array(MARKERS))


def load_inputs():
    d = np.load(INPUTS)
    clusters = pd.Series(pd.Categorical.from_codes(d['codes'], categories=list(d['categories'])))
    return d['umap3'], clusters, pd.DataFrame(d['markers'], columns=list(d['marker_names']))


def build(X3, clusters, markers):
    """The deterministic part of the walkthrough: grid, exact values, fate map and the condition comparison."""
    seed(0)
    gres = scRL.grids_from_embedding(X3)
    scRL.project_cluster(gres, clusters)
    scRL.align_pseudotime(gres, early_cluster='7MEP', seed=0)
    scRL.project_back(gres, 'pseudotime')
    scRL.project(gres, markers)
    info = {}
    scRL.d_rewards(gres, starts=['7MEP'], ends=['1Ery', '2Ery'])
    info['reach'] = scRL.reachability(gres, 'd', 'Decision')
    scRL.exact_values(gres, key_add='ery_exact')
    scRL.d_rewards(gres, starts=['7MEP'], ends=['1Ery', '2Ery'], mode='Contribution')
    scRL.exact_values(gres, reward_mode='Contribution', key_add='ery_contribution')
    scRL.c_rewards(gres, reward_keys=['Klf1', 'Gata1'], starts=['7MEP'])
    scRL.exact_values(gres, reward_type='c', key_add='ery_genes')
    for name, ends in LINEAGES.items():
        scRL.d_rewards(gres, starts=['7MEP'], ends=ends)
        scRL.exact_values(gres, key_add=f'{name}_exact')
    info['walks'] = scRL.exact_trajectories(gres, key='neutrophil_exact')    # the current reward is neutrophil
    scRL.fate_gate(gres, [f'{n}_exact' for n in LINEAGES], names=list(LINEAGES))
    scRL.fate_probabilities(gres, LINEAGES)
    # Two conditions made from the public data: cells go to 'treated' with probability 0.5, neutrophil-lineage
    # cells with probability 0.15
    rng = np.random.default_rng(0)
    conditions = np.where(rng.random(len(X3)) < .5, 'treated', 'control')
    neutrophil = clusters.isin(['16Neu', '17Neu']).to_numpy()
    conditions[neutrophil] = np.where(rng.random(neutrophil.sum()) < .15, 'treated', 'control')
    scRL.project_conditions(gres, conditions)
    scRL.compare_abundance(gres, test='treated', reference='control', n_shuffles=100)
    info['conditions'] = conditions
    info['fates'] = scRL.compare_fates(gres, 'treated', 'control', key='fate_top')
    return gres, info


def summarise(gres, info):
    """Compact results of ``build``: counts, the exact solution and the fate map."""
    exact = np.asarray(gres.grids['ery_exact'], dtype=float)
    mapped = np.asarray(gres.grids['mapped_grids'])
    at = np.linspace(0, len(mapped) - 1, 10).astype(int)
    counts = lambda v: {str(k): int(n) for k, n in sorted(pd.Series(np.asarray(v)).value_counts().items())}
    fates = info['fates']
    return {
        'counts': {
            'cells': int(len(gres.embedding['embedding'])),
            'kept_grid_points': int(len(mapped)),
            'travel_points': int(len(gres.grids['mapped_travel'])),
            'boundary_points': int(len(gres.grids['mapped_boundary'])),
            'rewarded_points': int(info['reach']['rewarded']),
            'reachable_rewarded_points': int(info['reach']['reached']),
            'grid_points_per_heaviest_lineage_walk': counts(gres.grids['fate_top']),
            'grid_points_per_heaviest_lineage_gate': counts(gres.grids['gate_top']),
            'cells_per_condition': counts(info['conditions']),
            'grid_points_per_abundance_call': counts(gres.grids['abundance_call']),
            'neutrophil_walks_from_7MEP': int(len(info['walks'])),
            'neutrophil_walks_entering': int(info['walks']['entered'].sum()),
            'neutrophil_first_walk_moves': int(info['walks']['moves'].iloc[0]),
        },
        'exact': {
            'decision_min': float(np.nanmin(exact)), 'decision_max': float(np.nanmax(exact)),
            'decision_mean': float(np.nanmean(exact)),
            'decision_at_grid_points': {str(int(mapped[i])): float(exact[i]) for i in at},
            'contribution_mean': float(np.nanmean(np.asarray(gres.grids['ery_contribution'], dtype=float))),
            'gene_reward_mean': float(np.nanmean(np.asarray(gres.grids['ery_genes'], dtype=float))),
            'neutrophil_walk_bound_share_median': float(info['walks']['bound_share'].median()),
        },
        'shares': {f'mean_{c}': float(np.nanmean(gres.grids['fate'][c].to_numpy(float)))
                   for c in gres.grids['fate'].columns},
        'comparison': {f'{col}_{cat}': float(fates.loc[cat, col]) for cat in fates.index
                       for col in ('reference', 'test', 'difference', 'pvalue')},
    }


def compare(found, expected):
    """List of mismatches between two summaries, with the tolerances of TOLERANCE."""
    bad = []
    for group, values in expected.items():
        for name, want in values.items():
            got = found.get(group, {}).get(name)
            if group == 'counts':
                if got != want:
                    bad.append(f'{group}.{name}: expected {want}, found {got}')
                continue
            wants = want if isinstance(want, dict) else {None: want}
            gots = got if isinstance(got, dict) else {None: got}
            for key, w in wants.items():
                g = gots.get(key)
                if g is None or not abs(g - w) <= TOLERANCE[group]:
                    label = f'{group}.{name}' + (f'[{key}]' if key is not None else '')
                    bad.append(f'{label}: expected {w!r}, found {g!r} (tolerance {TOLERANCE[group]:g})')
    return bad


def _n_values(summary):
    return sum(len(v) if isinstance(v, dict) else 1 for g in summary.values() for v in g.values())


def check():
    """Rerun ``build`` on the shipped coordinates and compare with the expected results; True when they agree."""
    X3, clusters, markers = load_inputs()
    gres, info = build(X3, clusters, markers)
    found = summarise(gres, info)
    expected = json.loads(RESULTS.read_text())
    bad = compare(found, expected)
    n = _n_values(expected)
    if bad:
        print(f'check failed: {len(bad)} mismatch(es)')
        print('\n'.join(bad))
    else:
        print(f'check passed: {n} expected values agree (counts exact; exact solution within {TOLERANCE["exact"]:g}, '
              f'walk shares within {TOLERANCE["shares"]:g}, condition comparison within '
              f'{TOLERANCE["comparison"]:g})')
    return not bad


def write_expected():
    X3, _, clusters, markers = prepare()
    save_inputs(X3, clusters, markers)
    X3, clusters, markers = load_inputs()
    gres, info = build(X3, clusters, markers)
    RESULTS.write_text(json.dumps(summarise(gres, info), indent=1) + '\n')
    print(f'wrote {INPUTS} and {RESULTS}')


def walkthrough():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    out = Path('grid3d_walkthrough_output')
    out.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(out / f'{name}.png')
        plt.close(fig)

    seed(0)
    with scRL.plot_style():
        # 1. Data and grid: a 3-D UMAP of the cells and a regular grid laid over it; then the deterministic
        #    part of the walkthrough (exact values, fate map, two conditions), as checked by --check
        X3, X_pca, clusters, markers = prepare()
        gres, info = build(X3, clusters, markers)
        save(scRL.plot_grids(gres, key='clusters').figure, '01_grid_clusters')
        save(scRL.plot_slices(gres, key='clusters'), '01_slices_clusters')
        print(f'1. grid with {len(gres.grids["mapped_grids"])} grid points over {len(X3)} cells')

        # 2. Lineage mode, Decision: exact solution and a soft actor-critic for the erythroid lineage
        reach = info['reach']
        scRL.d_rewards(gres, starts=['7MEP'], ends=['1Ery', '2Ery'])
        seed(0)
        trainer = scRL.trainer('SAC', gres, reward_type='d', reward_mode='Decision', starts_prob=False,
                               X_latent=X_pca, num_episodes=2000, capacity=2000)
        returns, values = trainer.train()
        scRL.get_state_value(gres, trainer, 'ery_decision')
        rho = spearmanr(gres.grids['ery_decision'], gres.grids['ery_exact']).correlation
        save(scRL.plot_training(returns, values), '02_training')
        save(scRL.plot_report(gres, 'ery_exact', clusters=ERY, title='Erythroid lineage, exact value'),
             '02_report_exact')
        save(scRL.plot_slices(gres, key='ery_exact', policy=gres.grids['ery_exact_best']), '02_slices_policy')
        print(f'2. {reach["reached"]} of {reach["rewarded"]} rewarded grid points reachable; '
              f'learned against exact values: Spearman {rho:.2f}')

        # 3. Lineage mode, Contribution: the same lineage valued toward its far end
        save(scRL.plot_trend(gres, ['ery_exact', 'ery_contribution'], labels=['Decision', 'Contribution'],
                             clusters=ERY, cells=False).figure, '03_trend_modes')
        print('3. erythroid values in Decision and Contribution mode along pseudotime')

        # 4. Gene mode: the reward of the erythroid genes Klf1 and Gata1
        rho = spearmanr(gres.grids['ery_genes'], gres.grids['ery_exact']).correlation
        print(f'4. gene reward against lineage reward: Spearman {rho:.2f}')

        # 5. Fate map over three lineages: a gate over exact values, and lineage shares from a walk
        save(scRL.plot_fates(gres), '05_fates_gate')
        save(scRL.plot_fates(gres, key_add='fate'), '05_fates_shares')
        share = pd.Series(gres.grids['fate_top']).value_counts(normalize=True).round(2).to_dict()
        mep = np.asarray(gres.grids['mapped_grids_clusters']) == '7MEP'
        from_mep = {n: round(float(gres.grids[f'fate_{n}'][mep].mean()), 2) for n in LINEAGES}
        print(f'5. heaviest lineage of the grid points from the walk: {share}; from 7MEP: {from_mep}, '
              f'arrived {gres.grids["fate_reached"][mep].mean():.2f}')

        # 6. Trajectories and simulator with the soft actor-critic of section 2
        scRL.d_rewards(gres, starts=['7MEP'], ends=['1Ery', '2Ery'])
        scRL.get_traj_df(gres, trainer, n=100, key='ery')
        scRL.traj_results(gres, ERY, 'ery')
        save(plt.gcf(), '06_trajectory')
        gres.grids['gene_exp'] = gres.grids['proj']
        trainer.train_simulator(500, 16)
        scRL.get_sim_df(gres, trainer, lineages=['1Ery', '2Ery', '14Mo', '15Mo'], steps=30, prefix='ery')
        here = Path.cwd()
        import os
        os.chdir(out)
        try:
            scRL.sim_results(gres, n_sample=1000, prefix='ery')
        finally:
            os.chdir(here)
        plt.close('all')
        print(f'6. trajectory of {len(gres.trajectory["ery_traj"])} averaged steps; simulator figures in '
              f'{out / "sim_output"}')
        # Trajectories from the exact solution, no training: the neutrophil lineage, each walk up to its entry
        scRL.d_rewards(gres, starts=['7MEP'], ends=['16Neu', '17Neu'])
        walks = scRL.exact_trajectories(gres, key='neutrophil_exact', until_entry=True, key_add='neu_walks')
        scRL.traj_results(gres, NEU, 'neu_walks')
        save(plt.gcf(), '06_trajectory_exact')
        print(f'   exact trajectories: {walks["entered"].mean():.2f} of {len(walks)} walks from 7MEP enter the '
              f'neutrophil lineage, after a median {walks["entry_move"].median():.0f} moves')

        # 7. Two conditions on one grid (made in section 1 from the public data: cells go to 'treated' with
        #    probability 0.5, neutrophil-lineage cells with probability 0.15)
        print('7. per cluster:')
        print(gres.grids['abundance_clusters'].round(2).to_string())
        save(scRL.plot_conditions(gres), '07_conditions')
        save(scRL.plot_projections(gres, key='abundance_call'), '07_calls')
        values = scRL.compare_values(gres, 'treated', 'control', ends=['16Neu', '17Neu'], starts=['7MEP'],
                                     mode='Contribution', n_shuffles=50)
        print(values.round(3).to_string())
        print(info['fates'].round(3).to_string())

        # 8. Sharing: a rotating view and an interactive page
        scRL.save_rotation(gres, out / 'rotation.gif', key='ery_exact', frames=36)
        try:
            page = scRL.plot_grids(gres, key=['ery_exact', 'fate_top', 'clusters'], interactive=True)
            page.write_html(out / 'interactive.html')
            print(f'8. rotation and interactive view written to {out}')
        except ImportError:
            print(f'8. rotation written to {out}; install plotly for the interactive view')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true', help='compare with the expected results in examples/expected')
    parser.add_argument('--write-expected', action='store_true', help='write the shipped inputs and expected results')
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    if args.write_expected:
        write_expected()
    else:
        walkthrough()

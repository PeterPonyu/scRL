"""The worked example reproduces its expected results on the shipped 3-D coordinates of paul15."""
import importlib.util
import json
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'grid3d_walkthrough.py'


def _walkthrough():
    spec = importlib.util.spec_from_file_location('grid3d_walkthrough', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_walkthrough_matches_expected_results():
    w = _walkthrough()
    X3, clusters, markers = w.load_inputs()
    gres, info = w.build(X3, clusters, markers)
    found = w.summarise(gres, info)
    expected = json.loads(w.RESULTS.read_text())
    assert w.compare(found, expected) == []
    assert found['counts']['cells'] == 2730


def test_compare_reports_a_changed_value():
    w = _walkthrough()
    expected = json.loads(w.RESULTS.read_text())
    changed = json.loads(json.dumps(expected))
    changed['exact']['decision_mean'] += 1e-6
    changed['counts']['kept_grid_points'] += 1
    bad = w.compare(changed, expected)
    assert len(bad) == 2 and any('decision_mean' in b for b in bad) and any('kept_grid_points' in b for b in bad)
    changed = json.loads(json.dumps(expected))
    changed['exact']['decision_mean'] += 1e-10                              # within the tolerance
    assert w.compare(changed, expected) == []

import importlib.util
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from netheal_research.common_diagnosis_io import question, diagnosis_options, public_audit, combine


def test_common_question_matches_frozen_jev_without_loading_environment():
    # Extract the legacy function and constant through AST, without executing its API/environment imports.
    import ast
    source = ROOT / 'netheal-topology-generalization/scripts/jev_official_eval_v4.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    selected = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == 'question') or
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'INSTRUCTIONS' for t in n.targets))]
    ns = {'json': __import__('json')}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), ns)
    for opts in (['link_failure', 'device_failure'], ['a->b', 'b->a'], [{'source': 'a', 'destination': 'b'}]):
        assert question('Select location for predicted fault_type link_failure.', opts) == ns['question']('Select location for predicted fault_type link_failure.', opts)


def test_candidates_do_not_depend_on_labels_and_links_are_direct():
    catalog = [{'category': 'diagnosis', 'action_type': 'link_failure', 'parameters': {'location': x}} for x in ['a->b', 'b->a', 'a->c']]
    assert diagnosis_options(catalog) == ['link_failure']
    assert diagnosis_options(catalog, 'link_failure') == ['a->b', 'b->a', 'a->c']


def test_truth_rejected_recursively():
    with pytest.raises(AssertionError):
        public_audit({'history': [{'result': {'ground_truth_fault': 'x'}}]})


def test_weighted_probabilities_and_stable_tie():
    q = question('Choose.', ['a', 'b'])
    plan = {'chunks': [{'input': {'questions': q}, 'weight': 1}, {'input': {'questions': q}, 'weight': 3}]}
    ans = combine(plan, [{'c000': 1., 'c001': 0.}, {'c000': 0., 'c001': 1.}])['answers']['decision']
    assert ans == {'choice': 'c001', 'probabilities': {'c000': .25, 'c001': .75}}
    assert combine(plan, [{'c000': .5, 'c001': .5}] * 2)['answers']['decision']['choice'] == 'c000'


def test_builder_stops_before_diagnosis_and_ignores_targets():
    path = ROOT / 'experiments/official_benchmark/laya/common_io_baseline.py'
    spec = importlib.util.spec_from_file_location('baseline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = {'hint': 'public', 'truth': {'type': 'secret'}, 'questions': ['secret'], 'trace': [
        {'tool': 'scan_network', 'arguments': {}, 'result': {'info': {'network_size': 2}}},
        {'tool': 'submit_diagnosis', 'arguments': {'fault_type': 'secret'}, 'result': {'ground_truth_fault': 'secret'}}]}
    state = module.build_case(c, [])
    assert len(state['history']) == 1 and 'secret' not in str(state)
    c['truth'] = {'type': 'changed'}
    assert module.build_case(c, []) == state
    assert module.same_location('a->b', 'b->a')
    assert not module.same_location('a', 'b')

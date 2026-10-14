import importlib.util
from pathlib import Path
from collections import Counter

def test_control_dataset_is_balanced_disjoint_and_tag_is_isolated():
    path=Path(__file__).resolve().parents[1]/'scripts/laya_learning_controls.py'
    spec=importlib.util.spec_from_file_location('learning_controls',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    train=module.select('train',4);valid=module.select('validation',8)
    assert len(train)==16 and len(valid)==32
    assert set(Counter(r['expected']['fault'] for r in train).values())=={4}
    assert set(Counter(r['expected']['fault'] for r in valid).values())=={8}
    assert not {r['episode_seed'] for r in train}&{r['episode_seed'] for r in valid}
    plain=module.convert(train,'real_table');control=module.convert(train,'label_tag')
    for source,real,tag in zip(train,plain,control):
        assert real['state']==source['state']
        assert 'CONTROL ONLY' not in real['state']
        assert tag['state'].split('\n',1)[1]==real['state']
        assert tag['state'].startswith('CONTROL ONLY: correct fault type = '+real['expected']['fault']+'.\n')
        assert real['questions']==tag['questions'] and real['expected']==tag['expected']
        assert set(real['questions'])=={'fault'}

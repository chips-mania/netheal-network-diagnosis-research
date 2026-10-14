"""Explicit compatibility entry points for preserved experiment implementations."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from .paths import project_root, workspace_path


def command_for(config, stage, extra=()):
    root = project_root()
    spec = config['stages'][stage]
    interpreter = workspace_path(spec['python'], root)
    cwd = workspace_path(spec.get('cwd', '.'), root)
    command = [str(interpreter), '-B', '-X', 'utf8']
    if 'module' in spec:
        command += ['-m', spec['module']]
    else:
        command += [str(workspace_path(spec['script'], root))]
    command += spec.get('args', []) + list(extra)
    return command, cwd


def main(config_path):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=config['description'])
    parser.add_argument('--list', action='store_true', help='List stages without importing models or calling APIs')
    parser.add_argument('--execute', action='store_true', help='Actually run the selected stage; default is preview only')
    parser.add_argument('stage', nargs='?', choices=list(config['stages']))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.list or not args.stage:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    extra = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    command, cwd = command_for(config, args.stage, extra)
    print(json.dumps({'command': command, 'cwd': str(cwd), 'execute': args.execute}, ensure_ascii=False), flush=True)
    if args.execute:
        env = dict(os.environ)
        root = project_root()
        pythonpath = [root / 'src', root / 'netheal-topology-generalization', root / 'netheal-ai-agent-benchmark']
        env['PYTHONPATH'] = os.pathsep.join(map(str, pythonpath)) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        raise SystemExit(subprocess.call(command, cwd=cwd, env=env))

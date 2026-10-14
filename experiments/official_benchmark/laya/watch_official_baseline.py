"""CPU-only watcher: persist resource samples and finalize report after evaluation."""
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/official_benchmark/laya/2026-10-12_official_baseline_r01'


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    matches = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmd = p.info.get('cmdline') or []
            if (p.info.get('name') or '').lower().startswith('python') and 'evaluate' in cmd and any(str(x).endswith('official_baseline.py') for x in cmd):
                matches.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # venv redirector and base interpreter may both appear; monitor all of them.
    assert matches, 'No active evaluation process found; watcher does not launch evaluations'
    pids = [p.pid for p in matches]
    save('watcher.json', {'status': 'monitoring', 'evaluation_pids': pids, 'watcher_pid': psutil.Process().pid})
    while any(p.is_running() for p in matches):
        sample = {'unix': time.time(), 'evaluation_pids': pids, 'rss_bytes': {}}
        for p in matches:
            try:
                sample['rss_bytes'][str(p.pid)] = p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        try:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, creationflags=0x08000000)
            mem, util = map(float, raw.strip().split(','))
            sample.update(device_memory_mib=mem, gpu_utilization_pct=util)
        except Exception:
            pass
        with (OUT / 'resource_samples.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(sample) + '\n')
        if (OUT / 'manifest.json').exists():
            manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
            if manifest.get('status') == 'completed':
                break
        time.sleep(30)
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8')) if (OUT / 'manifest.json').exists() else {}
    if manifest.get('status') == 'completed':
        with (OUT / 'report.log').open('a', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(Path(__file__).with_name('report_official_baseline.py'))], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        save('watcher.json', {'status': 'report_completed' if result.returncode == 0 else 'report_failed', 'report_exit_code': result.returncode, 'evaluation_pids': pids})
    else:
        save('watcher.json', {'status': 'evaluation_process_exited_before_completion', 'evaluation_pids': pids, 'manifest': manifest})


if __name__ == '__main__':
    main()

"""Sequential completion driver; waits for frozen training before any GPU inference."""
import gc, time, json
from pathlib import Path
import torch
import laya_observation_controls as controls
import laya_observation_batched_eval as batched
import report_laya_observation_controls as report
from laya_observation_data import OUT, sha

def main():
    torch.set_num_threads(4)
    deadline=time.monotonic()+3600
    print('WAITING FOR ALL SIX FROZEN TRAINING CHECKPOINTS',flush=True)
    while not (OUT/'training_complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Training completion manifest not available')
        time.sleep(2)
    if not (OUT/'batch_conformance.json').exists():
        print('TRAIN-ONLY BATCH CONFORMANCE',flush=True)
        batched.conformance();gc.collect();torch.cuda.empty_cache()
    audit=json.loads((OUT/'batch_conformance.json').read_text())
    assert audit['status']=='completed_verified' and audit['source_sha256']==sha(Path(batched.__file__))
    print('OPENING FIXED EVALUATION',flush=True)
    controls.predict=batched.predict;controls.evaluate=batched.evaluate
    controls.evaluate_all()
    controls.predict=batched.original_predict;controls.evaluate=batched.original_evaluate
    print('RESTORING SIX CHECKPOINTS FOR INDEPENDENT CANONICAL VERIFICATION',flush=True)
    if not (OUT/'restoration_audit.json').exists():controls.restore()
    print('VERIFYING RAW RESULTS AND WRITING REPORT',flush=True)
    report.main()
    print('ALL PHASES COMPLETED AND VERIFIED',flush=True)

if __name__=='__main__':main()

"""Preview or explicitly execute preserved experiment stages."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / '.gitmodules').is_file())
sys.path.insert(0, str(ROOT / 'src'))
from netheal_research.experiment_runner import main

if __name__ == '__main__':
    main(HERE.with_name('config.json'))

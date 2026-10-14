"""Workspace paths shared by new entry points; importing this module performs no IO writes."""
from pathlib import Path


def project_root(start=None):
    start = Path(start or __file__).resolve()
    for candidate in (start, *start.parents):
        if (candidate / '.gitmodules').is_file() and (candidate / 'netheal-topology-generalization').is_dir():
            return candidate
    raise FileNotFoundError('NetHeal research workspace not found')


def workspace_path(relative, root=None):
    root = Path(root or project_root()).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path must stay inside the research workspace')
    return path

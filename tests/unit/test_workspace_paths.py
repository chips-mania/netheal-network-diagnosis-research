import unittest

from netheal_research.paths import project_root, workspace_path


class WorkspacePathsTests(unittest.TestCase):
    def test_repository_root_from_nested_path(self):
        root = project_root()
        self.assertEqual(project_root(root / 'experiments/official_benchmark/laya'), root)

    def test_reject_escape(self):
        with self.assertRaises(ValueError):
            workspace_path('../outside')

    def test_moved_result_is_reachable_through_old_path(self):
        root = project_root()
        old = root / 'netheal-topology-generalization/results/netheal_official_jev_20261011_v4'
        new = root / 'runs/official_benchmark/jev/netheal_official_jev_20261011_v4'
        self.assertEqual(old.resolve(), new.resolve())


if __name__ == '__main__':
    unittest.main()

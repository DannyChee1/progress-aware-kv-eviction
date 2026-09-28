import json
from pathlib import Path
import tempfile
import unittest

from orderkv.experiment import RunStore, digest


class ExperimentTests(unittest.TestCase):
    def test_resume_requires_exact_completed_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(directory)
            key = digest({'revision': 'a', 'case': 1, 'condition': 'R0'})
            store.save(key, {'status': 'capped'})
            self.assertFalse(store.completed(key))
            store.save(key, {'status': 'failed'})
            self.assertFalse(store.completed(key))
            store.save(key, {'status': 'complete'})
            self.assertTrue(store.completed(key))
            self.assertFalse(store.completed(digest({'revision': 'b', 'case': 1, 'condition': 'R0'})))
            rows = [json.loads(line) for line in (Path(directory) / 'runs.jsonl').read_text().splitlines()]
            self.assertEqual([row['status'] for row in rows], ['capped', 'failed', 'complete'])

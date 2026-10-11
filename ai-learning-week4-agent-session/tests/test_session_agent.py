import json
import tempfile
import unittest
from pathlib import Path
from src.agent import init_session, review, run_session
from src.memory import Session
from src.tools import time_budget

class PersistentAgentTests(unittest.TestCase):
    def test_plan_has_dependencies(self):
        session = init_session('学习 agent，安排 6 小时', tempfile.mktemp(suffix='.json'))
        self.assertEqual(session.tasks[1].depends_on, ['t1'])
        self.assertEqual(session.tasks[2].depends_on, ['t2'])

    def test_run_respects_dependencies_and_persists(self):
        path = Path(tempfile.mktemp(suffix='.json'))
        session = init_session('学习 agent，安排 6 小时', path)
        first = run_session(session)
        session.save(path)
        restored = Session.load(path)
        self.assertEqual(len(first), 1)
        self.assertEqual(restored.tasks[0].status, 'done')
        second = run_session(restored)
        restored.save(path)
        self.assertEqual(len(second), 1)
        self.assertGreater(restored.completion(), 0)

    def test_review_reports_completion_and_success(self):
        session = init_session('学习 agent，安排 6 小时', tempfile.mktemp(suffix='.json'))
        run_session(session)
        result = review(session)
        self.assertEqual(result['completion_rate'], 0.25)
        self.assertEqual(result['tool_success_rate'], 1.0)

    def test_time_budget_rejects_invalid_hours(self):
        with self.assertRaises(ValueError):
            time_budget({'hours': 0})

    def test_llm_mode_without_key_falls_back(self):
        session = init_session('学习 agent', tempfile.mktemp(suffix='.json'), use_llm=True)
        self.assertEqual(session.tasks[0].id, 't1')

if __name__ == '__main__':
    unittest.main()

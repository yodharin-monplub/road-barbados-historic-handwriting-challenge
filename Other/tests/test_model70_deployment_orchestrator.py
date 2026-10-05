import os
from pathlib import Path
import subprocess
import unittest

from model70 import orchestrate_deployment as m


class Model70DeploymentOrchestratorTests(unittest.TestCase):
    def test_process_record_reads_identity_and_command(self):
        record = m._process_record(os.getpid())
        self.assertIsNotNone(record)
        assert record is not None
        start_ticks, state, command = record
        self.assertTrue(start_ticks.isdigit())
        self.assertNotEqual(state, "Z")
        self.assertTrue(command)

    def test_absent_process_returns_none(self):
        self.assertIsNone(m._process_record(2**31 - 1))

    def test_commands_are_fixed_and_contain_no_upload_or_network_tool(self):
        commands = m.deployment_commands()
        self.assertEqual(len(commands), 5)
        rendered = "\n".join(" ".join(command) for command in commands)
        self.assertIn("model35/check_candidate.py", rendered)
        self.assertIn("model36/export_checkpoint.py", rendered)
        self.assertEqual(rendered.count("model36/infer.py"), 2)
        self.assertIn("model35/build_submission.py", rendered)
        token_basenames = {
            Path(token).name.lower() for command in commands for token in command
        }
        for forbidden in ("upload", "upload.py", "curl", "wget", "zindi"):
            self.assertNotIn(forbidden, token_basenames)

    def test_wait_rejects_unrelated_live_pid(self):
        process = subprocess.Popen(["sleep", "5"])
        try:
            with self.assertRaisesRegex(m.DeploymentContinuationError, "not the expected"):
                m.wait_for_training(process.pid, timeout_seconds=1, poll_seconds=1)
        finally:
            process.terminate()
            process.wait()


if __name__ == "__main__":
    unittest.main()

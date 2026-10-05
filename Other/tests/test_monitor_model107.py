import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location("monitor107",Path(__file__).resolve().parents[1]/"scripts/monitor_model107.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VastStopStateTests(unittest.TestCase):
    def test_exited_container_with_stopped_instance_is_complete(self):
        self.assertTrue(module.is_stopped({"actual_status":"exited","cur_state":"stopped","intended_status":"stopped"}))

    def test_stop_request_does_not_mean_stopped(self):
        self.assertFalse(module.is_stopped({"actual_status":"running","cur_state":"running","intended_status":"stopped"}))
        self.assertFalse(module.is_stopped({"actual_status":"exited","cur_state":"running","intended_status":"running"}))


if __name__=="__main__": unittest.main()

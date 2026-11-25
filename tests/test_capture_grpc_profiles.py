import importlib.util, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class CaptureClockTests(unittest.TestCase):
    def test_static_capture_can_skip_unaccepted_timestamp_rpc_without_inventing_time(
        self,
    ):
        path = (
            Path(__file__).resolve().parents[1]
            / "ros/src/data_collection/src/data_collection/capture_grpc.py"
        )
        spec = importlib.util.spec_from_file_location("capture_clock_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for enabled in (False, True):
            client = module.MoraiCaptureClient.__new__(module.MoraiCaptureClient)
            client._request_type = SimpleNamespace
            client._timeout = 0.7
            client._read_sim_time = enabled
            client._sim_time_raw = Mock(return_value=123)
            client._sensor = SimpleNamespace(
                SaveSensorData=Mock(
                    return_value=SimpleNamespace(status=1, description="ok")
                )
            )
            result = client.capture("unique_frame", "output_alias")
            self.assertEqual(result["morai_sim_time_raw"], 123 if enabled else None)
            self.assertEqual(result["morai_sim_time_ns"], 123000 if enabled else None)
            self.assertEqual(client._sim_time_raw.call_count, int(enabled))
            self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()

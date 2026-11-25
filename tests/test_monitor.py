import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from inference import LlamaCpp
from server import Collector, ProcInfo, gpu_memory_mode


class HardwareTests(unittest.TestCase):
    def test_memory_layout_follows_the_gpu(self):
        for name, expected in (
            ("NVIDIA GB10", "unified"),
            ("NVIDIA GeForce RTX 2080 Ti", "discrete"),
            ("NVIDIA GeForce RTX 5090", "discrete"),
        ):
            with self.subTest(name=name):
                self.assertEqual(gpu_memory_mode(name, "auto"), expected)
        self.assertEqual(gpu_memory_mode("Unknown NVIDIA device", "unified"), "unified")

    def test_primary_gpu_keeps_the_fleet_format(self):
        gpu = {"name": "NVIDIA GB10", "util": 70, "temp": 60, "power_w": 42.0,
               "procs": [{"pid": 1, "vram": 1024}]}
        self.assertEqual(Collector.primary_gpu(gpu), {
            "name": "NVIDIA GB10", "util": 70, "temp": 60, "power_w": 42.0,
            "procs": [{"pid": 1, "gpu_mem": 1024}],
        })

    def test_process_table_redacts_api_keys(self):
        cmdline = b"llama-server\0--api-key\0secret-one\0--api-key=secret-two\0--port\0"
        with patch.object(ProcInfo, "_read", side_effect=[b"llama-server", cmdline]):
            _, cmd = ProcInfo().static(1)
        self.assertEqual(cmd, "llama-server --api-key=[redacted] --api-key=[redacted] --port")


class InferenceTests(unittest.TestCase):
    def test_llama_slots_keep_metrics_available_with_exporter_disabled(self):
        def get(url, *, headers):
            self.assertEqual(headers, {"Authorization": "Bearer test-key"})
            if url.endswith("/v1/models"):
                return b'{"data":[{"id":"test-model"}]}'
            if url.endswith("/metrics"):
                raise HTTPError(url, 501, "disabled", {}, None)
            self.assertTrue(url.endswith("/slots"))
            return b'[{"is_processing":true,"n_ctx":131072,"prompt":"private fixture"}]'

        with patch("inference._get", side_effect=get) as request:
            llama = LlamaCpp("http://localhost:18300/v1", "test-key")
            sample = llama.sample(1)
            self.assertEqual(llama.sample(2), sample)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(llama.sample(4), sample)
            self.assertEqual(request.call_count, 5)
        self.assertEqual(sample["models"], [{"name": "test-model", "running": 1, "ctx": 131072}])
        self.assertNotIn("private fixture", json.dumps(sample))

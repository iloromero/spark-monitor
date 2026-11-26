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
            self.assertEqual(request.call_count, 3)   # /v1/models, /slots, /metrics (501: skipped for 60 s)
            self.assertEqual(llama.sample(4), sample)
            self.assertEqual(request.call_count, 5)   # /v1/models, /slots
        self.assertEqual(sample["models"], [{"name": "test-model", "running": 1, "ctx": 131072, "kv_pct": 0.0}])
        self.assertNotIn("private fixture", json.dumps(sample))


class DiscoveryTests(unittest.TestCase):
    def test_classify_tells_servers_apart_by_owner(self):
        from inference import classify
        bodies = {"vllm": b'{"data":[{"id":"m","owned_by":"vllm"}]}',
                  "llamacpp": b'{"data":[{"id":"m","owned_by":"llamacpp"}]}',
                  "ollama": b'{"object":"list","data":[{"id":"m","owned_by":"library"}]}'}
        for kind, body in bodies.items():
            with self.subTest(kind=kind), patch("inference._get", return_value=body):
                self.assertEqual(classify("http://127.0.0.1:1"), kind)
        with patch("inference._get", side_effect=ConnectionRefusedError()):
            self.assertIsNone(classify("http://127.0.0.1:1"))

    def test_key_protected_llama_server_is_recognised(self):
        from inference import classify

        def get(url, **kw):
            if url.endswith("/v1/models"):
                raise HTTPError(url, 401, "unauthorized", {}, None)
            return b'{"status":"ok"}'
        with patch("inference._get", side_effect=get):
            self.assertEqual(classify("http://127.0.0.1:1"), "llamacpp")

    def test_scan_follows_servers_that_start_and_stop(self):
        from inference import Discovery, skip_port
        ports = {11434: "127.0.0.1", 18400: "127.0.0.1", 8000: "127.0.0.1", 29521: "127.0.0.1", 40001: "10.0.0.1"}
        kinds = {11434: "ollama", 18400: "llamacpp", 8000: "vllm"}
        asked = []

        def classify(base, api_key=None):
            port = int(base.rsplit(":", 1)[1]); asked.append(port); return kinds.get(port)
        d = Discovery(8888, ports=lambda: dict(ports), classify=classify)
        found = d.scan(0, {"vllm": [], "llamacpp": []})
        self.assertEqual([e.base for e in found["llamacpp"]], ["http://127.0.0.1:18400"])
        self.assertEqual([e.label for e in found["vllm"]], ["vLLM"])
        self.assertNotIn(29521, asked)            # torch rendezvous port: never probed
        self.assertNotIn(40001, asked)            # ephemeral worker port: never probed
        self.assertTrue(skip_port(29521) and skip_port(40001) and not skip_port(18400))
        del ports[18400]                          # the llama.cpp server stopped
        found = d.scan(d.SCAN_S + 1, {"vllm": [], "llamacpp": []})
        self.assertEqual(found["llamacpp"], [])
        asked.clear()
        found = d.scan(2 * d.SCAN_S + 2, {"vllm": [], "llamacpp": []})
        self.assertNotIn(11434, asked)            # not a vLLM/llama.cpp server: asked again only after RECHECK_S


class TempTests(unittest.TestCase):
    def test_devlink_serials_map_pci_functions_to_one_chip(self):
        from types import SimpleNamespace
        from server import TempCollector
        out = json.dumps({"info": {f"pci/{a}": {"serial_number": "SER1"} for a in
                                   ("0000:01:00.0", "0000:01:00.1", "0002:01:00.0", "0002:01:00.1")}})
        with patch("server.subprocess.run", return_value=SimpleNamespace(stdout=out)):
            serials = TempCollector._device_serials()
        self.assertEqual(set(serials.values()), {"SER1"})
        self.assertEqual(len(serials), 4)
        with patch("server.subprocess.run", side_effect=FileNotFoundError()):
            self.assertEqual(TempCollector._device_serials(), {})


class LlamaKvTests(unittest.TestCase):
    def test_kv_use_prefix_hits_and_rates(self):
        state = {"pt": 1000, "ct": 9000, "gt": 500}

        def get(url, *, headers):
            if url.endswith("/v1/models"):
                return b'{"data":[{"id":"m"}]}'
            if url.endswith("/slots"):
                return json.dumps([{"is_processing": True, "n_ctx": 1000, "n_prompt_tokens": 400,
                                    "n_prompt_tokens_cache": 300, "next_token": [{"n_decoded": 100}]}]).encode()
            return (f"llamacpp:prompt_tokens_total {state['pt']}\nllamacpp:prompt_tokens_cached_total {state['ct']}\n"
                    f"llamacpp:tokens_predicted_total {state['gt']}\nllamacpp:requests_processing 1\n").encode()

        with patch("inference._get", side_effect=get):
            llama = LlamaCpp("http://localhost:1", None)
            first = llama.sample(10)["models"][0]
            state.update(pt=1400, gt=700)
            second = llama.sample(20)["models"][0]
        self.assertEqual(first["kv_pct"], 50.0)                 # (400 prompt + 100 generated) / 1000
        self.assertEqual(first["cache_hit_pct"], 90.0)          # 9000 cached / 10000 prompt tokens
        self.assertNotIn("gen_tps", first)                      # no baseline yet
        self.assertEqual((second["prompt_tps"], second["gen_tps"]), (40.0, 20.0))

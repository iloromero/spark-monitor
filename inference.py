"""Read-only model metadata and serving metrics for vLLM and llama.cpp."""

import json
import math
import threading
import time
import traceback
import urllib.request
from urllib.error import HTTPError


def _get(url, timeout=0.5, *, headers=None):
    request = urllib.request.Request(url)
    for name, value in (headers or {}).items():
        request.add_unredirected_header(name, value)
    with urllib.request.urlopen(request, timeout=timeout) as r:
        return r.read()


VLLM_METRICS = {  # Prometheus series -> field; summed across engines per model
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv",
    "vllm:gpu_cache_usage_perc": "kv",  # name before vLLM 0.10
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:prompt_tokens_cached_total": "cached_tokens",  # prefix-cache hits inside prompt_tokens
    "vllm:prefix_cache_queries_total": "cache_queries",
    "vllm:prefix_cache_hits_total": "cache_hits",
    "vllm:generation_tokens_total": "gen_tokens",
}


def _prom(text):
    """{model_name: {field: value}} for the VLLM_METRICS series in a /metrics body."""
    out = {}
    for line in text.splitlines():
        name, sep, rest = line.partition("{")
        field = VLLM_METRICS.get(name)
        if not field or not sep:
            continue
        labels, _, value = rest.rpartition("}")
        model = next((kv.split("=", 1)[1].strip('"') for kv in labels.split(",")
                      if kv.startswith("model_name=")), "?")
        try:
            v = float(value.split()[0])
        except (ValueError, IndexError):
            continue
        m = out.setdefault(model, {})
        m[field] = m.get(field, 0.0) + v
    return out


class VLLM:
    """Models served by a vLLM OpenAI-compatible server, from /v1/models and
    /metrics (load, KV cache, token throughput); polled every 3 s."""

    def __init__(self, url):
        # optional display label: "TP2 cluster=http://host:port"
        label, sep, rest = url.partition("=")
        if sep and not label.startswith("http"):
            url = rest
        else:
            label = "vLLM"
        self.label = label
        self.base = url.rstrip("/")
        self.t = 0
        self.last = self.offline()
        self.prev = None  # (t, {model: (prompt_tokens, gen_tokens)}) for rates

    def offline(self):
        return {"up": False, "url": self.base, "label": self.label, "models": []}

    def sample(self, now):
        if now - self.t < 3:
            return self.last
        self.t = now
        try:
            models = json.loads(_get(self.base + "/v1/models")).get("data", [])
        except (OSError, ValueError):
            self.last, self.prev = self.offline(), None
            return self.last
        try:
            metrics = _prom(_get(self.base + "/metrics").decode("utf-8", "replace"))
        except OSError:  # metrics can be disabled; the model list still counts as up
            metrics = {}
        dt = now - self.prev[0] if self.prev else None
        prev = self.prev[1] if self.prev else {}

        def rate(name, i, cur):
            p = prev.get(name)
            if not dt or p is None or cur is None or p[i] is None or cur < p[i]:
                return None  # no baseline yet (first sample, counter missing last time, or server restarted)
            return (cur - p[i]) / dt

        rows, counters = [], {}
        for m in models:
            name = m.get("id")
            x = metrics.get(name, {})
            pt, gt = x.get("prompt_tokens"), x.get("gen_tokens")
            if pt is not None:  # rate only the prefill actually computed, not cache hits
                pt -= x.get("cached_tokens", 0.0)
            q = x.get("cache_queries")
            counters[name] = (pt, gt)
            rows.append({
                "name": name,
                "ctx": m.get("max_model_len"),
                "running": x.get("running"),
                "waiting": x.get("waiting"),
                "kv_pct": None if x.get("kv") is None else 100 * x["kv"],
                "cache_hit_pct": 100 * x.get("cache_hits", 0.0) / q if q else None,
                "prompt_tps": rate(name, 0, pt),
                "gen_tps": rate(name, 1, gt),
            })
        self.prev = (now, counters)
        self.last = {"up": True, "url": self.base, "label": self.label, "models": rows}
        return self.last


class InferencePoller(threading.Thread):
    """Refresh configured serving endpoints away from the hardware sampling thread."""

    def __init__(self, endpoints):
        super().__init__(daemon=True, name="inference")
        self.endpoints = endpoints
        self.results = {name: [endpoint.offline() for endpoint in group]
                        for name, group in endpoints.items()}
        self.kick = threading.Event()

    def run(self):
        while True:
            self.kick.wait()
            self.kick.clear()
            now = time.monotonic()
            self.results = {name: [self._sample(endpoint, now) for endpoint in group]
                            for name, group in self.endpoints.items()}

    @staticmethod
    def _sample(endpoint, now):
        try:
            return endpoint.sample(now)
        except Exception:
            traceback.print_exc()
            return endpoint.offline()


LLAMACPP_METRICS = {
    "llamacpp:requests_processing": "running",
    "llamacpp:requests_deferred": "waiting",
    "llamacpp:prompt_tokens_seconds": "prompt_tps",
    "llamacpp:predicted_tokens_seconds": "gen_tps",
}


class LlamaCpp:
    """Read model metadata and optional metrics from a single-model llama-server."""

    def __init__(self, url, api_key):
        self.base = url.rstrip("/").removesuffix("/v1")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.last = self.offline()
        self.t = float("-inf")
        self.retry_metrics_at = float("-inf")

    def offline(self):
        return {"up": False, "url": self.base, "label": "llama.cpp", "models": []}

    def _metrics(self, now):
        if now < self.retry_metrics_at:
            return {}
        try:
            body = _get(self.base + "/metrics", headers=self.headers).decode("utf-8", "replace")
        except HTTPError as exc:
            exc.close()
            if exc.code in (404, 501):  # disabled on this instance; check again after a minute
                self.retry_metrics_at = now + 60
            return {}
        except OSError:
            return {}
        metrics = {}
        for line in body.splitlines():
            fields = line.split()
            if len(fields) < 2 or fields[0] not in LLAMACPP_METRICS:
                continue
            try:
                value = float(fields[1])
            except ValueError:
                continue
            if math.isfinite(value):
                metrics[LLAMACPP_METRICS[fields[0]]] = value
        return metrics

    def _slots(self):
        try:
            slots = json.loads(_get(self.base + "/slots", headers=self.headers))
        except (OSError, ValueError):
            return {}
        if not isinstance(slots, list) or not slots or not all(isinstance(s, dict) for s in slots):
            return {}
        stats = {}
        if all(isinstance(s.get("is_processing"), bool) for s in slots):
            stats["running"] = sum(s["is_processing"] for s in slots)
        contexts = {s["n_ctx"] for s in slots if isinstance(s.get("n_ctx"), int) and s["n_ctx"] > 0}
        if len(contexts) == 1:
            stats["ctx"] = contexts.pop()
        return stats

    def sample(self, now):
        if now - self.t < 3:
            return self.last
        self.t = now
        try:
            models = json.loads(_get(self.base + "/v1/models", headers=self.headers)).get("data", [])
        except (OSError, ValueError):
            self.last = self.offline()
            return self.last
        metrics = {}
        if len(models) == 1:
            metrics = self._metrics(now)
            if not metrics:
                metrics = self._slots()
        self.last = {
            "up": True, "url": self.base, "label": "llama.cpp",
            "models": [{"name": model.get("id"), **metrics} for model in models],
        }
        return self.last

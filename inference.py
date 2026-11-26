"""Read-only model metadata and serving metrics for vLLM and llama.cpp, plus discovery of the
servers running on this machine (found on its listening ports, so the view follows whichever is up)."""

import http.client
import json
import math
import socket
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
        self.t = float("-inf")
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
    """Refresh serving endpoints away from the hardware sampling thread. With a Discovery, servers that
    start or stop on this machine are added or dropped automatically; configured ones are always kept."""

    def __init__(self, endpoints, discovery=None):
        super().__init__(daemon=True, name="inference")
        self.configured = {name: list(endpoints.get(name, [])) for name in ("vllm", "llamacpp")}
        self.endpoints = {name: list(group) for name, group in self.configured.items()}
        self.discovery = discovery
        self.results = {name: [endpoint.offline() for endpoint in group]
                        for name, group in self.endpoints.items()}
        self.kick = threading.Event()

    def run(self):
        while True:
            self.kick.wait()
            self.kick.clear()
            self.refresh(time.monotonic())

    def refresh(self, now):
        if self.discovery:
            try:
                found = self.discovery.scan(now, self.configured)
                self.endpoints = {name: self.configured[name] + found.get(name, [])
                                  for name in self.configured}
            except Exception:
                traceback.print_exc()
        results = {}
        for name, group in self.endpoints.items():
            rows = []
            for endpoint in group:
                r = self._sample(endpoint, now)
                # a discovered server that stopped answering is dropped; a configured one is shown as down
                if r.get("up") or endpoint in self.configured[name]:
                    rows.append(r)
            results[name] = rows
        self.results = results

    @staticmethod
    def _sample(endpoint, now):
        try:
            return endpoint.sample(now)
        except Exception:
            traceback.print_exc()
            return endpoint.offline()


LLAMACPP_METRICS = {  # /metrics (llama-server --metrics)
    "llamacpp:requests_processing": "running",
    "llamacpp:requests_deferred": "waiting",
    "llamacpp:prompt_tokens_total": "prompt_tokens",          # prompt tokens computed (cache misses)
    "llamacpp:prompt_tokens_cached_total": "cached_tokens",   # prompt tokens served from the prefix cache
    "llamacpp:tokens_predicted_total": "gen_tokens",
}


class LlamaCpp:
    """Read model metadata and optional metrics from a single-model llama-server."""

    def __init__(self, url, api_key, label="llama.cpp"):
        self.label = label
        self.base = url.rstrip("/").removesuffix("/v1")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.last = self.offline()
        self.t = float("-inf")
        self.retry_metrics_at = float("-inf")
        self.prev = None  # (t, prompt_tokens, gen_tokens) for rates

    def offline(self):
        return {"up": False, "url": self.base, "label": self.label, "models": []}

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
        except (OSError, ValueError, http.client.HTTPException):
            return {}
        if not isinstance(slots, list) or not slots or not all(isinstance(s, dict) for s in slots):
            return {}
        num = lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
        stats = {}
        if all(isinstance(s.get("is_processing"), bool) for s in slots):
            stats["running"] = sum(s["is_processing"] for s in slots)
        contexts = {s["n_ctx"] for s in slots if isinstance(s.get("n_ctx"), int) and s["n_ctx"] > 0}
        if len(contexts) == 1:
            stats["ctx"] = contexts.pop()
        total = sum(num(s.get("n_ctx")) for s in slots)
        if total:  # KV cache held by active requests (prompt + generated so far), as vLLM reports it
            used = 0
            for s in slots:
                if s.get("is_processing"):
                    nt = s.get("next_token")
                    nt = nt[0] if isinstance(nt, list) and nt and isinstance(nt[0], dict) else nt
                    used += num(s.get("n_prompt_tokens")) + num(nt.get("n_decoded") if isinstance(nt, dict) else 0)
            stats["kv_pct"] = min(100.0, 100 * used / total)
        prompt = sum(num(s.get("n_prompt_tokens")) for s in slots)
        if prompt:  # fallback hit rate (current/last request) when /metrics is off
            stats["cache_hit_pct"] = 100 * sum(num(s.get("n_prompt_tokens_cache")) for s in slots) / prompt
        return stats

    def sample(self, now):
        if now - self.t < 3:
            return self.last
        self.t = now
        try:
            models = json.loads(_get(self.base + "/v1/models", headers=self.headers)).get("data", [])
        except HTTPError as exc:
            exc.close()
            up = False
            if exc.code in (401, 403):  # running, but needs a key (--llamacpp-api-key-file)
                try:
                    up = json.loads(_get(self.base + "/health")).get("status") == "ok"
                except (OSError, ValueError, AttributeError, http.client.HTTPException):
                    pass
            self.last = ({"up": True, "url": self.base, "label": self.label,
                          "models": [{"name": "API key needed to show details"}]} if up else self.offline())
            return self.last
        except (OSError, ValueError, http.client.HTTPException):
            self.last = self.offline()
            return self.last
        metrics = {}
        if len(models) == 1:  # /slots: context size and KV use; /metrics (if enabled): counters for rates
            metrics = self._slots()
            m = self._metrics(now)
            for k in ("running", "waiting"):
                if k in m:
                    metrics[k] = m[k]
            pt, ct, gt = m.get("prompt_tokens"), m.get("cached_tokens"), m.get("gen_tokens")
            if pt is not None and ct is not None and pt + ct > 0:  # since server start, like vLLM's
                metrics["cache_hit_pct"] = 100 * ct / (pt + ct)
            if pt is not None and gt is not None:
                if self.prev and now > self.prev[0] and pt >= self.prev[1] and gt >= self.prev[2]:
                    dt = now - self.prev[0]
                    metrics["prompt_tps"] = (pt - self.prev[1]) / dt
                    metrics["gen_tps"] = (gt - self.prev[2]) / dt
                self.prev = (now, pt, gt)
        self.last = {
            "up": True, "url": self.base, "label": self.label,
            "models": [{"name": model.get("id"), **metrics} for model in models],
        }
        return self.last


# --- discovery ----------------------------------------------------------------------------------
SKIP_PORTS = {22, 25, 53, 80, 111, 139, 443, 445, 631, 3389, 5432, 6379}


def skip_port(port):
    """Ports never probed: well-known non-HTTP services, the PyTorch distributed rendezvous range
    (29400-29999, e.g. a TP cluster's master port) and the ephemeral range, where multi-node workers
    listen for NCCL/Gloo/Ray traffic. Inference servers use fixed lower ports."""
    return port in SKIP_PORTS or 29400 <= port <= 29999 or port >= 32768


def listening_ports():
    """{port: host to probe} for TCP sockets in LISTEN state (/proc/net/tcp and tcp6). Wildcard binds
    are probed on loopback, specific binds on their own address."""
    out = {}
    for path, v6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        try:
            with open(path) as fh:
                lines = fh.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) < 4 or f[3] != "0A":
                continue
            addr, port_hex = f[1].rsplit(":", 1)
            port = int(port_hex, 16)
            raw = bytes.fromhex(addr)
            if v6:
                ip = socket.inet_ntop(socket.AF_INET6, b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4)))
                if ip.startswith("::ffff:"):
                    ip = ip[7:]
            else:
                ip = socket.inet_ntop(socket.AF_INET, raw[::-1])
            if ip in ("0.0.0.0", "::"):
                host = "127.0.0.1"
            elif ip.startswith("127.0.0.5"):          # systemd-resolved stub listeners
                continue
            else:
                host = f"[{ip}]" if ":" in ip else ip
            if port not in out or host == "127.0.0.1":
                out[port] = host
    return out


def classify(base, api_key=None):
    """'vllm', 'llamacpp', 'ollama' or None for the HTTP server at base. One /v1/models request tells
    the three apart (owned_by); a key-protected llama-server is recognised by its public /health."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        body = json.loads(_get(base + "/v1/models", timeout=0.4, headers=headers))
    except HTTPError as exc:
        exc.close()
        if exc.code in (401, 403):
            try:
                if json.loads(_get(base + "/health", timeout=0.4)).get("status") == "ok":
                    return "llamacpp"
            except (OSError, ValueError, AttributeError):
                pass
        return None
    except (OSError, ValueError, http.client.HTTPException):
        return None
    except Exception:  # a non-HTTP service: never let one odd port stop the scan
        return None
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return None
    owners = {str(m.get("owned_by", "")).lower() for m in data if isinstance(m, dict)}
    if "vllm" in owners:
        return "vllm"
    if "llamacpp" in owners:
        return "llamacpp"
    if owners & {"library", "ollama"}:
        return "ollama"
    try:  # no models listed: llama-server still answers /props
        props = json.loads(_get(base + "/props", timeout=0.4, headers=headers))
        if isinstance(props, dict) and "default_generation_settings" in props:
            return "llamacpp"
    except (OSError, ValueError):
        pass
    return None


class Discovery:
    """Finds vLLM and llama.cpp servers on this machine's listening ports, every SCAN_S seconds. A port
    that answers as neither is asked again only after RECHECK_S, so ordinary web apps see one request
    every few minutes. Ollama stays with --ollama (its own poller)."""

    SCAN_S = 30
    RECHECK_S = 600

    def __init__(self, own_port, api_key=None, ports=listening_ports, classify=classify):
        self.skip = SKIP_PORTS | {own_port}
        self.api_key = api_key
        self.ports, self.classify = ports, classify
        self.t = float("-inf")
        self.known = {}       # port -> (kind, endpoint)
        self.negative = {}    # port -> time to ask again
        self.found = {"vllm": [], "llamacpp": []}

    @staticmethod
    def _port(endpoint):
        try:
            return int(endpoint.base.rsplit(":", 1)[1].split("/")[0])
        except (ValueError, IndexError):
            return None

    def scan(self, now, configured):
        if now - self.t < self.SCAN_S:
            return self.found
        self.t = now
        taken = {self._port(e) for group in configured.values() for e in group}
        listening = self.ports()
        for port in list(self.known):
            if port not in listening:
                del self.known[port]
        for port, host in sorted(listening.items()):
            if port in self.skip or skip_port(port) or port in taken or port in self.known or self.negative.get(port, 0) > now:
                continue
            base = f"http://{host}:{port}"
            kind = self.classify(base, self.api_key)
            if kind == "vllm":
                self.known[port] = (kind, VLLM(f"vLLM={base}"))
            elif kind == "llamacpp":
                self.known[port] = (kind, LlamaCpp(base, self.api_key))
            else:
                self.negative[port] = now + self.RECHECK_S
        self.found = {"vllm": [e for k, e in self.known.values() if k == "vllm"],
                      "llamacpp": [e for k, e in self.known.values() if k == "llamacpp"]}
        for kind, name in (("vllm", "vLLM"), ("llamacpp", "llama.cpp")):  # port only to tell two apart
            group = self.found[kind]
            for e in group:
                e.label = f"{name} :{self._port(e)}" if len(group) > 1 else name
        return self.found

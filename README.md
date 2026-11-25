# Spark Monitor

A lightweight, live resource dashboard for NVIDIA Linux systems, including
the **NVIDIA DGX Spark and Lenovo ThinkStation PGX (GB10)** and discrete-GPU
workstations such as the **RTX 5090** and **ThinkStation P520**.
One Python process collects Linux and NVIDIA metrics and streams them to a
self-contained browser interface. No pip packages, frontend dependencies,
external fonts, database, or build step are required.

## Features

- GPU utilization, power, and either VRAM usage or GPU clock, plus CPU,
  RAM, network utilization, disk activity, and root filesystem gauges.
- Physical core and SMT thread layout, grouped by frequency or cache topology.
- **Discrete GPUs** display dedicated VRAM under Memory, separately from system
  RAM, and report PCIe, fan, clock, and encoder/decoder telemetry where available.
- **GB10 systems** display unified memory with a GPU process allocation breakdown.
  The monitor detects the memory layout from the NVIDIA device and accepts an
  override for other hardware.
- GPU metrics and compute and graphics process allocations come from NVML.
  Slower GPU and sensor readings are sampled less often.
- Disk throughput, active physical network interfaces, GPU/SoC/NVMe/ConnectX
  temperatures, load averages, uptime, and the top eight CPU processes.
- Optional inference-server integration:
  - Ollama: loaded models, memory size, GPU share, context length, and unload time.
  - vLLM: served models, maximum context length, running and queued requests,
    KV-cache usage, prefix-cache hit rate since vLLM started, and
    prefill/generation tokens per second. The prefill (“in”) rate excludes
    prefix-cache hits, so it reflects computed tokens.
  - llama.cpp: served model, active and queued requests, and prompt/generation
    throughput where the server exposes these metrics.
- Light/dark themes with a saved browser preference, responsive cards,
  scrollable process tables, and 120-sample sparklines.
- On-demand sampling: the sampler waits without clients; hidden tabs close
  their stream. Process scans and inference-server polling run at most once per three seconds.

## Stack and architecture

| Component | Implementation |
| --- | --- |
| Backend | Python 3 standard library; `ThreadingHTTPServer` |
| GPU collector | `ctypes` bindings to the installed `libnvidia-ml.so.1` |
| Linux collectors | Read-only access to `/proc`, `/sys`, and filesystem statistics |
| Transport | Server-Sent Events (SSE), JSON at `/stream` |
| Frontend | Plain HTML, CSS, JavaScript, and inline SVG in `index.html` |
| Optional integrations | Ollama `GET /api/ps`; vLLM and llama.cpp `GET /v1/models` and `GET /metrics`; llama.cpp fallback `GET /slots` |
| Background service | systemd user service |
| Existing HTTPS | External Caddy reverse proxy managed with the local MyApps setup |

`Collector` owns the hardware and configured inference-server collectors.
A daemon thread produces a shared JSON snapshot while clients are
connected; a condition variable distributes updates to SSE request threads.
Inference endpoints are polled in the background to keep hardware updates responsive.
The frontend uses `EventSource`, renders the snapshot, and retains only a
rolling history in the browser. The backend does not persist samples.

Files:

```text
server.py                              collectors, sampler, CLI, HTTP/SSE
inference.py                           vLLM and llama.cpp collectors and polling
index.html                             complete browser application
spark-monitor.service                  original deployment unit
examples/spark-monitor.service.example portable unit with path placeholder
examples/Caddyfile.example             optional HTTPS example
```

## Requirements and credentials

- Linux with readable `/proc` and `/sys`; Python **3.12 is verified**.
- NVIDIA driver and its NVML shared library for GPU data. No NVIDIA Python
  package or CUDA toolkit is required by this app.
- A modern browser supporting SSE, SVG, CSS Grid, and `color-mix()`.
- Optional Ollama server (default `http://127.0.0.1:11434`), vLLM
  (default `http://127.0.0.1:8000`), or llama.cpp server (disabled by default).
- Optional systemd and Caddy for service management and HTTPS.

The optional llama.cpp integration can read a bearer key from a local file.
The dashboard has no login or authorization layer. Anyone who can reach the listener can
see machine metrics, process names and command lines, and loaded model names.
Use localhost, a trusted LAN, or an appropriately protected reverse proxy.
HTTPS encrypts traffic but does not add authentication. TLS keys and CA material
belong outside this repository.

## Install and run

```bash
git clone https://github.com/iloromero/spark-monitor.git
cd spark-monitor
python3 server.py --help
python3 server.py
```

Open `http://127.0.0.1:8765`. The private repository requires GitHub access to
clone; use GitHub's secure sign-in or your configured Git credentials.
There is no dependency installation or compilation step.

To serve on a trusted LAN, choose an available port:

```bash
python3 server.py --host 0.0.0.0 --port 8765 --interval 2
```

Do not start a second process on a port already used by the existing service.
Stop a foreground process with Ctrl+C.

### Configuration

All settings are CLI arguments. **The app does not load `.env` files.**
The examples contain placeholders and local addresses, never credentials.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Listener address; `0.0.0.0` exposes all IPv4 interfaces |
| `--port` | `8765` | HTTP port |
| `--interval` | `1.0` | Seconds between samples; use a positive value |
| `--title` | `Spark Monitor` | Dashboard and tab title |
| `--memory-mode` | `auto` | Auto-detect unified GB10 memory or discrete VRAM; override with `unified` or `discrete` |
| `--discrete-gpu` | off | Compatibility alias for `--memory-mode discrete` |
| `--ollama` | `http://127.0.0.1:11434` | Ollama base URL; `/api/ps` is appended; `""` disables |
| `--vllm` | `http://127.0.0.1:8000` | vLLM base URL, optionally `LABEL=URL` (the label replaces "vLLM" in the panel, e.g. `"vLLM · TP2 cluster=http://127.0.0.1:21888"`); repeat for several servers; unreachable ones are hidden; `""` disables |
| `--llamacpp` | disabled | Base URL of a llama.cpp server running a single model |
| `--llamacpp-api-key-file` | unset | Path to a file containing the llama.cpp server's bearer key |
| `--power-max` | `140` on unified-memory GPUs | Fallback power limit in watts when NVML omits the limit |

For inference servers on another trusted host or port, replace the placeholders.
vLLM in Docker is usually published on a host port other than 8000, so pass
that port:

```bash
python3 server.py --ollama http://OLLAMA_HOSTNAME:11434 \
  --vllm http://127.0.0.1:VLLM_HOST_PORT --vllm http://VLLM_HOSTNAME:8000
```

Each request has a 0.5-second timeout and runs at most every three seconds.
Servers that do not respond are hidden while another one is reachable; when
none respond, the memory card lists the configured providers as unreachable. Monitoring
continues either way. Token rates come from differences between vLLM counters,
so they appear from the second poll onward. If `/metrics` is unavailable, the
model list is still shown.

For a workstation serving a model with llama.cpp:

```bash
python3 server.py --title "Workstation Monitor" --interval 3 \
  --ollama "" --vllm "" --llamacpp http://127.0.0.1:18300
```

Add `--llamacpp-api-key-file /path/to/key` for a protected endpoint. The
monitor only reads metadata and metrics. llama.cpp must be started with
[`--metrics`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#get-metrics-prometheus-compatible-metrics-exporter)
to show queued requests and throughput. When `/metrics` is disabled, the
monitor reads `/slots` for active requests and context size if available.

For an RTX workstation running Ollama, use the default Ollama URL or pass its
port explicitly:

```bash
python3 server.py --title "RTX Monitor" --vllm ""
```

## Existing deployment, HTTPS, and schedules

The original machine already runs `spark-monitor.service` as a systemd **user**
service. The checked-in original unit starts:

```text
/usr/bin/python3 %h/Projects/spark-monitor/server.py --host 0.0.0.0 --port 8888 --vllm http://127.0.0.1:18300
```

Port 18300 is the host port Docker publishes for the vLLM container on that
machine.

It uses `Nice=10`, restarts on failure, and is enabled through the user's
default target. The source location is significant because the unit references
it directly; publishing this repository does not relocate the application.

Existing endpoints, with the actual hostname deliberately omitted:

| Route | Existing behavior |
| --- | --- |
| `http://MONITOR_HOSTNAME:8888` | Direct Python HTTP listener |
| `https://MONITOR_HOSTNAME:18888` | Caddy proxy to `127.0.0.1:8888` |

Caddy runs separately as part of the local MyApps deployment and uses a private
LAN certificate authority. Client devices must trust that CA. The live proxy
configuration, host inventory, certificates, private keys, and MyApps files are
outside this repository. The standalone [Caddy example](examples/Caddyfile.example)
illustrates a fresh deployment; it is not a copy of the active proxy configuration.

No Spark Monitor cron entry or systemd timer was found during publication.
Sampling is driven by connected browsers, not a scheduler. Repository publication
requires no service restart, firewall edit, certificate change, or schedule edit.

### Install a service on another machine

Use the [portable example](examples/spark-monitor.service.example), replace
`/APP_DIRECTORY` with the absolute checkout path, and choose the desired bind
address and unused port. The example binds localhost on 8765; the original unit
above binds all interfaces on 8888.

On a **new deployment**, after editing the example:

```bash
mkdir -p ~/.config/systemd/user
cp examples/spark-monitor.service.example ~/.config/systemd/user/spark-monitor.service
systemctl --user daemon-reload
systemctl --user enable --now spark-monitor.service
systemctl --user status spark-monitor.service
journalctl --user -u spark-monitor.service -n 50 --no-pager
```

These install commands replace that user's unit of the same name; do not run
them over an existing deployment without reviewing it first. If it must run
without an interactive login, an administrator can enable lingering for the
service account (`loginctl enable-linger USERNAME`).

## Testing and validation

Hardware detection and model-server fixtures can run without an NVIDIA driver
or an inference server:

```bash
python3 -m unittest discover -s tests
python3 server.py --help
python3 server.py --once --ollama "" --vllm ""
```

For a fresh test instance, use localhost and an unused port, then check from a
second terminal:

```bash
python3 server.py --host 127.0.0.1 --port 18765
# In another terminal:
curl --fail http://127.0.0.1:18765/ -o /dev/null
curl --no-buffer --max-time 3 http://127.0.0.1:18765/stream
```

The SSE request intentionally stays open; curl exit code 28 after its time limit
is expected. The first sample can have null CPU/disk rate fields until a second
sample provides a delta. Check two or more samples. Do not publish raw snapshots:
they contain process command lines and local workload details.

Publication checks and any environment-dependent gaps are recorded in
[docs/VALIDATION.md](docs/VALIDATION.md). Hardware checks require readable Linux
interfaces and an NVIDIA driver. Ollama, vLLM, systemd, and TLS checks require those
external services; they are separate from Python syntax and isolated HTTP checks.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Address already in use | Check `ss -ltn`; choose an unused `--port` instead of stopping the existing service |
| GPU shows n/a | Check driver health and `ldconfig -p` for `libnvidia-ml.so.1`; a missing library allows system monitoring to continue |
| Unexpected memory layout | Use `--memory-mode unified` or `--memory-mode discrete` if the driver reports an unfamiliar GPU name |
| CPU/rate fields initially blank | Wait for successive samples; utilization is calculated from counter deltas |
| No Ollama models | Ensure Ollama is reachable at the configured URL; `/api/ps` lists loaded models, not all installed models |
| vLLM missing or “not reachable” | `curl http://HOST:PORT/v1/models` against the `--vllm` URL; for Docker, use the published host port |
| vLLM shows no KV cache or tok/s | `/metrics` is not reachable or the vLLM version uses different series names; rates also need two polls |
| HTTPS certificate warning | Trust the deployment's private CA through your device's trusted certificate settings |
| HTTPS page loads but no updates | Confirm the proxy streams `/stream` without buffering and allows long-lived requests |
| Updates stop in a hidden tab | Expected: reopening the tab reconnects; all disconnected clients allow sampling to stop |
| Service fails on another checkout | Replace the unit's absolute source path and verify the configured Python executable |

## Known limitations

- Designed for GB10 unified memory and its CPU layout. The CPU cluster labels
  are based on a 3500 MHz threshold, not general CPU-model detection.
- Only GPU index 0 is monitored; this is not a multi-GPU or discrete-VRAM dashboard.
- Disk activity uses the first available device among `nvme0n1`, `sda`, and `vda`;
  filesystem usage is for `/` only. Virtual network interfaces are omitted.
- Sensors, per-process visibility, GPU memory reporting, and power limits depend
  on the kernel, driver, and user permissions. Missing readings may be null.
- GPU process memory is an estimate within shared memory, not an additional RAM
  pool. Summed process allocations may not equal unique physical memory use.
- NVML process calls use fixed arrays of 64 entries; unsupported functions and
  collector exceptions are not comprehensively recovered. A collector error can
  stop the sampler thread and require restarting the process.
- The Python HTTP server provides no authentication, TLS, rate limiting, or
  production hardening. Process arguments can contain sensitive information.
- Sparklines hold 120 samples: approximately two minutes at the default interval;
  history disappears on reload. There is no alerting, export, or historical database.
- vLLM figures are read from Prometheus series names used by current vLLM
  releases (`vllm:kv_cache_usage_perc`, or `vllm:gpu_cache_usage_perc` on older
  ones); renamed series show as blank rather than as errors.
- Low overhead is a design goal, not a portable benchmark guarantee. Inference-server
  timeouts, number of processes, browser clients, and hardware affect sampling cost.

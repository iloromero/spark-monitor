#!/usr/bin/env python3
"""Spark Monitor: low-overhead live resource dashboard for an NVIDIA DGX Spark (GB10).

Standard library only. GPU metrics come from NVML via ctypes (no nvidia-smi
subprocess); everything else is read from /proc and /sys. Sampling runs only
while at least one browser is connected, and results stream over SSE.
"""
import argparse
import ctypes
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


# ---------------------------------------------------------------- GPU (NVML)
class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Proc(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


NVML_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF


class GPU:
    def __init__(self, power_max):
        self.ok = False
        self.power_max = power_max
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            if self.lib.nvmlInit_v2() != 0:
                return
            self.h = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h)) != 0:
                return
            name = ctypes.create_string_buffer(96)
            self.lib.nvmlDeviceGetName(self.h, name, 96)
            self.name = name.value.decode()
            self.max_sm = self._uint("nvmlDeviceGetMaxClockInfo", 1)
            self.ok = True
        except OSError:
            pass

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        rc = getattr(self.lib, fn)(self.h, *args, ctypes.byref(v))
        return v.value if rc == 0 else None

    def _procs(self, fn):
        n = ctypes.c_uint(64)
        arr = (_Proc * 64)()
        if getattr(self.lib, fn)(self.h, ctypes.byref(n), arr) != 0:
            return []
        return [(arr[i].pid, None if arr[i].usedGpuMemory == NVML_NOT_AVAILABLE
                 else arr[i].usedGpuMemory) for i in range(n.value)]

    def sample(self):
        if not self.ok:
            return None
        u = _Util()
        util = u.gpu if self.lib.nvmlDeviceGetUtilizationRates(self.h, ctypes.byref(u)) == 0 else None
        power = self._uint("nvmlDeviceGetPowerUsage")
        limit = self._uint("nvmlDeviceGetEnforcedPowerLimit")  # not reported on GB10
        pstate = self._uint("nvmlDeviceGetPerformanceState")
        procs = {}  # pid -> (gpu_mem, kind): C = compute, G = graphics
        for fn, kind in (("nvmlDeviceGetComputeRunningProcesses_v3", "C"),
                         ("nvmlDeviceGetGraphicsRunningProcesses_v3", "G")):
            for pid, mem in self._procs(fn):
                prev = procs.get(pid)
                procs[pid] = (mem, kind) if not prev else (prev[0] or mem, prev[1] + "+" + kind)
        return {
            "name": self.name,
            "util": util,
            "temp": self._uint("nvmlDeviceGetTemperature", 0),
            "power_w": None if power is None else power / 1000,
            "power_max_w": limit / 1000 if limit else self.power_max,
            "sm_mhz": self._uint("nvmlDeviceGetClockInfo", 1),
            "sm_max_mhz": self.max_sm,
            "pstate": None if pstate is None or pstate == 32 else f"P{pstate}",
            "procs": [dict(_proc_info(pid, mem), kind=kind) for pid, (mem, kind) in procs.items()],
        }


def _read(path, default=None):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def _proc_info(pid, gpu_mem):
    status = _read(f"/proc/{pid}/status", "")
    rss = 0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
            break
    cmd = (_read(f"/proc/{pid}/cmdline", "") or "").replace("\0", " ").strip()
    name = (_read(f"/proc/{pid}/comm", "") or "?").strip()
    return {"pid": pid, "name": name, "cmd": cmd[:160], "rss": rss, "gpu_mem": gpu_mem}


# ---------------------------------------------------------------- CPU / system
def _cpu_layout():
    """Group CPUs into clusters by max frequency (X925 performance vs A725 efficiency)."""
    cpus = []
    for d in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        mx = _read(d / "cpufreq/cpuinfo_max_freq")
        cpus.append((int(d.name[3:]), int(mx) // 1000 if mx else 0))
    return cpus


class System:
    def __init__(self):
        self.cpus = _cpu_layout()
        self.prev_cpu = None
        self.prev_disk = None
        self.prev_net = None
        self.prev_t = None
        self.disk = self._pick_disk()
        self.nics = [p.name for p in Path("/sys/class/net").iterdir() if (p / "device").exists()]
        self.temps = self._find_temps()
        self.proc_prev = {}
        self.proc_t = 0
        self.top = []

    def _pick_disk(self):
        for name in ("nvme0n1", "sda", "vda"):
            if Path(f"/sys/block/{name}").exists():
                return name
        return None

    def _find_temps(self):
        out = []
        zones = sorted(Path("/sys/class/thermal").glob("thermal_zone*"))
        if zones:
            out.append(("SoC (max ACPI)", [z / "temp" for z in zones]))
        for h in Path("/sys/class/hwmon").glob("hwmon*"):
            name = (_read(h / "name", "") or "").strip()
            if name == "nvme":
                out.append(("NVMe SSD", [h / "temp1_input"]))
            elif name == "mlx5" and not any(n == "ConnectX NIC" for n, _ in out):
                out.append(("ConnectX NIC", [h / "temp1_input"]))
        return out

    def _cpu(self):
        stats = {}
        for line in _read("/proc/stat", "").splitlines():
            if not line.startswith("cpu"):
                break
            f = line.split()
            vals = list(map(int, f[1:9]))
            stats[f[0]] = (sum(vals), vals[3] + vals[4], vals[4])  # total, idle+iowait, iowait
        out = {"total": None, "per": [], "iowait": None}
        if self.prev_cpu:
            def pct(k):
                t0, i0, _ = self.prev_cpu.get(k, (0, 0, 0))
                t1, i1, _ = stats[k]
                dt = t1 - t0
                return 0.0 if dt <= 0 else 100.0 * (1 - (i1 - i0) / dt)
            out["total"] = pct("cpu")
            t0, _, w0 = self.prev_cpu["cpu"]
            t1, _, w1 = stats["cpu"]
            out["iowait"] = 0.0 if t1 <= t0 else 100.0 * (w1 - w0) / (t1 - t0)
            for cpu, mx in self.cpus:
                k = f"cpu{cpu}"
                if k in stats:
                    cur = _read(f"/sys/devices/system/cpu/{k}/cpufreq/scaling_cur_freq")
                    out["per"].append({"id": cpu, "util": pct(k), "mhz": int(cur) // 1000 if cur else None, "max_mhz": mx})
        self.prev_cpu = stats
        return out

    def _mem(self):
        m = {}
        for line in _read("/proc/meminfo", "").splitlines():
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0]) * 1024
        total, avail = m["MemTotal"], m["MemAvailable"]
        cache = m.get("Cached", 0) + m.get("SReclaimable", 0)
        buffers, free = m.get("Buffers", 0), m["MemFree"]
        return {
            "total": total, "used": total - avail, "avail": avail,
            # breakdown that sums exactly to total, for the stacked bar
            "resident": max(0, total - free - buffers - cache), "cache": cache, "buffers": buffers, "free": free,
            "swap_total": m.get("SwapTotal", 0), "swap_used": m.get("SwapTotal", 0) - m.get("SwapFree", 0),
        }

    def _disk_net(self, dt):
        disk = None
        if self.disk:
            for line in _read("/proc/diskstats", "").splitlines():
                f = line.split()
                if f[2] == self.disk:
                    cur = (int(f[5]), int(f[9]), int(f[12]))
                    if self.prev_disk and dt:
                        r, w, busy = (c - p for c, p in zip(cur, self.prev_disk))
                        disk = {"read_bps": r * 512 / dt, "write_bps": w * 512 / dt,
                                "util": min(100.0, busy / (dt * 10))}
                    self.prev_disk = cur
                    break
        st = os.statvfs("/")
        fs = {"total": st.f_blocks * st.f_frsize, "used": (st.f_blocks - st.f_bfree) * st.f_frsize}

        net_cur = {}
        for line in _read("/proc/net/dev", "").splitlines()[2:]:
            name, rest = line.split(":", 1)
            name = name.strip()
            if name in self.nics:
                f = rest.split()
                net_cur[name] = (int(f[0]), int(f[8]))
        net = []
        if self.prev_net and dt:
            for name, (rx, tx) in net_cur.items():
                if name in self.prev_net and (_read(f"/sys/class/net/{name}/operstate", "") or "").strip() == "up":
                    prx, ptx = self.prev_net[name]
                    try:
                        speed = int(_read(f"/sys/class/net/{name}/speed", "-1"))
                    except ValueError:
                        speed = -1
                    net.append({"name": name, "rx_bps": (rx - prx) / dt, "tx_bps": (tx - ptx) / dt,
                                "link_mbps": speed if speed > 0 else None})
        self.prev_net = net_cur
        return disk, fs, net

    def _temps(self):
        out = []
        for label, paths in self.temps:
            vals = [int(v) / 1000 for v in (_read(p) for p in paths) if v]
            if vals:
                out.append({"label": label, "c": max(vals)})
        return out

    def _top_procs(self, now):
        """Top CPU consumers; scanning /proc is the costliest step, so it runs every 3 s."""
        if now - self.proc_t < 3:
            return self.top
        cur = {}
        for d in os.scandir("/proc"):
            if not d.name.isdigit():
                continue
            s = _read(f"/proc/{d.name}/stat")
            if not s:
                continue
            r = s.rfind(")")
            f = s[r + 2:].split()
            cur[int(d.name)] = (s[s.find("(") + 1:r], int(f[11]) + int(f[12]), int(f[21]) * PAGE_SIZE)
        dt = now - self.proc_t
        if self.proc_prev and dt:
            rows = []
            for pid, (name, ticks, rss) in cur.items():
                prev = self.proc_prev.get(pid)
                if prev:
                    pct = (ticks - prev[1]) / CLK_TCK / dt * 100
                    rows.append({"pid": pid, "name": name, "cpu": pct, "rss": rss})
            rows.sort(key=lambda r: r["cpu"], reverse=True)
            self.top = rows[:8]
            for r in self.top:  # kernel threads have no cmdline; show [name] like ps does
                cmd = (_read(f"/proc/{r['pid']}/cmdline", "") or "").replace("\0", " ").strip()
                r["cmd"] = cmd[:200] or f"[{r['name']}]"
        self.proc_prev = cur
        self.proc_t = now
        return self.top

    def sample(self):
        now = time.monotonic()
        dt = now - self.prev_t if self.prev_t else None
        self.prev_t = now
        disk, fs, net = self._disk_net(dt)
        load = (_read("/proc/loadavg", "0 0 0") or "0 0 0").split()[:3]
        return {
            "cpu": self._cpu(), "mem": self._mem(), "disk": disk, "fs": fs, "net": net,
            "temps": self._temps(), "load": [float(x) for x in load],
            "uptime": float((_read("/proc/uptime", "0") or "0").split()[0]),
            "top": self._top_procs(now), "ncpu": len(self.cpus),
        }


# ---------------------------------------------------------------- Ollama
class Ollama:
    """Models currently loaded in Ollama, from /api/ps; polled every 3 s."""

    def __init__(self, url):
        self.url = url.rstrip("/") + "/api/ps"
        self.t = 0
        self.last = None

    def sample(self, now):
        if now - self.t < 3:
            return self.last
        self.t = now
        try:
            with urllib.request.urlopen(self.url, timeout=0.5) as r:
                models = json.load(r).get("models", [])
        except (OSError, ValueError):
            self.last = {"up": False, "models": []}
            return self.last
        self.last = {"up": True, "models": [{
            "name": m.get("name"),
            "size": m.get("size", 0),
            "size_vram": m.get("size_vram", 0),
            "params": m.get("details", {}).get("parameter_size"),
            "quant": m.get("details", {}).get("quantization_level"),
            "ctx": m.get("context_length"),
            "expires": m.get("expires_at"),
        } for m in models]}
        return self.last


# ---------------------------------------------------------------- sampler + HTTP
class Sampler:
    def __init__(self, interval, ollama_url, power_max):
        self.interval = interval
        self.gpu = GPU(power_max)
        self.sys = System()
        self.ollama = Ollama(ollama_url)
        self.cond = threading.Condition()
        self.latest = None
        self.seq = 0
        self.clients = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            with self.cond:
                while self.clients == 0:
                    self.cond.wait()
            t0 = time.monotonic()
            data = self.sys.sample()
            data["gpu"] = self.gpu.sample()
            data["ollama"] = self.ollama.sample(t0)
            data["ts"] = time.time()
            data["sample_ms"] = (time.monotonic() - t0) * 1000
            with self.cond:
                self.latest = json.dumps(data, separators=(",", ":"))
                self.seq += 1
                self.cond.notify_all()
            time.sleep(max(0.05, self.interval - (time.monotonic() - t0)))


class Handler(BaseHTTPRequestHandler):
    sampler: Sampler = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self._stream()
        else:
            self.send_error(404)

    def _stream(self):
        s = self.sampler
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with s.cond:
            s.clients += 1
            s.cond.notify_all()
        seen = -1
        try:
            while True:
                with s.cond:
                    s.cond.wait_for(lambda: s.seq != seen, timeout=15)
                    seen, payload = s.seq, s.latest
                self.wfile.write(f"data: {payload}\n\n".encode() if payload else b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with s.cond:
                s.clients -= 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama base URL")
    ap.add_argument("--power-max", type=float, default=140.0,
                    help="watts shown as 100%% on the power dial when NVML reports no limit (GB10 TDP: 140)")
    a = ap.parse_args()
    Handler.sampler = Sampler(a.interval, a.ollama, a.power_max)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Spark Monitor on http://{a.host}:{a.port}  (GPU via NVML: {Handler.sampler.gpu.ok})")
    srv.serve_forever()


if __name__ == "__main__":
    main()

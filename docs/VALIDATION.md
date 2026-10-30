# Publication validation

Validated on **October 29, 2026** using the deployment's Python 3.12.3 on Linux.
The application source and original service unit were preserved. No existing
unit was reloaded or restarted; no ports, databases, proxy configuration,
firewall rules, timers, or cron jobs were changed.

| Check | Result |
| --- | --- |
| Python AST parsing and compilation | Passed; compiled in memory without writing application bytecode |
| CLI `--help` | Passed using `/usr/bin/python3` |
| Linux collectors | Passed: successive samples yielded CPU deltas, memory, root filesystem, CPU count, and serializable JSON |
| NVIDIA NVML | Passed: real GPU sample returned successfully on this host |
| Missing NVML library | Passed: simulated missing library returns no GPU sample |
| Ollama `/api/ps` | Passed: existing local service was reachable; no model was loaded, unloaded, or modified for the check |
| Unavailable Ollama | Passed: simulated connection failure yields `up: false` and no models |
| Isolated HTTP server | Passed: OS-assigned localhost test port; `/` and `/index.html` return the exact frontend bytes; unknown route returns 404 |
| SSE | Passed: correct content type and successive parseable JSON samples; CPU delta and sampling cost present |
| Idle behavior | Passed: final test client disconnects and sampler sequence stops advancing |
| Original systemd unit | Passed `systemd-analyze --user verify` |
| Existing HTTP | Passed: port 8888 serves the exact local frontend |
| Existing HTTPS | Passed: port 18888 serves the exact local frontend; certificate checked against the existing private LAN CA |
| Staged-file secret scan | Passed Gitleaks with redacted output; no leaks found |
| Whitespace and file review | Passed `git diff --cached --check`; exact nine-file allowlist; screenshot contains only PNG image chunks |
| Browser rendering | Passed visual inspection of Firefox headless screenshot using entirely synthetic data and an isolated browser profile |
| Production service continuity | Same active service process before and after checks; existing HTTP/HTTPS listeners retained |

## Scope and limitations

There is no existing test suite, configured linter, or build pipeline. Python
syntax checks, targeted collector/HTTP assertions, systemd validation, and
browser rendering were used instead. No packaging or frontend build is needed.
The temporary validation harness, browser profile, and synthetic HTML fixture
were kept outside the repository. Only the reviewed synthetic screenshot is
included.

NVML, Linux hardware interfaces, Ollama, systemd, and the HTTPS proxy are
host-dependent integration checks. They were available on this host; another
machine without them cannot reproduce those integration results as-is. The
checks do not prove multi-GPU support, compatibility with every NVML version,
model-load correctness, performance under load, or production security hardening.
No fresh overhead benchmark was performed during publication.

## Publication hygiene

Only reviewed application source, documentation, placeholder examples, the
original generic service unit, and a fabricated-data screenshot are intended
for Git. The local RTX 5090 reference prompt and Python cache are excluded.
No live metrics, process dumps, model inventory, host inventory, credentials,
TLS material, logs, databases, dependencies, or browser authentication state
are included. Configuration examples require explicit path/hostname replacement.

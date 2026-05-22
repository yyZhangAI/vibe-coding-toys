# GPU Monitor

Real-time remote GPU monitoring dashboard. SSHes into GPU servers, polls `nvidia-smi`, and shows live GPU utilization, memory, temperature, and running processes in a browser dashboard.

Built with [Claude Code](https://claude.ai/code) powered by **DeepSeek-v4-pro**.

## Features

- Multi-server support — monitor any number of GPU servers simultaneously
- Reads `~/.ssh/config` — no need to duplicate SSH settings
- Per-server GPU cards with utilization, memory, and temperature
- Running process table with user, PID, and GPU assignment
- Manual refresh button + auto-refresh every 2 seconds
- Mock mode for local UI testing without real servers

## Quick Start

```bash
# Install dependencies
uv sync

# Test locally with mock data
uv run python server.py --mock
# Open http://127.0.0.1:8000

# Connect to real servers
cp config.yaml.example config.yaml
# Add your server names (must match Host entries in ~/.ssh/config)
uv run python server.py
```

## How It Works

```
Browser  <--WebSocket-->  server.py  <--SSH/paramiko-->  nvidia-smi on remote GPU servers
```

One SSH exec per poll per server — all `nvidia-smi` queries and `ps` calls are batched into a single shell script to minimize latency.

## Config

`config.yaml` only needs server names:

```yaml
servers:
  - name: A100
  - name: monkey
```

Host, port, user, and key file are read from `~/.ssh/config` automatically.

import time
import os
import yaml
import paramiko
from pathlib import Path


def _parse_ssh_config():
    """Parse ~/.ssh/config into a dict of hostname -> {key: value}."""
    ssh_config_path = Path.home() / ".ssh" / "config"
    if not ssh_config_path.exists():
        return {}
    hosts = {}
    current_host = None
    with open(ssh_config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].lower()
            value = " ".join(parts[1:])
            if key == "host":
                current_host = value
                hosts[current_host] = {}
            elif current_host and key in (
                "hostname", "port", "user", "identityfile",
                "proxycommand", "proxyjump",
            ):
                hosts[current_host][key] = value
    return hosts


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    servers = cfg.get("servers", [cfg])
    ssh_config = _parse_ssh_config()
    for s in servers:
        name = s.get("name", "")
        host_entry = ssh_config.get(name, {})
        s.setdefault("host", host_entry.get("hostname", name))
        s.setdefault("port", int(host_entry.get("port", 22)))
        s.setdefault("user", host_entry.get("user", os.environ.get("USER", "root")))
        if "key_file" not in s:
            idf = host_entry.get("identityfile", "")
            s["key_file"] = os.path.expanduser(idf) if idf else None
    return servers


class GPUCollector:
    def __init__(self, name, host, port, user, key_file):
        self.name = name
        self.host = host
        self.port = port
        self.user = user
        self.key_file = key_file
        self._client = None
        self._connected = False

    def connect(self):
        try:
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            connect_kwargs = dict(
                hostname=self.host,
                port=self.port,
                username=self.user,
                timeout=10,
                banner_timeout=10,
            )
            if self.key_file and os.path.isfile(self.key_file):
                connect_kwargs["key_filename"] = self.key_file
            self._client.connect(**connect_kwargs)
            self._connected = True
            return True
        except Exception as e:
            self._connected = False
            print(f"[{self.name}] SSH connection failed: {e}")
            return False

    def disconnect(self):
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._connected = False

    @property
    def connected(self):
        return self._connected

    def _exec(self, cmd):
        if not self._client:
            return ""
        try:
            transport = self._client.get_transport()
            if transport is None or not transport.is_active():
                self._connected = False
                return ""
            _, stdout, stderr = self._client.exec_command(cmd, timeout=10)
            return stdout.read().decode("utf-8", errors="replace")
        except Exception:
            self._connected = False
            return ""

    def poll(self):
        if not self._connected:
            if not self.connect():
                return {"server": self.name, "error": "SSH disconnected", "gpus": [], "processes": []}

        return self._poll_combined()

    def _poll_combined(self):
        # One SSH exec: all nvidia-smi data + batch ps, delimited by markers
        script = (
            "echo '###GPU'; "
            "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu "
            "--format=csv,noheader,nounits 2>/dev/null; "
            "echo '###PROC'; "
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_bus_id "
            "--format=csv,noheader,nounits 2>/dev/null; "
            "echo '###BUS'; "
            "nvidia-smi --query-gpu=index,gpu_bus_id "
            "--format=csv,noheader 2>/dev/null; "
            "echo '###PS'; "
            "pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr '\\n' ',' | sed 's/,$//'); "
            "[ -n \"$pids\" ] && ps -o user=,pid= -p \"$pids\" 2>/dev/null"
        )
        raw = self._exec(script)

        gpu_lines, proc_lines, bus_lines, ps_lines = [], [], [], []
        current = None
        for line in raw.strip().splitlines():
            s = line.strip()
            if s == "###GPU":
                current = gpu_lines
            elif s == "###PROC":
                current = proc_lines
            elif s == "###BUS":
                current = bus_lines
            elif s == "###PS":
                current = ps_lines
            elif s and current is not None:
                current.append(s)

        # Bus ID -> GPU index
        bus_to_index = {}
        for line in bus_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                bus_to_index[parts[1]] = int(parts[0])

        gpus = []
        for line in gpu_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "utilization": int(parts[2]),
                "memory_used": int(parts[3]),
                "memory_total": int(parts[4]),
                "temperature": int(parts[5]),
                "server": self.name,
            })

        # PID -> user (batch ps output: "user pid" per line)
        pid_to_user = {}
        for line in ps_lines:
            parts = line.split()
            if len(parts) >= 2:
                pid_to_user[int(parts[-1])] = parts[0]

        processes = []
        for line in proc_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            pid = int(parts[0])
            processes.append({
                "pid": pid,
                "name": parts[1],
                "memory_used": int(parts[2]),
                "gpu": bus_to_index.get(parts[3], -1),
                "user": pid_to_user.get(pid, "?"),
                "server": self.name,
            })

        return {
            "server": self.name,
            "gpus": gpus,
            "processes": processes,
        }


class MockGPUCollector:
    def __init__(self, name):
        self.name = name
        self.connected = True

    def connect(self):
        return True

    def disconnect(self):
        pass

    def poll(self):
        import random, time
        gpu_count = random.randint(2, 8)
        gpus = []
        for i in range(gpu_count):
            mem_used = random.randint(4000, 24000)
            gpus.append({
                "index": i,
                "name": "NVIDIA GeForce RTX 4090",
                "utilization": random.randint(0, 100),
                "memory_used": mem_used,
                "memory_total": 24576,
                "temperature": random.randint(35, 85),
                "server": self.name,
            })
        users = ["alice", "bob", "charlie", "dave"]
        tasks = ["python train.py", "python eval.py", "torchrun ddp.py", "jupyter-lab"]
        processes = []
        for _ in range(random.randint(0, 4)):
            processes.append({
                "pid": random.randint(1000, 99999),
                "user": random.choice(users),
                "name": random.choice(tasks),
                "gpu": random.randint(0, gpu_count - 1),
                "memory_used": random.randint(2000, 12000),
                "server": self.name,
            })
        return {"server": self.name, "gpus": gpus, "processes": processes}

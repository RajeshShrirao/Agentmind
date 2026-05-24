import subprocess
import os
import re
from stats_logger import GLOBAL as log

PAGE_SIZE = 16384

def _vm_stat():
    r = subprocess.run(['vm_stat'], capture_output=True, text=True)
    stats = {}
    for line in r.stdout.strip().split('\n'):
        if ':' not in line:
            continue
        key, val = line.split(':', 1)
        key = key.strip().strip('.').lower()
        val = val.strip().strip('.')
        try:
            stats[key] = int(val.replace('.', ''))
        except ValueError:
            pass
    return stats

def _sysctl_str(key):
    r = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True)
    return r.stdout.strip()

def _parse_swap_gb(swap_str: str) -> float:
    m = re.search(r'used\s*=\s*([\d.]+)([MG])', swap_str)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2)
    return val / 1e3 if unit == 'M' else val

def _hw_metrics():
    total_ram = int(_sysctl_str('hw.memsize'))
    stats = _vm_stat()
    pages_free = stats.get('pages free', 0)
    pages_active = stats.get('pages active', 0)
    pages_wired = stats.get('pages wired down', 0)
    pages_compressed = stats.get('pages stored in compressor', 0)

    used = (pages_active + pages_wired + pages_compressed) * PAGE_SIZE
    free = pages_free * PAGE_SIZE
    used_pct = used / total_ram * 100 if total_ram else 0

    load_1m = os.getloadavg()[0]
    cpu_count = os.cpu_count() or 1
    cpu_pct = (load_1m / cpu_count) * 100

    swap_r = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True)
    swap_str = swap_r.stdout.strip()

    return {
        "ram_used_gb": used / 1e9,
        "ram_total_gb": total_ram / 1e9,
        "ram_pct": used_pct,
        "cpu_pct": cpu_pct,
        "swap": swap_str,
        "swap_used_gb": _parse_swap_gb(swap_str),
        "pages_compressed": pages_compressed,
    }

def hw_summary():
    m = _hw_metrics()
    return (f"CPU:{m['cpu_pct']:.0f}% "
            f"RAM:{m['ram_pct']:.0f}% ({m['ram_used_gb']:.1f}/{m['ram_total_gb']:.1f}GB) "
            f"swap:[{m['swap']}]")

def get_hw_metrics():
    return _hw_metrics()

def print_hw(label=""):
    m = _hw_metrics()
    prefix = f"[hw] {label}  " if label else "[hw] "
    print(f"{prefix}{hw_summary()}")
    log.hw(label, m["cpu_pct"], m["ram_pct"], m["ram_used_gb"],
           m["ram_total_gb"], m["swap"])


class ResourceScheduler:
    """Monitor system resources and cap seq_len to avoid swap thrashing.

    Key insight: macOS aggressively uses available RAM for file cache.
    Total RAM usage will always look high (~12-14GB baseline). Training
    only adds ~2GB. The real danger signal is *sustained swap growth*
    beyond the baseline — not total RAM percentage.

    Behaves like TCP congestion control:
    - Records baseline swap + RAM at startup
    - Multiplicative decrease on sustained swap *delta* exceeding threshold
    - Additive increase (up to curriculum target) when clear for N checks
    """

    def __init__(self,
                 swap_delta_threshold_gb: float = 0.1,
                 decrease_factor: float = 0.75,
                 increase_factor: float = 1.1,
                 decrease_checks: int = 3,
                 increase_checks: int = 10,
                 min_seq_len: int = 64):
        self.swap_delta_threshold_gb = swap_delta_threshold_gb
        self.decrease_factor = decrease_factor
        self.increase_factor = increase_factor
        self.decrease_checks = decrease_checks
        self.increase_checks = increase_checks
        self.min_seq_len = min_seq_len
        self._baseline_ram_gb = None
        self._baseline_swap_gb = None
        self._swap_streak = 0
        self._clean_streak = 0
        self._current_cap = None

    def record_baseline(self):
        m = _hw_metrics()
        self._baseline_ram_gb = m["ram_used_gb"]
        self._baseline_swap_gb = m.get("swap_used_gb", 0.0)
        self._swap_streak = 0
        self._clean_streak = 0
        self._current_cap = None
        print(f"  [RESOURCE] Baseline: RAM {self._baseline_ram_gb:.1f}GB | "
              f"swap {self._baseline_swap_gb:.2f}GB")

    def get_max_seq_len(self, requested_seq_len: int) -> int:
        if self._current_cap is None:
            self._current_cap = requested_seq_len

        m = _hw_metrics()
        swap_now = m.get("swap_used_gb", 0.0)
        swap_delta = swap_now - (self._baseline_swap_gb or 0)
        ram_delta = m["ram_used_gb"] - (self._baseline_ram_gb or 0)

        if swap_delta > self.swap_delta_threshold_gb:
            self._swap_streak += 1
            self._clean_streak = 0
            if self._swap_streak >= self.decrease_checks:
                new_cap = max(self.min_seq_len,
                              int(self._current_cap * self.decrease_factor))
                if new_cap < self._current_cap:
                    print(f"  [RESOURCE] swap Δ+{swap_delta:.2f}GB | "
                          f"Δmem {ram_delta:+.1f}GB | "
                          f"cap {self._current_cap}->{new_cap}")
                self._current_cap = new_cap
        else:
            self._swap_streak = 0
            self._clean_streak += 1
            if (self._clean_streak >= self.increase_checks
                    and self._current_cap < requested_seq_len):
                new_cap = min(requested_seq_len,
                              int(self._current_cap * self.increase_factor))
                if new_cap > self._current_cap:
                    print(f"  [RESOURCE] swap stable for {self._clean_streak} checks "
                          f"| raising cap {self._current_cap}->{new_cap}")
                self._current_cap = new_cap

        return min(requested_seq_len, self._current_cap)

    @property
    def active_cap(self):
        return self._current_cap

    def report(self) -> str:
        m = _hw_metrics()
        swap_delta = m.get("swap_used_gb", 0.0) - (self._baseline_swap_gb or 0)
        return (f"RAM={m['ram_used_gb']:.1f}GB Δ={m['ram_used_gb']-(self._baseline_ram_gb or 0):+.1f}GB "
                f"swap Δ={swap_delta:+.2f}GB cap={self._current_cap or 'N/A'}")

import subprocess
import os
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

    # CPU load (1min average)
    load_1m = os.getloadavg()[0]
    cpu_count = os.cpu_count() or 1
    cpu_pct = (load_1m / cpu_count) * 100

    # Swap
    swap_r = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True)
    swap_str = swap_r.stdout.strip()

    return {
        "ram_used_gb": used / 1e9,
        "ram_total_gb": total_ram / 1e9,
        "ram_pct": used_pct,
        "cpu_pct": cpu_pct,
        "swap": swap_str,
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

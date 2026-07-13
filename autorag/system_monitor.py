from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil
import pynvml


@dataclass
class Sample:
    timestamp: float
    cpu_percent: float
    ram_used_mb: float
    gpu_util_percent: float | None = None
    gpu_mem_used_mb: float | None = None


@dataclass
class MonitorSummary:
    duration_s: float
    samples: list[Sample] = field(default_factory=list)
    
    def _avg(self, attr: str) -> float | None:
        values = [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]
        return sum(values) / len(values) if values else None
    
    def _max(self, attr: str) -> float | None:
        values = [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]
        return max(values) if values else None

    def as_dict(self) -> dict:
        return {
            "duration_s": self.duration_s,
            "cpu_percent_avg": self._avg("cpu_percent"),
            "cpu_percent_max": self._max("cpu_percent"),
            "ram_used_mb_avg": self._avg("ram_used_mb"),
            "ram_used_mb_max": self._max("ram_used_mb"),
            "gpu_util_percent_avg": self._avg("gpu_util_percent"),
            "gpu_mem_used_mb_avg": self._avg("gpu_mem_used_mb"),
            "num_samples": len(self.samples),
        }


class SystemMonitor:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples: list[Sample] = []
        self._start_time: float | None = None
        pynvml.nvmlInit()
        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
    def _sample_once(self) -> Sample:
        cpu_percent = psutil.cpu_percent(interval=None)
        vmem = psutil.virtual_memory()
        
        util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
        gpu_util = float(util.gpu)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        gpu_mem = mem.used / (1024 ** 2)
        
        return Sample(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            ram_used_mb=vmem.used / (1024 ** 2),
            gpu_util_percent=gpu_util,
            gpu_mem_used_mb=gpu_mem
        )
    
    def _run(self):
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            self._samples.append(self._sample_once())
            self._stop_event.wait(self.interval_s)
    
    def start(self):
        self._samples = []
        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def stop(self) -> MonitorSummary:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2 + 1)
        duration = time.time() - (self._start_time or time.time())
        return MonitorSummary(duration_s=duration, samples=self._samples)
    
    def shutdown(self):
        if self._nvml_handle is not None:
            pynvml.nvmlShutdown()
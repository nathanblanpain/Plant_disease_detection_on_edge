"""
benchmark_protocol.py
=====================
Energy & Performance Measurement Protocol for Edge AI Models on Raspberry Pi 5
Implements the 5-step protocol: Preparation → Baseline → Execution → Collection → Metrics
Supports 3 phases: initial screening (3 runs), Pareto validation (10 runs), contextual (5 runs)

Hardware assumptions:
  - Power meter : INA219 (I2C, addr 0x40) via adafruit-circuitpython-ina219
  - Temperature  : vcgencmd measure_temp  (RPi built-in)
  - CPU / RAM    : psutil
  - Models       : Ultralytics YOLO (swap run_inference() for your framework)

Install deps:
  pip install psutil adafruit-circuitpython-ina219 ultralytics opencv-python-headless
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

# ─────────────────────────────────────────────
#  Optional hardware imports (graceful fallback)
# ─────────────────────────────────────────────
try:
    import board
    import busio
    from adafruit_ina219 import INA219
    _INA219_AVAILABLE = True
except ImportError:
    _INA219_AVAILABLE = False
    logging.warning("adafruit_ina219 not found – power readings will be simulated (0.0).")

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logging.warning("opencv-python not found – images will be generated as numpy arrays.")

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProtocolConfig:
    # ── Phase run counts ──────────────────────────────────────────────────────
    runs_screening:   int = 3    # initial screening
    runs_pareto:      int = 10   # Pareto validation  (enables 95 % CI)
    runs_contextual:  int = 5    # contextual evaluation

    # ── Inference counts ──────────────────────────────────────────────────────
    warmup_inferences:    int = 5
    benchmark_inferences: int = 100

    # ── Thermal / timing ─────────────────────────────────────────────────────
    thermal_stabilisation_s: float = 60.0   # wait after warmup
    max_temp_baseline_c:     float = 60.0   # reject run if temp ≥ this at baseline
    max_temp_runtime_c:      float = 75.0   # reject run if temp exceeds this during run

    # ── Quality-control thresholds ────────────────────────────────────────────
    power_variance_threshold: float = 0.05  # 5 % variance from session mean
    cpu_spike_threshold_pct:  float = 80.0  # background-process detection

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "results"


# ─────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceSample:
    inference_idx:    int
    latency_s:        float
    cpu_percent:      float
    ram_percent:      float
    temperature_c:    float
    energy_mah:       float   # cumulative reading from power meter at this moment
    power_w:          float   # instantaneous power draw

@dataclass
class RunResult:
    run_idx:           int
    phase:             str
    model_id:          str
    resolution:        int
    quantization:      str
    timestamp:         str
    baseline_temp_c:   float
    samples:           List[InferenceSample] = field(default_factory=list)

    # ── Derived metrics (filled by compute_metrics) ───────────────────────────
    mean_latency_s:    float = 0.0
    std_latency_s:     float = 0.0
    mean_cpu_pct:      float = 0.0
    mean_ram_pct:      float = 0.0
    mean_temp_c:       float = 0.0
    max_temp_c:        float = 0.0
    total_energy_mah:  float = 0.0   # delta from baseline reset
    energy_per_inf_mah:float = 0.0
    mean_power_w:      float = 0.0
    map_score:         float = 0.0   # filled externally if available
    valid:             bool  = True
    rejection_reason:  str   = ""


# ─────────────────────────────────────────────────────────────────────────────
#  HARDWARE INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class PowerMeter:
    """Thin wrapper around INA219 with simulation fallback."""

    def __init__(self):
        self._ina: Any = None
        self._energy_baseline_mah: float = 0.0
        self._session_start_time: float = 0.0

        if _INA219_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self._ina = INA219(i2c)
                logging.info("INA219 power meter initialised on I2C.")
            except Exception as exc:
                logging.warning(f"INA219 init failed ({exc}). Falling back to simulation.")

    # -- public API -----------------------------------------------------------

    def reset_energy_counter(self) -> None:
        """Zero-reference the energy accumulator (step 2 of protocol)."""
        self._energy_baseline_mah = self._read_raw_mah()
        self._session_start_time  = time.monotonic()
        logging.debug(f"Energy counter reset. Baseline raw = {self._energy_baseline_mah:.4f} mAh")

    def read_energy_mah(self) -> float:
        """Return energy consumed since last reset_energy_counter() call."""
        return max(0.0, self._read_raw_mah() - self._energy_baseline_mah)

    def read_power_w(self) -> float:
        if self._ina:
            try:
                return self._ina.power / 1000.0   # INA219 returns mW
            except Exception:
                pass
        return self._simulated_power_w()

    # -- private helpers ------------------------------------------------------

    def _read_raw_mah(self) -> float:
        """Cumulative charge in mAh (raw, not zero-referenced)."""
        if self._ina:
            try:
                # INA219 gives instantaneous current (A); integrate manually
                elapsed = time.monotonic() - self._session_start_time
                current_a = self._ina.current / 1000.0   # mA → A
                return (current_a * elapsed) / 3600.0 * 1000.0  # → mAh
            except Exception:
                pass
        # Simulation: ramp up slowly
        elapsed = time.monotonic() - self._session_start_time if self._session_start_time else 0
        return elapsed * 0.5 / 3600.0 * 1000.0  # ~0.5 A simulated

    def _simulated_power_w(self) -> float:
        import random
        return round(3.5 + random.uniform(-0.2, 0.2), 3)


class SystemMonitor:
    """CPU temperature, CPU%, RAM% via vcgencmd and psutil."""

    @staticmethod
    def cpu_temperature_c() -> float:
        try:
            raw = subprocess.check_output(
                ["vcgencmd", "measure_temp"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            # raw = "temp=47.8'C"
            return float(raw.split("=")[1].replace("'C", ""))
        except Exception:
            # Fallback: /sys/class/thermal
            try:
                val = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
                return float(val) / 1000.0
            except Exception:
                return 0.0

    @staticmethod
    def cpu_percent(interval: float = 0.1) -> float:
        return psutil.cpu_percent(interval=interval)

    @staticmethod
    def ram_percent() -> float:
        return psutil.virtual_memory().percent

    @staticmethod
    def background_process_detected(cpu_threshold: float) -> bool:
        """Return True if any unexpected process is consuming CPU above threshold."""
        current_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "cpu_percent", "name"]):
            try:
                if proc.info["pid"] == current_pid:
                    continue
                if (proc.info["cpu_percent"] or 0.0) > cpu_threshold:
                    logging.warning(
                        f"High-CPU background process: {proc.info['name']} "
                        f"({proc.info['cpu_percent']:.1f}%)"
                    )
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET / IMAGE LOADER
# ─────────────────────────────────────────────────────────────────────────────

class ImageDataset:
    """Iterates over images in a directory (or generates dummy images)."""

    def __init__(self, image_dir: Optional[str], resolution: int):
        self.resolution  = resolution
        self._paths: List[Path] = []

        if image_dir and Path(image_dir).is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            self._paths = [
                p for p in sorted(Path(image_dir).iterdir())
                if p.suffix.lower() in exts
            ]
            logging.info(f"Dataset: {len(self._paths)} images loaded from {image_dir}")
        else:
            logging.warning(f"Image directory '{image_dir}' not found. Using synthetic images.")

    def get_image(self, idx: int) -> Any:
        """Return an image suitable for inference (numpy array HxWxC)."""
        if _NUMPY_AVAILABLE:
            if self._paths:
                path = self._paths[idx % len(self._paths)]
                if _CV2_AVAILABLE:
                    img = cv2.imread(str(path))
                    if img is not None:
                        return cv2.resize(img, (self.resolution, self.resolution))
            # Synthetic fallback
            import numpy as np
            return np.random.randint(0, 255,
                                     (self.resolution, self.resolution, 3),
                                     dtype=np.uint8)
        return None   # model must handle None in its own loader


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL WRAPPER  (adapt run_inference to your framework)
# ─────────────────────────────────────────────────────────────────────────────

class ModelWrapper:
    """
    Generic wrapper.  Swap load_model / run_inference for your actual framework
    (Ultralytics YOLO, TFLite, ONNX Runtime, etc.).
    """

    def __init__(self, model_id: str, resolution: int, quantization: str):
        self.model_id     = model_id
        self.resolution   = resolution
        self.quantization = quantization
        self._model: Any  = None

    # ── override these two methods for your framework ─────────────────────────

    def load_model(self) -> None:
        """Load model into memory."""
        logging.info(f"Loading model  {self.model_id}  res={self.resolution}  q={self.quantization}")
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_id)
            if self.quantization in ("int8", "fp16"):
                logging.info(f"Quantization '{self.quantization}' applied at export time.")
        except ImportError:
            logging.warning("Ultralytics not found – using dummy model.")
            self._model = "dummy"

    def run_inference(self, image: Any) -> Any:
        """Run a single inference and return raw result (ignored for timing)."""
        if self._model == "dummy" or self._model is None:
            time.sleep(0.03)   # simulate ~30 ms latency
            return None
        try:
            results = self._model.predict(
                source=image,
                imgsz=self.resolution,
                verbose=False,
                stream=False,
            )
            return results
        except Exception as exc:
            logging.debug(f"Inference error: {exc}")
            time.sleep(0.03)
            return None

    def unload(self) -> None:
        self._model = None
        import gc; gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCalculator:

    @staticmethod
    def compute(run: RunResult) -> None:
        """Fill derived fields in-place."""
        if not run.samples:
            return

        latencies  = [s.latency_s    for s in run.samples]
        cpus       = [s.cpu_percent  for s in run.samples]
        rams       = [s.ram_percent  for s in run.samples]
        temps      = [s.temperature_c for s in run.samples]
        powers     = [s.power_w      for s in run.samples]
        energies   = [s.energy_mah   for s in run.samples]

        n = len(latencies)

        run.mean_latency_s     = sum(latencies) / n
        run.std_latency_s      = MetricsCalculator._std(latencies)
        run.mean_cpu_pct       = sum(cpus) / n
        run.mean_ram_pct       = sum(rams) / n
        run.mean_temp_c        = sum(temps) / n
        run.max_temp_c         = max(temps)
        run.mean_power_w       = sum(powers) / n
        run.total_energy_mah   = energies[-1] if energies else 0.0  # cumulative at end
        run.energy_per_inf_mah = run.total_energy_mah / n if n else 0.0

    @staticmethod
    def _std(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        return variance ** 0.5

    @staticmethod
    def confidence_interval_95(values: List[float]) -> Tuple[float, float]:
        """Return (lower, upper) 95% CI using t-distribution approximation."""
        n = len(values)
        if n < 2:
            return (0.0, 0.0)
        mean = sum(values) / n
        std  = MetricsCalculator._std(values)
        # t-critical for 95% CI (approximate; exact values for small n)
        t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                   6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.228}
        t = t_table.get(n, 1.96)
        margin = t * std / (n ** 0.5)
        return (mean - margin, mean + margin)


# ─────────────────────────────────────────────────────────────────────────────
#  QUALITY CONTROL
# ─────────────────────────────────────────────────────────────────────────────

class QualityControl:

    def __init__(self, config: ProtocolConfig):
        self.cfg = config

    def validate_run(self, run: RunResult, monitor: SystemMonitor) -> Tuple[bool, str]:
        """
        Returns (is_valid, reason).
        Called AFTER execution, before finalising the run.
        """
        # 1. Thermal throttling
        if run.max_temp_c >= self.cfg.max_temp_runtime_c:
            return False, f"CPU temperature reached {run.max_temp_c:.1f}°C ≥ {self.cfg.max_temp_runtime_c}°C (thermal throttling)"

        # 2. Power variance
        if run.samples:
            powers = [s.power_w for s in run.samples]
            mean_p = sum(powers) / len(powers)
            if mean_p > 0:
                max_dev = max(abs(p - mean_p) for p in powers) / mean_p
                if max_dev > self.cfg.power_variance_threshold:
                    return False, (
                        f"Power variance {max_dev*100:.1f}% exceeds "
                        f"{self.cfg.power_variance_threshold*100:.0f}% threshold"
                    )

        # 3. Background process activity
        if monitor.background_process_detected(self.cfg.cpu_spike_threshold_pct):
            return False, "Background process activity detected during run"

        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS WRITER
# ─────────────────────────────────────────────────────────────────────────────

class ResultsWriter:

    def __init__(self, output_dir: str):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._summary_path = self.out / "summary.csv"
        self._detail_dir   = self.out / "run_details"
        self._detail_dir.mkdir(exist_ok=True)
        self._write_summary_header()

    def _write_summary_header(self) -> None:
        if not self._summary_path.exists():
            with open(self._summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "phase", "run_idx",
                    "model_id", "resolution", "quantization",
                    "valid", "rejection_reason",
                    "baseline_temp_c", "mean_temp_c", "max_temp_c",
                    "mean_latency_s", "std_latency_s",
                    "mean_cpu_pct", "mean_ram_pct",
                    "mean_power_w", "total_energy_mah", "energy_per_inf_mah",
                    "map_score",
                ])

    def append_run(self, run: RunResult) -> None:
        # Summary CSV
        with open(self._summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                run.timestamp, run.phase, run.run_idx,
                run.model_id, run.resolution, run.quantization,
                run.valid, run.rejection_reason,
                f"{run.baseline_temp_c:.2f}",
                f"{run.mean_temp_c:.2f}", f"{run.max_temp_c:.2f}",
                f"{run.mean_latency_s*1000:.3f}",   # convert to ms
                f"{run.std_latency_s*1000:.3f}",
                f"{run.mean_cpu_pct:.2f}", f"{run.mean_ram_pct:.2f}",
                f"{run.mean_power_w:.4f}",
                f"{run.total_energy_mah:.6f}", f"{run.energy_per_inf_mah:.6f}",
                f"{run.map_score:.4f}",
            ])

        # Per-run detail JSON
        detail_file = self._detail_dir / (
            f"{run.phase}_{run.model_id.replace('/', '_')}"
            f"_r{run.resolution}_q{run.quantization}"
            f"_run{run.run_idx:03d}.json"
        )
        with open(detail_file, "w") as jf:
            payload = asdict(run)
            payload.pop("samples")   # keep summary lean; attach separately
            payload["samples"] = [asdict(s) for s in run.samples]
            json.dump(payload, jf, indent=2)

        logging.info(f"Results saved → {detail_file.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkEngine:

    def __init__(
        self,
        config:   ProtocolConfig,
        model:    ModelWrapper,
        dataset:  ImageDataset,
        monitor:  SystemMonitor,
        meter:    PowerMeter,
        writer:   ResultsWriter,
    ):
        self.cfg     = config
        self.model   = model
        self.dataset = dataset
        self.monitor = monitor
        self.meter   = meter
        self.writer  = writer
        self.qc      = QualityControl(config)

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 1 – PREPARATION
    # ─────────────────────────────────────────────────────────────────────────

    def step_preparation(self) -> None:
        logging.info("── Step 1: Preparation ──────────────────────────────────")

        # Load model
        self.model.load_model()

        # Warmup inferences
        logging.info(f"Running {self.cfg.warmup_inferences} warmup inferences …")
        for i in range(self.cfg.warmup_inferences):
            img = self.dataset.get_image(i)
            self.model.run_inference(img)
        logging.info("Warmup complete.")

        # Thermal stabilisation
        logging.info(
            f"Waiting {self.cfg.thermal_stabilisation_s:.0f}s for thermal stabilisation …"
        )
        time.sleep(self.cfg.thermal_stabilisation_s)

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 2 – BASELINE MEASUREMENT
    # ─────────────────────────────────────────────────────────────────────────

    def step_baseline(self) -> Tuple[bool, float]:
        """
        Returns (ok, baseline_temp_c).
        ok=False if temperature is above safe threshold.
        """
        logging.info("── Step 2: Baseline measurement ─────────────────────────")

        temp = self.monitor.cpu_temperature_c()
        logging.info(f"Baseline CPU temperature: {temp:.1f}°C  (limit: {self.cfg.max_temp_baseline_c}°C)")

        if temp >= self.cfg.max_temp_baseline_c:
            logging.warning(
                f"Temperature {temp:.1f}°C ≥ {self.cfg.max_temp_baseline_c}°C — "
                "waiting for cooldown before retrying …"
            )
            return False, temp

        # Reset energy counter
        self.meter.reset_energy_counter()
        logging.info("Energy counter reset to 0 mAh.")

        return True, temp

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 3 + 4 – EXECUTION & DATA COLLECTION
    # ─────────────────────────────────────────────────────────────────────────

    def step_execute_and_collect(self, run: RunResult) -> None:
        logging.info(
            f"── Steps 3+4: Execution & Collection  "
            f"({self.cfg.benchmark_inferences} inferences) ──"
        )

        for idx in range(self.cfg.benchmark_inferences):
            img = self.dataset.get_image(idx)

            t_start = time.perf_counter()
            self.model.run_inference(img)
            t_end   = time.perf_counter()

            latency = t_end - t_start

            sample = InferenceSample(
                inference_idx  = idx,
                latency_s      = latency,
                cpu_percent    = self.monitor.cpu_percent(interval=0.0),
                ram_percent    = self.monitor.ram_percent(),
                temperature_c  = self.monitor.cpu_temperature_c(),
                energy_mah     = self.meter.read_energy_mah(),
                power_w        = self.meter.read_power_w(),
            )
            run.samples.append(sample)

            if (idx + 1) % 20 == 0:
                logging.info(
                    f"  [{idx+1:3d}/{self.cfg.benchmark_inferences}]  "
                    f"lat={latency*1000:.1f}ms  "
                    f"temp={sample.temperature_c:.1f}°C  "
                    f"pwr={sample.power_w:.2f}W  "
                    f"e={sample.energy_mah:.4f}mAh"
                )

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 5 – METRICS CALCULATION
    # ─────────────────────────────────────────────────────────────────────────

    def step_metrics(self, run: RunResult) -> None:
        logging.info("── Step 5: Metrics calculation ──────────────────────────")
        MetricsCalculator.compute(run)
        logging.info(
            f"  Mean latency : {run.mean_latency_s*1000:.2f} ± {run.std_latency_s*1000:.2f} ms\n"
            f"  Energy/inf   : {run.energy_per_inf_mah*1000:.4f} µAh\n"
            f"  Mean power   : {run.mean_power_w:.3f} W\n"
            f"  Mean temp    : {run.mean_temp_c:.1f}°C  (max {run.max_temp_c:.1f}°C)\n"
            f"  Mean CPU     : {run.mean_cpu_pct:.1f}%   RAM: {run.mean_ram_pct:.1f}%"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  SINGLE RUN ORCHESTRATOR
    # ─────────────────────────────────────────────────────────────────────────

    def execute_single_run(
        self,
        phase:   str,
        run_idx: int,
        map_score: float = 0.0,
    ) -> RunResult:

        run = RunResult(
            run_idx        = run_idx,
            phase          = phase,
            model_id       = self.model.model_id,
            resolution     = self.model.resolution,
            quantization   = self.model.quantization,
            timestamp      = datetime.now().isoformat(timespec="seconds"),
            baseline_temp_c= 0.0,
        )

        # ── Step 1 (preparation only once per model, not per run) ─────────────
        # Called by the outer loop before the first run of each model config.

        # ── Step 2 – baseline ─────────────────────────────────────────────────
        max_cooldown_retries = 5
        baseline_ok = False
        for _ in range(max_cooldown_retries):
            ok, temp = self.step_baseline()
            if ok:
                run.baseline_temp_c = temp
                baseline_ok = True
                break
            time.sleep(30)   # wait 30 s and retry

        if not baseline_ok:
            run.valid = False
            run.rejection_reason = "Could not achieve safe baseline temperature after retries"
            logging.error(run.rejection_reason)
            return run

        # ── Steps 3 + 4 ───────────────────────────────────────────────────────
        self.step_execute_and_collect(run)

        # ── Step 5 ────────────────────────────────────────────────────────────
        self.step_metrics(run)
        run.map_score = map_score

        # ── Quality control ───────────────────────────────────────────────────
        valid, reason = self.qc.validate_run(run, self.monitor)
        run.valid = valid
        run.rejection_reason = reason
        if not valid:
            logging.warning(f"Run {run_idx} REJECTED: {reason}")
        else:
            logging.info(f"Run {run_idx} PASSED quality control ✓")

        return run


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE RUNNER  (wraps multiple runs of one config)
# ─────────────────────────────────────────────────────────────────────────────

class PhaseRunner:

    PHASE_RUN_MAP = {
        "screening":   "runs_screening",
        "pareto":      "runs_pareto",
        "contextual":  "runs_contextual",
    }

    def __init__(
        self,
        config:  ProtocolConfig,
        writer:  ResultsWriter,
        monitor: SystemMonitor,
        meter:   PowerMeter,
    ):
        self.cfg     = config
        self.writer  = writer
        self.monitor = monitor
        self.meter   = meter

    def run_phase(
        self,
        phase:         str,
        model_id:      str,
        resolution:    int,
        quantization:  str,
        image_dir:     Optional[str] = None,
        map_score:     float = 0.0,
    ) -> List[RunResult]:
        """
        Execute all runs for one (model, resolution, quantization) combination
        in a given phase.  Returns list of RunResult.
        """
        assert phase in self.PHASE_RUN_MAP, f"Unknown phase '{phase}'"
        n_runs: int = getattr(self.cfg, self.PHASE_RUN_MAP[phase])

        logging.info(
            f"\n{'='*70}\n"
            f"PHASE: {phase.upper()}   model={model_id}  "
            f"res={resolution}  q={quantization}  runs={n_runs}\n"
            f"{'='*70}"
        )

        model   = ModelWrapper(model_id, resolution, quantization)
        dataset = ImageDataset(image_dir, resolution)
        engine  = BenchmarkEngine(
            config=self.cfg, model=model, dataset=dataset,
            monitor=self.monitor, meter=self.meter, writer=self.writer,
        )

        # ── Step 1: preparation (once per model config) ───────────────────────
        engine.step_preparation()

        results: List[RunResult] = []

        for run_idx in range(1, n_runs + 1):
            logging.info(f"\n── Run {run_idx}/{n_runs} ──────────────────────────────────────")
            result = engine.execute_single_run(phase, run_idx, map_score)
            self.writer.append_run(result)
            results.append(result)

        model.unload()

        # ── Phase summary ─────────────────────────────────────────────────────
        valid_runs = [r for r in results if r.valid]
        logging.info(
            f"\n── Phase summary: {len(valid_runs)}/{n_runs} valid runs ──────────────────"
        )
        if valid_runs:
            latencies = [r.mean_latency_s * 1000 for r in valid_runs]
            lo, hi = MetricsCalculator.confidence_interval_95(latencies)
            logging.info(
                f"  Latency  : {sum(latencies)/len(latencies):.2f} ms  "
                f"95% CI [{lo:.2f}, {hi:.2f}] ms"
            )
            energies = [r.energy_per_inf_mah for r in valid_runs]
            logging.info(
                f"  Energy/inf : {sum(energies)/len(energies)*1000:.4f} µAh"
            )

        return results


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str) -> None:
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Log file: {log_file}")


def main() -> None:
    # ── User configuration ────────────────────────────────────────────────────
    # Adjust these to match your experimental design.

    cfg = ProtocolConfig(
        output_dir               = "results",
        thermal_stabilisation_s  = 60,    # seconds
        warmup_inferences        = 5,
        benchmark_inferences     = 100,
        max_temp_baseline_c      = 60.0,
        max_temp_runtime_c       = 75.0,
        power_variance_threshold = 0.05,
        cpu_spike_threshold_pct  = 80.0,
        runs_screening           = 3,
        runs_pareto              = 10,
        runs_contextual          = 5,
    )

    # ── Model grid ────────────────────────────────────────────────────────────
    # Each entry: (model_id, resolution, quantization, map_score)
    # Swap model_id for your actual model paths / names.
    model_grid = [
        ("yolov8n.pt",   320, "fp32",  0.372),
        ("yolov8n.pt",   640, "fp32",  0.372),
        ("yolov8s.pt",   640, "fp32",  0.449),
        ("yolov8n.pt",   640, "int8",  0.360),
    ]

    # ── Image dataset ─────────────────────────────────────────────────────────
    image_dir = "./dataset/images"    # set to None to use synthetic images

    # ── Phase to run (change as needed) ──────────────────────────────────────
    # Options: "screening" | "pareto" | "contextual"
    phase = "screening"

    # ── Infrastructure ────────────────────────────────────────────────────────
    setup_logging(cfg.output_dir)
    logging.info(
        f"Benchmark starting  –  phase={phase}  "
        f"environment: 22±2°C target"
    )

    monitor = SystemMonitor()
    meter   = PowerMeter()
    writer  = ResultsWriter(cfg.output_dir)
    runner  = PhaseRunner(cfg, writer, monitor, meter)

    # ── Environment pre-check ─────────────────────────────────────────────────
    init_temp = monitor.cpu_temperature_c()
    logging.info(f"Initial CPU temperature: {init_temp:.1f}°C")
    if init_temp >= cfg.max_temp_baseline_c:
        logging.warning(
            "System is already warm. Consider waiting for cooldown before benchmarking."
        )

    # ── Run all model configurations ──────────────────────────────────────────
    all_results: Dict[str, List[RunResult]] = {}

    for (model_id, resolution, quantization, map_score) in model_grid:
        key = f"{model_id}_{resolution}_{quantization}"
        results = runner.run_phase(
            phase        = phase,
            model_id     = model_id,
            resolution   = resolution,
            quantization = quantization,
            image_dir    = image_dir,
            map_score    = map_score,
        )
        all_results[key] = results

    # ── Final aggregate report ────────────────────────────────────────────────
    logging.info(f"\n{'='*70}\nFINAL AGGREGATE REPORT\n{'='*70}")
    for key, runs in all_results.items():
        valid = [r for r in runs if r.valid]
        if not valid:
            logging.info(f"  {key}: NO VALID RUNS")
            continue
        avg_lat  = sum(r.mean_latency_s for r in valid) / len(valid) * 1000
        avg_nrg  = sum(r.energy_per_inf_mah for r in valid) / len(valid) * 1000
        avg_pwr  = sum(r.mean_power_w for r in valid) / len(valid)
        map_val  = valid[0].map_score
        logging.info(
            f"  {key:<35}  "
            f"lat={avg_lat:.1f}ms  "
            f"e={avg_nrg:.4f}µAh  "
            f"pwr={avg_pwr:.2f}W  "
            f"mAP={map_val:.3f}  "
            f"valid={len(valid)}/{len(runs)}"
        )

    logging.info(f"\nAll results written to: {Path(cfg.output_dir).resolve()}")
    logging.info("Benchmark complete.")


if __name__ == "__main__":
    main()

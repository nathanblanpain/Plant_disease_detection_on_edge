"""
benchmark_protocol.py
=====================
Energy & Performance Measurement Protocol for Edge AI Models on Raspberry Pi 5
Implements the 5-step protocol: Preparation → Baseline → Execution → Collection → Metrics
Supports 3 phases: initial screening (3 runs), Pareto validation (10 runs), contextual (5 runs)

Hardware assumptions:
  - Temperature  : vcgencmd measure_temp  (RPi built-in)
  - CPU / RAM    : psutil
  - Models       : Ultralytics YOLO (.pt)  OR  TFLite (.tflite)
                   Auto-detected from file extension via create_model_wrapper().

Install deps:
  pip install psutil ultralytics opencv-python-headless
  pip install tflite-runtime          # lightweight TFLite (recommended on Pi)
  # OR: pip install tensorflow        # full TF (also provides tf.lite.Interpreter)
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
from typing import Any, Dict, List, Optional, Tuple

import psutil

# ─────────────────────────────────────────────
#  Optional hardware imports (graceful fallback)
# ─────────────────────────────────────────────

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
    np = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProtocolConfig:
    runs_screening:           int   = 3
    runs_pareto:              int   = 10
    runs_contextual:          int   = 5
    warmup_inferences:        int   = 5
    benchmark_inferences:     int   = 100
    thermal_stabilisation_s:  float = 30.0
    max_temp_baseline_c:      float = 60.0
    max_temp_runtime_c:       float = 75.0
    cpu_spike_threshold_pct:  float = 80.0
    output_dir:               str   = "results"


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
    gt_class_id:      int   = -1
    pred_class_id:    int   = -1
    pred_confidence:  float = 0.0
    correct:          bool  = False
    condition:        str   = "unknown"   # "fog" | "rain" | "original" | "unknown"


@dataclass
class RunResult:
    run_idx:          int
    phase:            str
    model_id:         str
    resolution:       int
    quantization:     str
    timestamp:        str
    baseline_temp_c:  float
    samples:          List[InferenceSample] = field(default_factory=list)
    mean_latency_s:   float = 0.0
    std_latency_s:    float = 0.0
    var_latency_s:    float = 0.0
    mean_cpu_pct:     float = 0.0
    std_cpu_pct:      float = 0.0
    var_cpu_pct:      float = 0.0
    mean_ram_pct:     float = 0.0
    std_ram_pct:      float = 0.0
    var_ram_pct:      float = 0.0
    mean_temp_c:      float = 0.0
    std_temp_c:       float = 0.0
    var_temp_c:       float = 0.0
    max_temp_c:       float = 0.0
    map_score:        float = 0.0
    accuracy:         float = 0.0
    correct_count:    int   = 0
    labeled_count:    int   = 0
    valid:            bool  = True
    rejection_reason: str   = ""


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED PREDICTION RESULT
#  Both ModelWrapper backends return this so the engine stays backend-agnostic.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Normalised prediction returned by every ModelWrapper backend.

    boxes        : list of [x_center, y_center, w, h]  (normalised 0-1)
    confidences  : confidence score per detection
    class_ids    : integer class index per detection
    names        : {class_id: class_name}
    """
    boxes:       List[List[float]]
    confidences: List[float]
    class_ids:   List[int]
    names:       Dict[int, str] = field(default_factory=dict)

    @property
    def best(self) -> Optional[Tuple[List[float], float, int]]:
        """(box, confidence, class_id) for the highest-confidence detection."""
        if not self.confidences:
            return None
        idx = int(max(range(len(self.confidences)), key=lambda i: self.confidences[i]))
        return self.boxes[idx], self.confidences[idx], self.class_ids[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  HARDWARE INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class SystemMonitor:

    @staticmethod
    def cpu_temperature_c() -> float:
        try:
            raw = subprocess.check_output(
                ["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            return float(raw.split("=")[1].replace("'C", ""))
        except Exception:
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
    """
    Iterates over images whose filename starts with "fog", "rain", or "original".

    Label lookup tries (in order):
      1. <image_dir>/<stem>.txt               (labels alongside images)
      2. <image_dir>/../labels/<stem>.txt     (YOLO standard layout)
      3. <image_dir>/../../labels/<stem>.txt  (two levels up, for train/val splits)
    """

    CONDITIONS = ("fog", "rain", "original")

    def __init__(self, image_dir: Optional[str], resolution: int):
        self.resolution = resolution
        self._paths: List[Path] = []

        if image_dir and Path(image_dir).is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            all_images = sorted(
                p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts
            )
            # Keep only images whose name starts with one of the known conditions
            self._paths = [
                p for p in all_images
                if any(p.name.startswith(cond) for cond in self.CONDITIONS)
            ]
            skipped = len(all_images) - len(self._paths)
            logging.info(
                f"Dataset: {len(self._paths)} images from {image_dir} "
                f"(fog/rain/original only; {skipped} skipped)"
            )
            # Log per-condition breakdown
            for cond in self.CONDITIONS:
                n = sum(1 for p in self._paths if p.name.startswith(cond))
                logging.info(f"  {cond:<10}: {n} images")
        else:
            logging.warning(f"Image dir '{image_dir}' not found — using synthetic images.")

    def get_condition(self, idx: int) -> str:
        """Return the condition tag for the image at idx."""
        if not self._paths:
            return "unknown"
        name = self._paths[idx % len(self._paths)].name
        for cond in self.CONDITIONS:
            if name.startswith(cond):
                return cond
        return "unknown"

    def get_image(self, idx: int) -> Any:
        if _NUMPY_AVAILABLE:
            if self._paths:
                path = self._paths[idx % len(self._paths)]
                if _CV2_AVAILABLE:
                    img = cv2.imread(str(path))
                    if img is not None:
                        return cv2.resize(img, (self.resolution, self.resolution))
            # Synthetic fallback
            return np.random.randint(0, 256,
                                     (self.resolution, self.resolution, 3),
                                     dtype=np.uint8)
        return None

    def get_ground_truth_class(self, idx: int) -> int:
        if not self._paths:
            return -1
        img_path = self._paths[idx % len(self._paths)]

        # Try multiple candidate label locations
        candidates = [
            img_path.with_suffix(".txt"),                                                        # alongside image
            img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name,              # ../labels/
            img_path.parent.parent.parent / "labels" / img_path.with_suffix(".txt").name,       # ../../labels/
        ]

        for label_path in candidates:
            if label_path.exists():
                try:
                    first_line = label_path.read_text().strip().splitlines()[0]
                    return int(first_line.split()[0])
                except Exception:
                    logging.debug(f"Could not parse label: {label_path}")
                    return -1

        return -1  # no label file found in any candidate location


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

class ModelWrapper:
    """Base interface — subclasses implement load_model / run_inference / unload."""

    def __init__(self, model_id: str, resolution: int, quantization: str):
        self.model_id     = model_id
        self.resolution   = resolution
        self.quantization = quantization

    def load_model(self) -> None:
        raise NotImplementedError

    def run_inference(self, image: Any) -> Optional[PredictionResult]:
        raise NotImplementedError

    def unload(self) -> None:
        import gc; gc.collect()


# ── Ultralytics YOLO backend (.pt files) ─────────────────────────────────────

class UltralyticsWrapper(ModelWrapper):

    def __init__(self, model_id: str, resolution: int, quantization: str):
        super().__init__(model_id, resolution, quantization)
        self._model: Any = None

    def load_model(self) -> None:
        logging.info(f"[YOLO] Loading {self.model_id}  res={self.resolution}  q={self.quantization}")
        from ultralytics import YOLO
        self._model = YOLO(self.model_id)

    def run_inference(self, image: Any) -> Optional[PredictionResult]:
        if self._model is None:
            time.sleep(0.03)
            return None
        try:
            results = self._model.predict(
                source=image, imgsz=self.resolution, verbose=False, stream=False,
            )
        except Exception as exc:
            logging.debug(f"[YOLO] Inference error: {exc}")
            return None

        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return PredictionResult(boxes=[], confidences=[], class_ids=[],
                                    names=results[0].names if results else {})
        r = results[0]
        return PredictionResult(
            boxes       = r.boxes.xywhn.tolist(),
            confidences = r.boxes.conf.tolist(),
            class_ids   = [int(c) for c in r.boxes.cls.tolist()],
            names       = r.names,
        )

    def unload(self) -> None:
        self._model = None
        super().unload()


# ── TFLite backend (.tflite files) ───────────────────────────────────────────

class TFLiteWrapper(ModelWrapper):
    """
    Wraps a YOLOv8 .tflite model (float32, float16, or int8).

    YOLOv8 TFLite output shape: [1, 4 + num_classes, num_anchors]
      rows 0-3  : x_center, y_center, w, h  (normalised 0-1)
      rows 4+   : per-class scores  (no separate objectness in v8)

    INT8 dequantisation is applied automatically when the output dtype is
    int8/uint8 using the scale + zero_point stored in output_details.
    """

    def __init__(self, model_id: str, resolution: int, quantization: str):
        super().__init__(model_id, resolution, quantization)
        self._interpreter:    Any   = None
        self._input_details:  Any   = None
        self._output_details: Any   = None
        self._input_scale:    float = 1.0
        self._input_zero:     int   = 0
        self._output_scale:   float = 1.0
        self._output_zero:    int   = 0
        self._is_int8_input:  bool  = False
        self._is_int8_output: bool  = False

    def load_model(self) -> None:
        logging.info(f"[TFLite] Loading {self.model_id}  res={self.resolution}  q={self.quantization}")

        # Try backends: ai-edge-litert (new name) → tflite-runtime → tensorflow
        self._interpreter = None
        for _backend in ["ai_edge_litert", "tflite_runtime", "tensorflow"]:
            try:
                if _backend == "ai_edge_litert":
                    from ai_edge_litert.interpreter import Interpreter
                    self._interpreter = Interpreter(model_path=self.model_id)
                    logging.info("[TFLite] Backend: ai_edge_litert")
                elif _backend == "tflite_runtime":
                    import tflite_runtime.interpreter as _tflite
                    self._interpreter = _tflite.Interpreter(model_path=self.model_id)
                    logging.info("[TFLite] Backend: tflite_runtime")
                else:
                    import tensorflow as tf
                    self._interpreter = tf.lite.Interpreter(model_path=self.model_id)
                    logging.info("[TFLite] Backend: tensorflow.lite")
                break
            except (ImportError, ModuleNotFoundError):
                continue

        if self._interpreter is None:
            raise RuntimeError(
                "No TFLite backend found. Install one with:\n"
                "  pip install ai-edge-litert"
            )

        self._interpreter.allocate_tensors()
        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        import numpy as _np
        inp_dtype = self._input_details[0]["dtype"]
        out_dtype = self._output_details[0]["dtype"]

        self._is_int8_input  = inp_dtype in (_np.int8, _np.uint8)
        self._is_int8_output = out_dtype in (_np.int8, _np.uint8)

        if self._is_int8_input:
            q = self._input_details[0].get("quantization", (1.0, 0))
            self._input_scale, self._input_zero = float(q[0]), int(q[1])
            logging.info(f"[TFLite] INT8 input  scale={self._input_scale:.6f}  zero={self._input_zero}")

        if self._is_int8_output:
            q = self._output_details[0].get("quantization", (1.0, 0))
            self._output_scale, self._output_zero = float(q[0]), int(q[1])
            logging.info(f"[TFLite] INT8 output scale={self._output_scale:.6f}  zero={self._output_zero}")

        logging.info(f"[TFLite] Input  shape={self._input_details[0]['shape']}  dtype={inp_dtype}")
        logging.info(f"[TFLite] Output shape={self._output_details[0]['shape']}  dtype={out_dtype}")

    def run_inference(self, image: Any, conf_threshold: float = 0.25) -> Optional[PredictionResult]:
        if self._interpreter is None or not _NUMPY_AVAILABLE:
            time.sleep(0.03)
            return None
        try:
            input_data = self._preprocess(image)
            self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
            self._interpreter.invoke()
            raw = self._interpreter.get_tensor(self._output_details[0]["index"])
        except Exception as exc:
            logging.debug(f"[TFLite] Inference error: {exc}")
            return None

        return self._postprocess(raw, conf_threshold)

    def _preprocess(self, image: Any) -> "np.ndarray":
        import numpy as _np

        if image is None:
            image = _np.random.randint(0, 256,
                                       (self.resolution, self.resolution, 3),
                                       dtype=_np.uint8)

        h, w = image.shape[:2]
        if (h != self.resolution or w != self.resolution) and _CV2_AVAILABLE:
            image = cv2.resize(image, (self.resolution, self.resolution))

        # BGR → RGB
        if _CV2_AVAILABLE:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_f = image.astype(_np.float32) / 255.0   # [0, 1]
        img_f = _np.expand_dims(img_f, axis=0)       # [1, H, W, 3]

        if self._is_int8_input:
            inp_dtype = self._input_details[0]["dtype"]
            img_q = _np.round(img_f / self._input_scale + self._input_zero)
            img_q = _np.clip(img_q,
                             _np.iinfo(inp_dtype).min,
                             _np.iinfo(inp_dtype).max).astype(inp_dtype)
            return img_q

        return img_f

    def _postprocess(self, raw: "np.ndarray", conf_threshold: float) -> PredictionResult:
        import numpy as _np

        # Dequantise INT8 output if needed
        if self._is_int8_output:
            raw = (raw.astype(_np.float32) - self._output_zero) * self._output_scale

        output = _np.squeeze(raw)   # remove batch dim → 2-D

        if output.ndim == 1:
            output = output.reshape(1, -1)  # single-anchor edge case

        rows, cols = output.shape
        # YOLOv8 default TFLite export: [4+classes, anchors]
        # If rows > cols the tensor came out transposed — fix it.
        if rows < cols:
            output = output.T   # → [anchors, 4+classes]

        num_anchors, num_fields = output.shape
        boxes_raw    = output[:, :4]                                      # xywh norm
        class_scores = output[:, 4:]                                      # [anchors, classes]

        class_ids_all   = _np.argmax(class_scores, axis=1)
        confidences_all = class_scores[_np.arange(num_anchors), class_ids_all]

        mask = confidences_all >= conf_threshold

        return PredictionResult(
            boxes       = boxes_raw[mask].tolist(),
            confidences = confidences_all[mask].tolist(),
            class_ids   = [int(c) for c in class_ids_all[mask].tolist()],
        )

    def unload(self) -> None:
        self._interpreter    = None
        self._input_details  = None
        self._output_details = None
        super().unload()


# ── Factory: picks the right wrapper from the file extension ─────────────────

def create_model_wrapper(model_id: str, resolution: int, quantization: str) -> ModelWrapper:
    """
    .tflite  → TFLiteWrapper
    anything else (.pt, folder …) → UltralyticsWrapper
    """
    if model_id.endswith(".tflite"):
        return TFLiteWrapper(model_id, resolution, quantization)
    return UltralyticsWrapper(model_id, resolution, quantization)


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCalculator:

    @staticmethod
    def compute(run: RunResult) -> None:
        if not run.samples:
            return
        latencies = [s.latency_s     for s in run.samples]
        cpus      = [s.cpu_percent   for s in run.samples]
        rams      = [s.ram_percent   for s in run.samples]
        temps     = [s.temperature_c for s in run.samples]
        n = len(latencies)

        run.mean_latency_s = sum(latencies) / n
        run.std_latency_s  = MetricsCalculator._std(latencies)
        run.var_latency_s  = run.std_latency_s ** 2

        run.mean_cpu_pct   = sum(cpus) / n
        run.std_cpu_pct    = MetricsCalculator._std(cpus)
        run.var_cpu_pct    = run.std_cpu_pct ** 2

        run.mean_ram_pct   = sum(rams) / n
        run.std_ram_pct    = MetricsCalculator._std(rams)
        run.var_ram_pct    = run.std_ram_pct ** 2

        run.mean_temp_c    = sum(temps) / n
        run.std_temp_c     = MetricsCalculator._std(temps)
        run.var_temp_c     = run.std_temp_c ** 2
        run.max_temp_c     = max(temps)

    @staticmethod
    def _std(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5

    @staticmethod
    def confidence_interval_95(values: List[float]) -> Tuple[float, float]:
        n = len(values)
        if n < 2:
            return (0.0, 0.0)
        mean = sum(values) / n
        std  = MetricsCalculator._std(values)
        t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                   6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.228}
        t = t_table.get(n, 1.96)
        margin = t * std / (n ** 0.5)
        return (mean - margin, mean + margin)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFUSION MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class ConfusionMatrixBuilder:
    """
    Builds one confusion matrix per condition (fog / rain / original)
    plus one overall matrix, from a list of InferenceSample.

    Matrix convention : rows = ground-truth class, cols = predicted class.
    Samples where gt_class_id == -1 (no label) are skipped.
    Samples where pred_class_id == -1 (no detection) are counted in a
    dedicated "no_detection" column so recall issues are visible.
    """

    NO_DETECTION_LABEL = "no_detection"

    @staticmethod
    def build(
        samples:    List[InferenceSample],
        class_names: Dict[int, str],
    ) -> Dict[str, "pd.DataFrame | Dict"]:
        """
        Returns a dict  { condition_key → matrix_dict }
        where condition_key is one of "Overall", "fog", "rain", "original".

        matrix_dict is a nested dict  { gt_label → { pred_label → count } }.
        Call to_csv_block() to render it.
        """
        conditions = ["Overall"] + list(ImageDataset.CONDITIONS)
        # initialise empty counters
        matrices: Dict[str, Dict[str, Dict[str, int]]] = {c: {} for c in conditions}

        labeled = [s for s in samples if s.gt_class_id != -1]

        for s in labeled:
            gt_label   = class_names.get(s.gt_class_id,   f"cls_{s.gt_class_id}")
            pred_label = (
                class_names.get(s.pred_class_id, f"cls_{s.pred_class_id}")
                if s.pred_class_id != -1
                else ConfusionMatrixBuilder.NO_DETECTION_LABEL
            )
            for key in ("Overall", s.condition):
                if key not in matrices:
                    continue
                row = matrices[key].setdefault(gt_label, {})
                row[pred_label] = row.get(pred_label, 0) + 1

        return matrices

    @staticmethod
    def to_csv_lines(matrix: Dict[str, Dict[str, int]]) -> List[str]:
        """Render a single matrix as CSV lines (list of strings).

        Columns are derived from GT rows (not observed predictions) so that
        a class that was never predicted still appears as an all-zero column.
        """
        if not matrix:
            return ["(empty — no labeled samples for this condition)"]

        nd = ConfusionMatrixBuilder.NO_DETECTION_LABEL

        # Rows  = all GT classes, sorted
        all_gt   = sorted(matrix.keys())
        # Columns = same GT classes (ensures square matrix) + any extra predicted
        #           labels (e.g. no_detection) that are not GT classes
        extra_pred = sorted({
            p
            for row in matrix.values()
            for p in row
            if p not in all_gt and p != nd
        })
        all_pred = all_gt + extra_pred + ([nd] if any(
            nd in row for row in matrix.values()
        ) else [])

        header = ["GT \\ Pred"] + all_pred
        lines  = [",".join(header)]
        for gt in all_gt:
            row = [gt] + [str(matrix[gt].get(pred, 0)) for pred in all_pred]
            lines.append(",".join(row))
        return lines

    @staticmethod
    def save(
        matrices:    Dict[str, Dict[str, Dict[str, int]]],
        output_dir:  Path,
        file_prefix: str,
    ) -> None:
        """Write one CSV file per condition + one overall."""
        for condition, matrix in matrices.items():
            filename = output_dir / f"{file_prefix}_confusion_{condition}.csv"
            lines    = ConfusionMatrixBuilder.to_csv_lines(matrix)
            filename.write_text("\n".join(lines))
            logging.info(f"Confusion matrix saved → {filename.name}")

    @staticmethod
    def log_summary(
        matrices: Dict[str, Dict[str, Dict[str, int]]],
    ) -> None:
        """Log a compact per-condition accuracy from the matrices."""
        logging.info("── Confusion matrix summary ─────────────────────────────")
        for condition, matrix in matrices.items():
            total   = sum(v for row in matrix.values() for v in row.values())
            correct = sum(matrix[cls].get(cls, 0) for cls in matrix)
            acc     = correct / total if total else 0.0
            logging.info(
                f"  {condition:<12}: {correct}/{total} correct  ({acc:.1%})"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  QUALITY CONTROL
# ─────────────────────────────────────────────────────────────────────────────

class QualityControl:

    def __init__(self, config: ProtocolConfig):
        self.cfg = config

    def validate_run(self, run: RunResult, monitor: SystemMonitor) -> Tuple[bool, str]:
        if run.max_temp_c >= self.cfg.max_temp_runtime_c:
            return False, (
                f"CPU temperature reached {run.max_temp_c:.1f}°C "
                f"≥ {self.cfg.max_temp_runtime_c}°C (thermal throttling)"
            )
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
                csv.writer(f).writerow([
                    "timestamp", "phase", "run_idx",
                    "model_id", "resolution", "quantization",
                    "valid", "rejection_reason",
                    "baseline_temp_c",
                    "mean_latency_ms", "std_latency_ms", "var_latency_us2",
                    "mean_cpu_pct",    "std_cpu_pct",    "var_cpu_pct2",
                    "mean_ram_pct",    "std_ram_pct",    "var_ram_pct2",
                    "mean_temp_c",     "std_temp_c",     "var_temp_c2",     "max_temp_c",
                    "map_score",
                    "accuracy", "correct_count", "labeled_count",
                ])

    def append_run(self, run: RunResult) -> None:
        with open(self._summary_path, "a", newline="") as f:
            csv.writer(f).writerow([
                run.timestamp, run.phase, run.run_idx,
                run.model_id, run.resolution, run.quantization,
                run.valid, run.rejection_reason,
                f"{run.baseline_temp_c:.2f}",
                f"{run.mean_latency_s*1000:.3f}",
                f"{run.std_latency_s*1000:.3f}",
                f"{run.var_latency_s*1e6:.4f}",        # µs²
                f"{run.mean_cpu_pct:.2f}",
                f"{run.std_cpu_pct:.4f}",
                f"{run.var_cpu_pct:.4f}",
                f"{run.mean_ram_pct:.2f}",
                f"{run.std_ram_pct:.4f}",
                f"{run.var_ram_pct:.4f}",
                f"{run.mean_temp_c:.2f}",
                f"{run.std_temp_c:.4f}",
                f"{run.var_temp_c:.4f}",
                f"{run.max_temp_c:.2f}",
                f"{run.map_score:.4f}",
                f"{run.accuracy:.4f}", run.correct_count, run.labeled_count,
            ])

        detail_file = self._detail_dir / (
            f"{run.phase}_{run.model_id.replace('/', '_')}"
            f"_r{run.resolution}_q{run.quantization}"
            f"_run{run.run_idx:03d}.json"
        )
        with open(detail_file, "w") as jf:
            payload = asdict(run)
            payload["samples"] = [asdict(s) for s in run.samples]
            json.dump(payload, jf, indent=2)

        logging.info(f"Results saved → {detail_file.name}")

        # ── Confusion matrices ────────────────────────────────────────────────
        matrices = getattr(run, "_matrices", None)
        if matrices:
            prefix = (
                f"{run.phase}_{run.model_id.replace('/', '_')}"
                f"_r{run.resolution}_q{run.quantization}"
                f"_run{run.run_idx:03d}"
            )
            ConfusionMatrixBuilder.save(matrices, self._detail_dir, prefix)


# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkEngine:

    def __init__(
        self,
        config:  ProtocolConfig,
        model:   ModelWrapper,
        dataset: ImageDataset,
        monitor: SystemMonitor,
        writer:  ResultsWriter,
    ):
        self.cfg     = config
        self.model   = model
        self.dataset = dataset
        self.monitor = monitor
        self.writer  = writer
        self.qc      = QualityControl(config)

    def step_preparation(self) -> None:
        logging.info("── Step 1: Preparation ──────────────────────────────────")
        self.model.load_model()
        logging.info(f"Running {self.cfg.warmup_inferences} warmup inferences …")
        for i in range(self.cfg.warmup_inferences):
            self.model.run_inference(self.dataset.get_image(i))
        logging.info("Warmup complete.")
        logging.info(f"Waiting {self.cfg.thermal_stabilisation_s:.0f}s for thermal stabilisation …")
        time.sleep(self.cfg.thermal_stabilisation_s)

    def step_baseline(self) -> Tuple[bool, float]:
        logging.info("── Step 2: Baseline measurement ─────────────────────────")
        temp = self.monitor.cpu_temperature_c()
        logging.info(f"Baseline CPU temperature: {temp:.1f}°C  (limit: {self.cfg.max_temp_baseline_c}°C)")
        if temp >= self.cfg.max_temp_baseline_c:
            logging.warning(f"Temperature {temp:.1f}°C ≥ {self.cfg.max_temp_baseline_c}°C — waiting …")
            return False, temp
        return True, temp

    def step_execute_and_collect(self, run: RunResult) -> None:
        import random

        n_images = len(self.dataset._paths) if self.dataset._paths else 1

        # Contextual phase → iterate sequentially over every image once.
        # All other phases → draw benchmark_inferences random samples.
        if run.phase == "contextual":
            idx_sequence = list(range(n_images))
            logging.info(
                f"── Steps 3+4: Execution & Collection  "
                f"({n_images} inferences — full sequential pass) ──"
            )
        else:
            idx_sequence = [random.randrange(n_images)
                            for _ in range(self.cfg.benchmark_inferences)]
            logging.info(
                f"── Steps 3+4: Execution & Collection  "
                f"({self.cfg.benchmark_inferences} inferences — random sampling) ──"
            )

        run_names: Dict[int, str] = {}

        for seq, idx in enumerate(idx_sequence):
            img       = self.dataset.get_image(idx)
            gt_class  = self.dataset.get_ground_truth_class(idx)
            condition = self.dataset.get_condition(idx)

            t_start = time.perf_counter()
            result  = self.model.run_inference(img)   # always returns PredictionResult | None
            t_end   = time.perf_counter()
            latency = t_end - t_start

            # ── Extract best prediction ───────────────────────────────────────
            pred_class_id   = -1
            pred_confidence = 0.0

            if result is not None:
                if result.names and not run_names:
                    run_names = result.names
                best = result.best
                if best is not None:
                    _, pred_confidence, pred_class_id = best

            correct = (gt_class != -1) and (pred_class_id == gt_class)

            if gt_class != -1:
                status    = "✓" if correct else "✗"
                gt_name   = run_names.get(gt_class,      f"id={gt_class}")
                pred_name = run_names.get(pred_class_id, f"id={pred_class_id}")
                logging.debug(
                    f"  [{idx:3d}] GT={gt_name:<30} PRED={pred_name:<30} "
                    f"conf={pred_confidence:.2f}  {status}"
                )

            run.samples.append(InferenceSample(
                inference_idx   = idx,
                latency_s       = latency,
                cpu_percent     = self.monitor.cpu_percent(interval=0.0),
                ram_percent     = self.monitor.ram_percent(),
                temperature_c   = self.monitor.cpu_temperature_c(),
                gt_class_id     = gt_class,
                pred_class_id   = pred_class_id,
                pred_confidence = pred_confidence,
                correct         = correct,
                condition       = condition,
            ))

            # Expose accumulated class names so step_metrics can use them
            run._class_names = run_names  # type: ignore[attr-defined]

            if (seq + 1) % 20 == 0:
                total_inferences = len(idx_sequence)
                labeled_so_far = sum(1 for s in run.samples if s.gt_class_id != -1)
                correct_so_far = sum(1 for s in run.samples if s.correct)
                acc = correct_so_far / labeled_so_far if labeled_so_far else 0.0
                logging.info(
                    f"  [{seq+1:3d}/{total_inferences}]  "
                    f"lat={latency*1000:.1f}ms  "
                    f"temp={run.samples[-1].temperature_c:.1f}°C  "
                    f"acc={acc:.1%} ({correct_so_far}/{labeled_so_far})"
                )

    def step_metrics(self, run: RunResult) -> None:
        logging.info("── Step 5: Metrics calculation ──────────────────────────")
        MetricsCalculator.compute(run)
        run.labeled_count = sum(1 for s in run.samples if s.gt_class_id != -1)
        run.correct_count = sum(1 for s in run.samples if s.correct)
        run.accuracy      = run.correct_count / run.labeled_count if run.labeled_count else 0.0
        logging.info(
            f"  Latency  : {run.mean_latency_s*1000:.2f} ms  "
            f"± {run.std_latency_s*1000:.2f}  var={run.var_latency_s*1e6:.4f} µs²\n"
            f"  CPU      : {run.mean_cpu_pct:.1f}%  "
            f"± {run.std_cpu_pct:.2f}  var={run.var_cpu_pct:.4f} %²\n"
            f"  RAM      : {run.mean_ram_pct:.1f}%  "
            f"± {run.std_ram_pct:.2f}  var={run.var_ram_pct:.4f} %²\n"
            f"  Temp     : {run.mean_temp_c:.1f}°C  "
            f"± {run.std_temp_c:.2f}  var={run.var_temp_c:.4f} °C²  (max {run.max_temp_c:.1f}°C)\n"
            f"  Accuracy : {run.accuracy:.1%}  ({run.correct_count}/{run.labeled_count} labeled)"
        )

        # ── Confusion matrices (one per condition + overall) ──────────────────
        class_names: Dict[int, str] = getattr(run, "_class_names", {})
        matrices = ConfusionMatrixBuilder.build(run.samples, class_names)
        ConfusionMatrixBuilder.log_summary(matrices)

        # Store on run for the writer to save
        run._matrices    = matrices       # type: ignore[attr-defined]
        run._class_names = class_names    # type: ignore[attr-defined]

    def execute_single_run(self, phase: str, run_idx: int, map_score: float = 0.0) -> RunResult:
        run = RunResult(
            run_idx         = run_idx,
            phase           = phase,
            model_id        = self.model.model_id,
            resolution      = self.model.resolution,
            quantization    = self.model.quantization,
            timestamp       = datetime.now().isoformat(timespec="seconds"),
            baseline_temp_c = 0.0,
        )

        for _ in range(5):
            ok, temp = self.step_baseline()
            if ok:
                run.baseline_temp_c = temp
                break
            time.sleep(30)
        else:
            run.valid            = False
            run.rejection_reason = "Could not achieve safe baseline temperature"
            logging.error(run.rejection_reason)
            return run

        self.step_execute_and_collect(run)
        self.step_metrics(run)
        run.map_score = map_score

        valid, reason = self.qc.validate_run(run, self.monitor)
        run.valid, run.rejection_reason = valid, reason
        if not valid:
            logging.warning(f"Run {run_idx} REJECTED: {reason}")
        else:
            logging.info(f"Run {run_idx} PASSED quality control ✓")

        return run


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class PhaseRunner:

    PHASE_RUN_MAP = {
        "screening":  "runs_screening",
        "pareto":     "runs_pareto",
        "contextual": "runs_contextual",
    }

    def __init__(self, config: ProtocolConfig, writer: ResultsWriter, monitor: SystemMonitor):
        self.cfg     = config
        self.writer  = writer
        self.monitor = monitor

    def run_phase(
        self,
        phase:        str,
        model_id:     str,
        resolution:   int,
        quantization: str,
        image_dir:    Optional[str] = None,
    ) -> List[RunResult]:
        assert phase in self.PHASE_RUN_MAP, f"Unknown phase '{phase}'"
        n_runs: int = getattr(self.cfg, self.PHASE_RUN_MAP[phase])

        logging.info(
            f"\n{'='*70}\n"
            f"PHASE: {phase.upper()}   model={model_id}  "
            f"res={resolution}  q={quantization}  runs={n_runs}\n"
            f"{'='*70}"
        )

        # ← Factory: .tflite → TFLiteWrapper, otherwise → UltralyticsWrapper
        model   = create_model_wrapper(model_id, resolution, quantization)
        dataset = ImageDataset(image_dir, resolution)
        engine  = BenchmarkEngine(
            config=self.cfg, model=model, dataset=dataset,
            monitor=self.monitor, writer=self.writer,
        )

        engine.step_preparation()

        results: List[RunResult] = []
        for run_idx in range(1, n_runs + 1):
            logging.info(f"\n── Run {run_idx}/{n_runs} ──────────────────────────────────────")
            result = engine.execute_single_run(phase, run_idx)
            self.writer.append_run(result)
            results.append(result)
            time.sleep(20)

        model.unload()

        valid_runs = [r for r in results if r.valid]
        logging.info(f"\n── Phase summary: {len(valid_runs)}/{n_runs} valid runs ──────────────────")
        if valid_runs:
            latencies = [r.mean_latency_s * 1000 for r in valid_runs]
            lo, hi    = MetricsCalculator.confidence_interval_95(latencies)
            avg_acc   = sum(r.accuracy for r in valid_runs) / len(valid_runs)
            avg_lat   = sum(latencies) / len(latencies)
            std_lat   = MetricsCalculator._std(latencies)
            var_lat   = std_lat ** 2

            summary = (
                f"  Latency  : {avg_lat:.2f} ms  "
                f"95% CI [{lo:.2f}, {hi:.2f}] ms\n"
                f"  Accuracy : {avg_acc:.1%} (avg across valid runs)"
            )
            if phase == "pareto":
                summary += (
                    f"\n  Variance : {var_lat:.4f} ms²  "
                    f"(std={std_lat:.4f} ms  CoV={std_lat/avg_lat*100:.2f}%)"
                )
            logging.info(summary)

        return results


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str) -> None:
    log_dir  = Path(output_dir)
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
    cfg = ProtocolConfig(
        output_dir              = "results",
        thermal_stabilisation_s = 30,
        warmup_inferences       = 5,
        benchmark_inferences    = 100,
        max_temp_baseline_c     = 60.0,
        max_temp_runtime_c      = 75.0,
        cpu_spike_threshold_pct = 80.0,
        runs_screening          = 3,
        runs_pareto             = 10,
        runs_contextual         = 5,
    )

    # .tflite → TFLiteWrapper   |   .pt or folder → UltralyticsWrapper
    model_grid = [
        ("yolov8_224/best_saved_model/best_int8.tflite", 224, "int8"),
        # ("yolov8_224/best.pt", 224, "fp32"),   # uncomment to also benchmark .pt
    ]

    image_dir = "./PlantVillage_224/"
    phase     = "screening"

    setup_logging(cfg.output_dir)
    logging.info(f"Benchmark starting  –  phase={phase}  environment: 22±2°C target")

    monitor = SystemMonitor()
    writer  = ResultsWriter(cfg.output_dir)
    runner  = PhaseRunner(cfg, writer, monitor)

    init_temp = monitor.cpu_temperature_c()
    logging.info(f"Initial CPU temperature: {init_temp:.1f}°C")
    if init_temp >= cfg.max_temp_baseline_c:
        logging.warning("System is already warm — consider waiting before benchmarking.")

    all_results: Dict[str, List[RunResult]] = {}
    for (model_id, resolution, quantization) in model_grid:
        key = f"{model_id}_{resolution}_{quantization}"
        all_results[key] = runner.run_phase(
            phase=phase, model_id=model_id, resolution=resolution,
            quantization=quantization, image_dir=image_dir,
        )

    logging.info(f"\n{'='*70}\nFINAL AGGREGATE REPORT\n{'='*70}")
    for key, runs in all_results.items():
        valid = [r for r in runs if r.valid]
        if not valid:
            logging.info(f"  {key}: NO VALID RUNS")
            continue
        avg_lat = sum(r.mean_latency_s for r in valid) / len(valid) * 1000
        avg_acc = sum(r.accuracy for r in valid) / len(valid)
        logging.info(
            f"  {key:<55}  lat={avg_lat:.1f}ms  acc={avg_acc:.1%}  "
            f"valid={len(valid)}/{len(runs)}"
        )

    logging.info(f"\nAll results written to: {Path(cfg.output_dir).resolve()}")
    logging.info("Benchmark complete.")


if __name__ == "__main__":
    main()
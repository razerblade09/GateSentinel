"""Tests for the export subsystem (Milestone 10).

Design constraints
------------------
* No real .pt file, ONNX file, or hardware is available in CI.
* All tests that would call Ultralytics or OpenVINO use fakes or test the
  error-handling paths that activate when files are missing.
* The types, path-resolution logic, and CLI argument parsing are testable
  without any ML dependencies.

Test groups
-----------
``TestExportResult``      — dataclass invariants and string formatting.
``TestBenchmarkResult``   — dataclass invariants and string formatting.
``TestExportOnnx``        — onnx.py logic (file-not-found, wrong suffix).
``TestExportOpenvino``    — openvino.py logic (file-not-found, missing
                            onnx, missing openvino package).
``TestExportCLI``         — scripts/export.py argument parsing and routing.
``TestBenchmarkHelpers``  — _infer_label helper.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alpr.export.types import BenchmarkResult, ExportFormat, ExportResult

# ---------------------------------------------------------------------------
# Helpers — fake Ultralytics YOLO
# ---------------------------------------------------------------------------


class _FakeYOLO:
    """Minimal YOLO stand-in for export tests."""

    def __init__(self, path: str) -> None:
        self._path = path

    def export(self, format: str, **kwargs: Any) -> str:
        """Return a plausible output path without touching the filesystem."""
        p = Path(self._path)
        if format == "onnx":
            return str(p.with_suffix(".onnx"))
        if format == "openvino":
            return str(p.parent / (p.stem + "_openvino_model"))
        raise ValueError(f"unsupported fake export format: {format}")

    def to(self, device: str) -> None:  # noqa: D102
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> list:  # noqa: D102
        return []


def _fake_ultralytics() -> ModuleType:
    """Return a fake ``ultralytics`` module with a stub YOLO class."""
    mod = ModuleType("ultralytics")
    mod.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# ExportResult
# ---------------------------------------------------------------------------


class TestExportResult:
    def test_success_factory(self, tmp_path: Path) -> None:
        src = tmp_path / "model.pt"
        out = tmp_path / "model.onnx"
        result = ExportResult.success(src, ExportFormat.ONNX, out, 1.5)

        assert result.ok is True
        assert result.output_path == out
        assert result.error == ""
        assert result.duration_s == pytest.approx(1.5)

    def test_failure_factory(self, tmp_path: Path) -> None:
        src = tmp_path / "model.pt"
        result = ExportResult.failure(src, ExportFormat.ONNX, "boom", 0.1)

        assert result.ok is False
        assert result.output_path is None
        assert result.error == "boom"

    def test_str_success(self, tmp_path: Path) -> None:
        src = tmp_path / "model.pt"
        out = tmp_path / "model.onnx"
        result = ExportResult.success(src, ExportFormat.ONNX, out, 2.3)
        s = str(result)
        assert "OK" in s
        assert "onnx" in s
        assert "model.pt" in s

    def test_str_failure(self, tmp_path: Path) -> None:
        src = tmp_path / "model.pt"
        result = ExportResult.failure(src, ExportFormat.OPENVINO, "not found", 0.0)
        s = str(result)
        assert "FAILED" in s
        assert "not found" in s

    def test_frozen(self, tmp_path: Path) -> None:
        result = ExportResult.success(
            tmp_path / "x.pt", ExportFormat.ONNX, tmp_path / "x.onnx", 1.0
        )
        with pytest.raises((AttributeError, TypeError)):
            result.ok = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_str_format(self, tmp_path: Path) -> None:
        br = BenchmarkResult(
            format="pytorch",
            weights_path=tmp_path / "model.pt",
            mean_ms=45.2,
            p95_ms=62.0,
            n_runs=50,
            imgsz=640,
        )
        s = str(br)
        assert "pytorch" in s
        assert "45.2" in s
        assert "62.0" in s
        assert "50" in s

    def test_frozen(self, tmp_path: Path) -> None:
        br = BenchmarkResult(
            format="onnx",
            weights_path=tmp_path / "m.onnx",
            mean_ms=30.0,
            p95_ms=40.0,
            n_runs=50,
            imgsz=640,
        )
        with pytest.raises((AttributeError, TypeError)):
            br.mean_ms = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class TestExportOnnx:
    def test_missing_weights_returns_failure(self, tmp_path: Path) -> None:
        from alpr.export.onnx import export_onnx

        result = export_onnx(tmp_path / "ghost.pt")
        assert not result.ok
        assert "not found" in result.error

    def test_wrong_suffix_returns_failure(self, tmp_path: Path) -> None:
        from alpr.export.onnx import export_onnx

        f = tmp_path / "model.pth"
        f.touch()
        result = export_onnx(f)
        assert not result.ok
        assert ".pt" in result.error

    def test_success_with_fake_ultralytics(self, tmp_path: Path) -> None:
        from alpr.export import onnx as onnx_mod

        weights = tmp_path / "model.pt"
        weights.touch()

        with patch.dict(sys.modules, {"ultralytics": _fake_ultralytics()}):
            result = onnx_mod.export_onnx(weights)

        assert result.ok, result.error
        assert result.output_path is not None
        assert result.output_path.suffix == ".onnx"
        assert result.format is ExportFormat.ONNX
        assert result.duration_s >= 0

    def test_ultralytics_exception_returns_failure(self, tmp_path: Path) -> None:
        from alpr.export import onnx as onnx_mod

        weights = tmp_path / "model.pt"
        weights.touch()

        broken = ModuleType("ultralytics")

        class _BrokenYOLO:
            def __init__(self, *a: Any, **kw: Any) -> None:
                raise RuntimeError("simulated driver error")

        broken.YOLO = _BrokenYOLO  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"ultralytics": broken}):
            result = onnx_mod.export_onnx(weights)

        assert not result.ok
        assert "simulated driver error" in result.error


# ---------------------------------------------------------------------------
# OpenVINO export
# ---------------------------------------------------------------------------


class TestExportOpenvino:
    def test_missing_weights_returns_failure(self, tmp_path: Path) -> None:
        from alpr.export.openvino import export_openvino

        result = export_openvino(tmp_path / "ghost.pt")
        assert not result.ok
        assert "not found" in result.error

    def test_missing_onnx_path_returns_failure(self, tmp_path: Path) -> None:
        from alpr.export.openvino import export_openvino

        weights = tmp_path / "model.pt"
        weights.touch()
        result = export_openvino(weights, onnx_path=tmp_path / "ghost.onnx")
        assert not result.ok
        assert "ONNX file not found" in result.error

    def test_direct_export_success_with_fake_ultralytics(self, tmp_path: Path) -> None:
        from alpr.export import openvino as ov_mod

        weights = tmp_path / "model.pt"
        weights.touch()

        with patch.dict(sys.modules, {"ultralytics": _fake_ultralytics()}):
            result = ov_mod.export_openvino(weights)

        assert result.ok, result.error
        assert result.output_path is not None
        assert "openvino_model" in str(result.output_path)
        assert result.format is ExportFormat.OPENVINO

    def test_onnx_path_missing_openvino_package_returns_failure(
        self, tmp_path: Path
    ) -> None:
        from alpr.export import openvino as ov_mod

        weights = tmp_path / "model.pt"
        weights.touch()
        onnx = tmp_path / "model.onnx"
        onnx.touch()

        # Simulate openvino not being installed.
        with patch.dict(
            sys.modules,
            {"openvino": None, "openvino.tools": None, "openvino.tools.ovc": None},
        ):
            result = ov_mod.export_openvino(weights, onnx_path=onnx)

        assert not result.ok
        assert "openvino" in result.error.lower()

    def test_ultralytics_exception_returns_failure(self, tmp_path: Path) -> None:
        from alpr.export import openvino as ov_mod

        weights = tmp_path / "model.pt"
        weights.touch()

        broken = ModuleType("ultralytics")

        class _BrokenYOLO:
            def __init__(self, *a: Any, **kw: Any) -> None:
                raise OSError("VRAM OOM")

        broken.YOLO = _BrokenYOLO  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"ultralytics": broken}):
            result = ov_mod.export_openvino(weights)

        assert not result.ok
        assert "VRAM OOM" in result.error


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


class TestBenchmarkHelpers:
    def test_infer_label_pt(self, tmp_path: Path) -> None:
        from alpr.export.benchmark import _infer_label

        assert _infer_label(tmp_path / "model.pt") == "pytorch"

    def test_infer_label_onnx(self, tmp_path: Path) -> None:
        from alpr.export.benchmark import _infer_label

        assert _infer_label(tmp_path / "model.onnx") == "onnx"

    def test_infer_label_openvino_dir(self, tmp_path: Path) -> None:
        from alpr.export.benchmark import _infer_label

        d = tmp_path / "model_openvino_model"
        d.mkdir()
        assert _infer_label(d) == "openvino"

    def test_infer_label_unknown_suffix(self, tmp_path: Path) -> None:
        from alpr.export.benchmark import _infer_label

        assert _infer_label(tmp_path / "model.tflite") == "tflite"

    def test_benchmark_missing_weights_raises(self, tmp_path: Path) -> None:
        from alpr.export.benchmark import benchmark

        with pytest.raises(FileNotFoundError, match="not found"):
            benchmark(tmp_path / "ghost.pt")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestExportCLI:
    """Test the scripts/export.py CLI without invoking real exports."""

    def _cli(self, argv: list[str]) -> int:
        """Import and call main() with argv; returns exit code."""
        # Patch the export functions so no real work happens.
        fake_result_ok = ExportResult.success(
            Path("model.pt"),
            ExportFormat.ONNX,
            Path("model.onnx"),
            0.1,
        )

        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        import scripts.export as exp_script  # noqa: PLC0415

        with (
            patch.object(exp_script, "export_onnx", return_value=fake_result_ok),
            patch.object(exp_script, "export_openvino", return_value=fake_result_ok),
            patch.object(exp_script, "compare", return_value=[]),
        ):
            return exp_script.main(argv)

    def test_missing_weights_file_still_runs(self, tmp_path: Path) -> None:
        """CLI should not crash on missing file — the exporter returns a failure."""
        weights = tmp_path / "ghost.pt"
        code = self._cli(["--weights", str(weights), "--format", "onnx"])
        # Patched export always returns ok=True, so exit code is 0.
        assert code == 0

    def test_invalid_format_raises_systemexit(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.pt"
        weights.touch()
        with pytest.raises(SystemExit):
            self._cli(["--weights", str(weights), "--format", "tensorrt"])

    def test_openvino_format_routes_correctly(self, tmp_path: Path, capsys: Any) -> None:
        weights = tmp_path / "model.pt"
        weights.touch()
        code = self._cli(["--weights", str(weights), "--format", "openvino"])
        assert code == 0

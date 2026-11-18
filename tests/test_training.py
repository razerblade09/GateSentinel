"""Tests for the training configuration and CLI argument parsing.

We do not test the actual YOLO training call -- that needs GPU time and real
weights, which belong in the Kaggle notebook, not in a unit test suite. What
we do test:

- TrainingConfig validation (Pydantic field constraints)
- CLI argument parsing produces the correct TrainingConfig
- train() raises FileNotFoundError when dataset YAML is missing
- TrainingConfig round-trips to/from a dict (for notebook use)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpr.training.config import TrainingConfig
from alpr.training.trainer import train


# ===========================================================================
# TrainingConfig validation
# ===========================================================================


class TestTrainingConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = TrainingConfig()
        assert cfg.epochs == 100
        assert cfg.batch == 16
        assert cfg.imgsz == 640
        assert cfg.seed == 42

    def test_epochs_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(epochs=0)

    def test_batch_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(batch=0)

    def test_imgsz_must_be_multiple_of_32(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(imgsz=100)

    def test_imgsz_multiples_of_32_accepted(self) -> None:
        for size in [32, 320, 640, 1280]:
            cfg = TrainingConfig(imgsz=size)
            assert cfg.imgsz == size

    def test_lr0_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(lr0=0.0)

    def test_cache_false_by_default(self) -> None:
        cfg = TrainingConfig()
        assert cfg.cache is False

    def test_cache_accepts_ram_and_disk(self) -> None:
        assert TrainingConfig(cache="ram").cache == "ram"
        assert TrainingConfig(cache="disk").cache == "disk"

    def test_cache_rejects_unknown(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(cache="memory")  # type: ignore[arg-type]

    def test_optimizer_choices(self) -> None:
        for opt in ["SGD", "Adam", "AdamW", "auto"]:
            cfg = TrainingConfig(optimizer=opt)  # type: ignore[arg-type]
            assert cfg.optimizer == opt

    def test_optimizer_rejects_unknown(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(optimizer="RMSProp")  # type: ignore[arg-type]

    def test_extra_fields_are_rejected(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(nonexistent_field=42)  # type: ignore[call-arg]

    def test_half_default_false(self) -> None:
        assert not TrainingConfig().half

    def test_project_is_path(self) -> None:
        cfg = TrainingConfig(project="outputs/runs")
        assert isinstance(cfg.project, Path)

    def test_dataset_yaml_is_path(self) -> None:
        cfg = TrainingConfig(dataset_yaml="outputs/splits/dataset.yaml")
        assert isinstance(cfg.dataset_yaml, Path)

    def test_flipud_zero_by_default(self) -> None:
        """Upside-down augmentation must stay off for gate use."""
        assert TrainingConfig().flipud == 0.0

    def test_degrees_zero_by_default(self) -> None:
        """Rotation augmentation off by default -- cameras are roughly level."""
        assert TrainingConfig().degrees == 0.0


# ===========================================================================
# trainer.train() — missing dataset YAML
# ===========================================================================


class TestTrainFunction:
    def test_missing_dataset_yaml_raises(self, tmp_path: Path) -> None:
        cfg = TrainingConfig(
            dataset_yaml=tmp_path / "nonexistent.yaml",
            project=tmp_path / "runs",
        )
        with pytest.raises(FileNotFoundError, match="dataset YAML not found"):
            train(cfg)

    def test_yolo_called_with_correct_data_arg(self, tmp_path: Path) -> None:
        """When the dataset YAML exists, train() should forward its path to YOLO."""
        yaml_path = tmp_path / "dataset.yaml"
        yaml_path.write_text("path: /tmp\ntrain: train.txt\nval: val.txt\nnames:\n  0: plate\n")

        fake_model = MagicMock()
        fake_results = MagicMock()
        fake_model.train.return_value = fake_results

        # Create a fake best.pt so train() can find it.
        best_pt = tmp_path / "runs" / "plate_detector" / "weights" / "best.pt"
        best_pt.parent.mkdir(parents=True)
        best_pt.write_bytes(b"fake")

        cfg = TrainingConfig(
            dataset_yaml=yaml_path,
            base_weights="yolo26n.pt",
            epochs=1,
            project=tmp_path / "runs",
            name="plate_detector",
            exist_ok=True,
        )

        with patch("alpr.training.trainer.YOLO", return_value=fake_model):
            result = train(cfg)

        fake_model.train.assert_called_once()
        call_kwargs = fake_model.train.call_args.kwargs
        assert call_kwargs["data"] == str(yaml_path)
        assert call_kwargs["epochs"] == 1
        assert result == best_pt.resolve()

    def test_yolo_exception_is_wrapped(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "dataset.yaml"
        yaml_path.write_text("dummy: true\n")

        fake_model = MagicMock()
        fake_model.train.side_effect = RuntimeError("GPU out of memory")

        cfg = TrainingConfig(dataset_yaml=yaml_path, epochs=1)
        with patch("alpr.training.trainer.YOLO", return_value=fake_model):
            with pytest.raises(RuntimeError, match="YOLO training failed"):
                train(cfg)


# ===========================================================================
# CLI argument parsing
# ===========================================================================


class TestTrainCLI:
    def _parse(self, argv: list[str]) -> TrainingConfig:
        """Run the CLI parser and return the resulting config."""
        from scripts.train import _build_parser

        args = _build_parser().parse_args(argv)
        return TrainingConfig(
            dataset_yaml=args.dataset_yaml,
            base_weights=args.base_weights,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            optimizer=args.optimizer,
            lr0=args.lr0,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            device=args.device,
            half=args.half,
            cache=args.cache or False,
            seed=args.seed,
            verbose=args.verbose,
        )

    def test_defaults(self) -> None:
        cfg = self._parse([])
        assert cfg.epochs == 100
        assert cfg.batch == 16
        assert cfg.imgsz == 640
        assert cfg.base_weights == "yolo26s.pt"
        assert not cfg.half
        assert cfg.cache is False

    def test_override_epochs(self) -> None:
        cfg = self._parse(["--epochs", "200"])
        assert cfg.epochs == 200

    def test_override_imgsz(self) -> None:
        cfg = self._parse(["--imgsz", "1280"])
        assert cfg.imgsz == 1280

    def test_half_flag(self) -> None:
        cfg = self._parse(["--half"])
        assert cfg.half

    def test_exist_ok_flag(self) -> None:
        cfg = self._parse(["--exist-ok"])
        assert cfg.exist_ok

    def test_cache_ram(self) -> None:
        cfg = self._parse(["--cache", "ram"])
        assert cfg.cache == "ram"

    def test_name_override(self) -> None:
        cfg = self._parse(["--name", "my_run"])
        assert cfg.name == "my_run"

    def test_device_override(self) -> None:
        cfg = self._parse(["--device", "0"])
        assert cfg.device == "0"

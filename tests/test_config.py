"""Tests for the configuration layer.

These cover the two things most likely to break silently: layer precedence and
schema validation. A config bug that is not caught here surfaces as a confusing
runtime failure hours into a training run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from alpr.config.loader import (
    ConfigError,
    deep_merge,
    env_overrides,
    find_project_root,
    load_config,
)
from alpr.config.schema import AppConfig


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A minimal configs/ directory with a baseline and one profile."""
    directory = tmp_path / "configs"
    directory.mkdir()
    (directory / "default.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {"data_dir": "data"},
                "runtime": {"device": "cpu", "max_fps": 15.0},
                "plate_detector": {"confidence": 0.35},
            }
        ),
        encoding="utf-8",
    )
    (directory / "kaggle.yaml").write_text(
        yaml.safe_dump({"runtime": {"device": "cuda"}}), encoding="utf-8"
    )
    return directory


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    return tmp_path


def test_defaults_are_valid() -> None:
    """The schema must be constructible with no input at all."""
    assert AppConfig().plate_detector.confidence == 0.35


def test_loads_baseline(config_dir: Path, project_root: Path) -> None:
    config = load_config(config_dir=config_dir, project_root=project_root, environ={})
    assert config.runtime.device == "cpu"


def test_profile_overrides_baseline(config_dir: Path, project_root: Path) -> None:
    config = load_config(
        "kaggle", config_dir=config_dir, project_root=project_root, environ={}
    )
    assert config.runtime.device == "cuda"
    # Untouched baseline values must survive the merge.
    assert config.runtime.max_fps == 15.0


def test_env_beats_yaml(config_dir: Path, project_root: Path) -> None:
    config = load_config(
        config_dir=config_dir,
        project_root=project_root,
        environ={"ALPR__RUNTIME__DEVICE": "mps"},
    )
    assert config.runtime.device == "mps"


def test_explicit_override_beats_env(config_dir: Path, project_root: Path) -> None:
    config = load_config(
        config_dir=config_dir,
        project_root=project_root,
        environ={"ALPR__RUNTIME__DEVICE": "mps"},
        overrides={"runtime": {"device": "cpu"}},
    )
    assert config.runtime.device == "cpu"


def test_env_values_are_typed() -> None:
    """Env vars arrive as strings; numeric and boolean values must coerce."""
    result = env_overrides(
        {
            "ALPR__PLATE_DETECTOR__CONFIDENCE": "0.75",
            "ALPR__DATABASE__ECHO": "true",
            "ALPR__VEHICLE_DETECTOR__IMGSZ": "960",
            "UNRELATED": "ignored",
        }
    )
    assert result["plate_detector"]["confidence"] == 0.75
    assert result["database"]["echo"] is True
    assert result["vehicle_detector"]["imgsz"] == 960
    assert "unrelated" not in result


def test_relative_paths_resolve_against_project_root(
    config_dir: Path, project_root: Path
) -> None:
    config = load_config(config_dir=config_dir, project_root=project_root, environ={})
    assert config.paths.data_dir.is_absolute()
    assert config.paths.data_dir == (project_root / "data").resolve()


def test_paths_absolute_even_when_section_omitted(
    tmp_path: Path, project_root: Path
) -> None:
    """Omitting the paths section must still yield absolute paths.

    Regression test: schema defaults are relative, so if they are applied after
    path resolution they end up interpreted against the working directory.
    """
    directory = tmp_path / "bare"
    directory.mkdir()
    (directory / "default.yaml").write_text("runtime:\n  device: cpu\n", encoding="utf-8")

    config = load_config(config_dir=directory, project_root=project_root, environ={})

    assert config.paths.data_dir.is_absolute()
    assert config.paths.models_dir.is_absolute()
    assert config.paths.outputs_dir.is_absolute()
    assert config.paths.data_dir == (project_root / "data").resolve()


def test_unknown_key_is_rejected(config_dir: Path, project_root: Path) -> None:
    """A typo in a config key must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        load_config(
            config_dir=config_dir,
            project_root=project_root,
            environ={},
            overrides={"plate_detector": {"confidance": 0.9}},
        )


def test_out_of_range_confidence_is_rejected(
    config_dir: Path, project_root: Path
) -> None:
    with pytest.raises(ValidationError):
        load_config(
            config_dir=config_dir,
            project_root=project_root,
            environ={},
            overrides={"plate_detector": {"confidence": 1.5}},
        )


def test_tracking_requires_vehicle_stage() -> None:
    """Voting needs a per-vehicle identity; the combination must be refused."""
    with pytest.raises(ValidationError, match="tracking.enabled requires"):
        AppConfig.model_validate(
            {"tracking": {"enabled": True}, "vehicle_detector": {"enabled": False}}
        )


def test_min_reads_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        AppConfig.model_validate(
            {"tracking": {"min_reads_for_vote": 20, "max_reads_per_track": 5}}
        )


def test_half_precision_rejected_on_cpu() -> None:
    with pytest.raises(ValidationError, match="not supported on device"):
        AppConfig.model_validate({"runtime": {"device": "cpu", "half": True}})


def test_invalid_plate_template_is_rejected() -> None:
    """Slot templates may only contain L (letter) and D (digit)."""
    with pytest.raises(ValidationError, match="only 'L'"):
        AppConfig.model_validate({"plate_format": {"formats": {"car": "LLLXDDD"}}})


def test_empty_plate_formats_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one plate format"):
        AppConfig.model_validate({"plate_format": {"formats": {}}})


def test_missing_baseline_raises(tmp_path: Path, project_root: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigError, match="missing baseline config"):
        load_config(config_dir=empty, project_root=project_root, environ={})


def test_unknown_profile_lists_available(config_dir: Path, project_root: Path) -> None:
    with pytest.raises(ConfigError, match="available: default, kaggle"):
        load_config(
            "nope", config_dir=config_dir, project_root=project_root, environ={}
        )


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 3}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3}}
    assert base == {"a": {"x": 1, "y": 2}}


def test_find_project_root_locates_marker(project_root: Path) -> None:
    nested = project_root / "src" / "alpr" / "config"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == project_root

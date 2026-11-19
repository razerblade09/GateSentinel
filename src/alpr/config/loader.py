"""Configuration loading with explicit, layered precedence.

Precedence, lowest to highest::

    configs/default.yaml   committed, the documented baseline
    configs/<profile>.yaml committed, e.g. "kaggle" for training runs
    configs/local.yaml     gitignored, machine-specific paths and camera source
    ALPR__* env vars       deployment overrides and anything secret
    explicit overrides     test fixtures and CLI flags

Each layer only needs to state what it changes; everything else falls through.
The order is written out in :func:`load_config` rather than being inferred by a
framework, so there is never a question about which layer won.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from alpr.config.schema import AppConfig, PathsConfig

ENV_PREFIX = "ALPR__"
ENV_DELIMITER = "__"

_PATH_FIELDS = ("project_root", "data_dir", "models_dir", "outputs_dir")


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is invalid."""


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until a directory containing pyproject.toml.

    Anchoring on a marker file rather than the current working directory is what
    makes the project runnable from anywhere. Resolving paths against ``cwd`` is
    the single most common source of "works on my machine" path bugs.
    """
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(
        f"could not locate pyproject.toml above {current}; "
        "pass paths.project_root explicitly"
    )


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested dicts merge key by key; every other type replaces wholesale. Lists
    replace rather than concatenate -- appending would make it impossible to
    *remove* a default entry, and silent accumulation is harder to debug than
    plain replacement.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping, tolerating an empty file but not a malformed one."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return raw


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Translate ``ALPR__SECTION__KEY`` variables into a nested dict.

    ``ALPR__DATABASE__URL=postgresql://...`` becomes
    ``{"database": {"url": "postgresql://..."}}``.

    Values are parsed as YAML scalars so ``true``, ``3`` and ``0.5`` arrive as
    bool, int and float rather than as strings. Environment variables are always
    strings, and without this step every numeric override would fail validation.
    """
    environ = os.environ if environ is None else environ
    result: dict[str, Any] = {}

    for raw_key, raw_value in environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in raw_key[len(ENV_PREFIX):].split(ENV_DELIMITER)]
        if not path or any(not part for part in path):
            raise ConfigError(f"malformed config environment variable: {raw_key}")

        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value

        cursor = result
        for part in path[:-1]:
            existing = cursor.get(part)
            if not isinstance(existing, dict):
                existing = {}
                cursor[part] = existing
            cursor = existing
        cursor[path[-1]] = value

    return result


def _resolve_paths(data: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Make every path in ``paths`` absolute, anchored on the project root.

    Schema defaults are folded in *before* resolution rather than being left to
    pydantic. If a config file omits the ``paths`` section, pydantic would supply
    its relative defaults after this function had already run, and those values
    would then be interpreted against the current working directory --
    reintroducing the exact "works only when launched from the right folder"
    bug this layer exists to prevent.
    """
    supplied = dict(data.get("paths") or {})
    paths: dict[str, Any] = {**PathsConfig().model_dump(), **supplied}

    root = Path(paths["project_root"]) if "project_root" in supplied else project_root
    if not root.is_absolute():
        root = (project_root / root).resolve()
    paths["project_root"] = root

    for field in _PATH_FIELDS:
        if field == "project_root":
            continue
        candidate = Path(paths[field])
        paths[field] = candidate if candidate.is_absolute() else (root / candidate).resolve()

    return {**data, "paths": paths}


def load_config(
    profile: str | None = None,
    *,
    config_dir: Path | None = None,
    project_root: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Build a validated :class:`AppConfig` from all configuration layers.

    Args:
        profile: Optional profile name, loading ``configs/<profile>.yaml``.
        config_dir: Directory holding the YAML files. Defaults to
            ``<project_root>/configs``.
        project_root: Project root. Discovered from pyproject.toml if omitted.
        overrides: Highest-priority values, used by tests and CLI flags.
        environ: Environment mapping. Defaults to ``os.environ``.

    Raises:
        ConfigError: If a required file is missing or a file is malformed.
        pydantic.ValidationError: If the merged result violates the schema.
    """
    root = project_root or find_project_root()
    directory = config_dir or (root / "configs")

    default_file = directory / "default.yaml"
    if not default_file.is_file():
        raise ConfigError(f"missing baseline config: {default_file}")

    data: dict[str, Any] = _read_yaml(default_file)

    if profile:
        profile_file = directory / f"{profile}.yaml"
        if not profile_file.is_file():
            available = sorted(p.stem for p in directory.glob("*.yaml"))
            raise ConfigError(
                f"unknown profile {profile!r}; available: {', '.join(available)}"
            )
        data = deep_merge(data, _read_yaml(profile_file))

    local_file = directory / "local.yaml"
    if local_file.is_file():
        data = deep_merge(data, _read_yaml(local_file))

    data = deep_merge(data, env_overrides(environ))

    if overrides:
        data = deep_merge(data, overrides)

    data = _resolve_paths(data, root)

    return AppConfig.model_validate(data)

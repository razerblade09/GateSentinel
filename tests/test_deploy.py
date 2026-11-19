"""Tests for Milestone 11 — Deployment preparation.

What is verified here
---------------------
Unlike previous milestones, Milestone 11 is primarily about *artifacts*
(config files, shell scripts, service files, Dockerfiles) rather than
importable Python modules.  The tests therefore check:

1. **Deploy config** — ``configs/deploy.yaml`` parses cleanly via the
   existing config loader and schema, and the values it sets are within
   valid ranges.
2. **Systemd service file** — Structural checks: required sections present,
   ExecStart points into the venv, Restart policy set.
3. **Dockerfile** — Multi-stage structure, non-root USER, VOLUME declarations,
   HEALTHCHECK present.
4. **docker-compose.yml** — Parses as valid YAML, required keys present.
5. **install.sh** — Shell safety flags set, key commands present, service
   enable step included.
6. **setup_logging helper** — Added to utils/logging in Milestone 10;
   tested here as it supports deploy-time CLI scripts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DEPLOY_DIR = PROJECT_ROOT / "deploy"
CONFIGS_DIR = PROJECT_ROOT / "configs"


@pytest.fixture
def deploy_yaml_text() -> str:
    path = CONFIGS_DIR / "deploy.yaml"
    assert path.exists(), f"configs/deploy.yaml not found at {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def service_text() -> str:
    path = DEPLOY_DIR / "gatesentinel.service"
    assert path.exists(), f"deploy/gatesentinel.service not found at {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def dockerfile_text() -> str:
    path = DEPLOY_DIR / "Dockerfile"
    assert path.exists(), f"deploy/Dockerfile not found at {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def compose_text() -> str:
    path = DEPLOY_DIR / "docker-compose.yml"
    assert path.exists(), f"deploy/docker-compose.yml not found at {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def install_sh_text() -> str:
    path = DEPLOY_DIR / "install.sh"
    assert path.exists(), f"deploy/install.sh not found at {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Deploy config (configs/deploy.yaml)
# ---------------------------------------------------------------------------


class TestDeployConfig:
    def test_file_exists(self) -> None:
        assert (CONFIGS_DIR / "deploy.yaml").exists()

    def test_parses_as_valid_yaml(self, deploy_yaml_text: str) -> None:
        data = yaml.safe_load(deploy_yaml_text)
        assert isinstance(data, dict), "deploy.yaml must be a YAML mapping"

    def test_device_is_openvino(self, deploy_yaml_text: str) -> None:
        data = yaml.safe_load(deploy_yaml_text)
        assert data["runtime"]["device"] == "openvino"

    def test_max_fps_capped(self, deploy_yaml_text: str) -> None:
        data = yaml.safe_load(deploy_yaml_text)
        assert data["runtime"]["max_fps"] <= 15.0, (
            "Deploy profile should cap FPS to conserve CPU on gate hardware"
        )

    def test_half_is_false(self, deploy_yaml_text: str) -> None:
        # FP16 is unsupported on openvino CPU device; config must not set it.
        data = yaml.safe_load(deploy_yaml_text)
        assert data["runtime"]["half"] is False

    def test_ocr_thresholds_tighter_than_defaults(self, deploy_yaml_text: str) -> None:
        data = yaml.safe_load(deploy_yaml_text)
        # Production thresholds should be stricter than defaults (0.50 / 0.60).
        assert data["ocr"]["min_char_confidence"] >= 0.55
        assert data["ocr"]["min_string_confidence"] >= 0.65

    def test_min_reads_tighter_than_default(self, deploy_yaml_text: str) -> None:
        data = yaml.safe_load(deploy_yaml_text)
        # Production should require more evidence before opening the gate.
        assert data["tracking"]["min_reads_for_vote"] >= 4

    def test_logging_format_is_json(self, deploy_yaml_text: str) -> None:
        data = yaml.safe_load(deploy_yaml_text)
        assert data["logging"]["format"] == "json"

    def test_wal_enabled(self, deploy_yaml_text: str) -> None:
        data = yaml.safe_load(deploy_yaml_text)
        assert data["database"]["wal"] is True

    def test_merges_with_schema(self, deploy_yaml_text: str) -> None:
        """Deploy config must survive the full loader → schema validation path."""
        from alpr.config.loader import load_config  # noqa: PLC0415

        cfg = load_config(profile="deploy")
        # Spot-check a few deploy-specific values.
        assert cfg.runtime.device == "openvino"
        assert cfg.runtime.max_fps <= 15.0
        assert cfg.logging.format == "json"


# ---------------------------------------------------------------------------
# Systemd service file
# ---------------------------------------------------------------------------


class TestServiceFile:
    def test_file_exists(self) -> None:
        assert (DEPLOY_DIR / "gatesentinel.service").exists()

    def test_has_unit_section(self, service_text: str) -> None:
        assert "[Unit]" in service_text

    def test_has_service_section(self, service_text: str) -> None:
        assert "[Service]" in service_text

    def test_has_install_section(self, service_text: str) -> None:
        assert "[Install]" in service_text

    def test_execstart_uses_venv_python(self, service_text: str) -> None:
        # The ExecStart line must call the venv python, not system python.
        assert ".venv" in service_text
        assert "ExecStart=" in service_text

    def test_restart_policy_set(self, service_text: str) -> None:
        assert "Restart=on-failure" in service_text

    def test_non_root_user(self, service_text: str) -> None:
        assert "User=gatesentinel" in service_text

    def test_wantedbymultiusertarget(self, service_text: str) -> None:
        assert "WantedBy=multi-user.target" in service_text

    def test_references_deploy_profile(self, service_text: str) -> None:
        # The run command must pass the deploy profile.
        assert "deploy" in service_text

    def test_timeout_stop_set(self, service_text: str) -> None:
        assert "TimeoutStopSec" in service_text


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


class TestDockerfile:
    def test_file_exists(self) -> None:
        assert (DEPLOY_DIR / "Dockerfile").exists()

    def test_multi_stage_build(self, dockerfile_text: str) -> None:
        # Two FROM lines = two stages.
        from_lines = [l for l in dockerfile_text.splitlines() if l.startswith("FROM")]
        assert len(from_lines) >= 2, "Dockerfile should use a multi-stage build"

    def test_non_root_user(self, dockerfile_text: str) -> None:
        assert "USER gatesentinel" in dockerfile_text

    def test_volume_declarations(self, dockerfile_text: str) -> None:
        assert "VOLUME" in dockerfile_text
        assert "/app/data" in dockerfile_text
        assert "/app/models" in dockerfile_text

    def test_healthcheck_present(self, dockerfile_text: str) -> None:
        assert "HEALTHCHECK" in dockerfile_text

    def test_cmd_uses_deploy_profile(self, dockerfile_text: str) -> None:
        assert "deploy" in dockerfile_text

    def test_headless_opencv(self, dockerfile_text: str) -> None:
        # Headless variant avoids pulling in X11 and display libs.
        assert "opencv-contrib-python-headless" in dockerfile_text

    def test_label_references_repo(self, dockerfile_text: str) -> None:
        assert "razerblade09/GateSentinel" in dockerfile_text


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------


class TestDockerCompose:
    def test_file_exists(self) -> None:
        assert (DEPLOY_DIR / "docker-compose.yml").exists()

    def test_parses_as_valid_yaml(self, compose_text: str) -> None:
        data = yaml.safe_load(compose_text)
        assert isinstance(data, dict)

    def test_has_services_key(self, compose_text: str) -> None:
        data = yaml.safe_load(compose_text)
        assert "services" in data

    def test_gatesentinel_service_defined(self, compose_text: str) -> None:
        data = yaml.safe_load(compose_text)
        assert "gatesentinel" in data["services"]

    def test_restart_policy(self, compose_text: str) -> None:
        data = yaml.safe_load(compose_text)
        svc = data["services"]["gatesentinel"]
        assert svc.get("restart") == "unless-stopped"

    def test_model_volume_readonly(self, compose_text: str) -> None:
        # Models should be mounted read-only so a runaway process cannot
        # corrupt the weight files.
        assert ":ro" in compose_text

    def test_resource_limits_set(self, compose_text: str) -> None:
        data = yaml.safe_load(compose_text)
        svc = data["services"]["gatesentinel"]
        limits = svc.get("deploy", {}).get("resources", {}).get("limits", {})
        assert "cpus" in limits
        assert "memory" in limits


# ---------------------------------------------------------------------------
# install.sh
# ---------------------------------------------------------------------------


class TestInstallScript:
    def test_file_exists(self) -> None:
        assert (DEPLOY_DIR / "install.sh").exists()

    def test_bash_shebang(self, install_sh_text: str) -> None:
        assert install_sh_text.startswith("#!/usr/bin/env bash")

    def test_errexit_set(self, install_sh_text: str) -> None:
        # set -e (or -euo pipefail) is mandatory for safe install scripts.
        assert "set -euo pipefail" in install_sh_text

    def test_creates_system_user(self, install_sh_text: str) -> None:
        assert "useradd" in install_sh_text

    def test_installs_openvino(self, install_sh_text: str) -> None:
        assert "openvino" in install_sh_text

    def test_enables_service(self, install_sh_text: str) -> None:
        assert "systemctl enable" in install_sh_text

    def test_installs_tesseract(self, install_sh_text: str) -> None:
        assert "tesseract-ocr" in install_sh_text

    def test_root_check(self, install_sh_text: str) -> None:
        assert "EUID" in install_sh_text

    def test_references_repo_url(self, install_sh_text: str) -> None:
        assert "razerblade09/GateSentinel" in install_sh_text


# ---------------------------------------------------------------------------
# setup_logging helper (added in Milestone 10, used by deploy CLIs)
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_importable(self) -> None:
        from alpr.utils.logging import setup_logging  # noqa: PLC0415
        assert callable(setup_logging)

    def test_accepts_level_string(self) -> None:
        import logging  # noqa: PLC0415
        from alpr.utils.logging import setup_logging  # noqa: PLC0415

        # Reset root logger handlers so the test is idempotent.
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        try:
            root.handlers.clear()
            setup_logging(level="WARNING")
            assert root.level == logging.WARNING
        finally:
            root.handlers = original_handlers
            root.level = original_level

    def test_does_not_duplicate_handlers(self) -> None:
        import logging  # noqa: PLC0415
        from alpr.utils.logging import setup_logging  # noqa: PLC0415

        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        try:
            root.handlers.clear()
            setup_logging()
            count_after_first = len(root.handlers)
            setup_logging()
            # Second call must not add another handler (handlers exist already).
            assert len(root.handlers) == count_after_first
        finally:
            root.handlers = original_handlers
            root.level = original_level

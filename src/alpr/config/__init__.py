"""Configuration loading and validation."""

from alpr.config.loader import ConfigError, find_project_root, load_config
from alpr.config.schema import AppConfig

__all__ = ["AppConfig", "ConfigError", "find_project_root", "load_config"]

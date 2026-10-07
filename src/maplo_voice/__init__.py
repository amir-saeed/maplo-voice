"""Maplo Voice — real-time voice agent and spoken-English assessment service."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("maplo-voice")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.0.0+local"

__all__ = ["__version__"]

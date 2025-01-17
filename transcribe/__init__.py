"""Open wsl paths in windows."""

import importlib.metadata

__version__ = importlib.metadata.version("transcribe")

from .cli import app as app

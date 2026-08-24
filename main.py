"""Thin ASGI entrypoint for the unified Northstar Commerce & Fitzy Sales Agent service."""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
clothing_app_dir = root_dir / "clothing_app"

if str(clothing_app_dir) not in sys.path:
    sys.path.insert(0, str(clothing_app_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

for key in list(sys.modules.keys()):
    if key == "app" or key.startswith("app."):
        del sys.modules[key]

from clothing_app.app.main import app

__all__ = ["app"]

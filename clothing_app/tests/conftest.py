"""Pytest configuration for clothing_app tests."""
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent.parent)
clothing_app_dir = str(Path(__file__).resolve().parent.parent)

if clothing_app_dir not in sys.path:
    sys.path.insert(0, clothing_app_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Ensure 'app' module maps to clothing_app
for key in list(sys.modules.keys()):
    if key == "app" or key.startswith("app."):
        del sys.modules[key]

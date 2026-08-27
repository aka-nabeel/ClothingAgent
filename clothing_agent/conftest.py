"""Pytest configuration for clothing_agent tests."""
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
clothing_app_dir = str(Path(__file__).resolve().parent.parent / "clothing_app")

if clothing_app_dir not in sys.path:
    sys.path.insert(0, clothing_app_dir)
if root_dir not in sys.path:
    sys.path.insert(1, root_dir)


"""Clothing application API used by the AI shopping-agent demo."""

import sys

# Bind sys.modules['app'] to clothing_app.app so internal app.* imports resolve cleanly
sys.modules["app"] = sys.modules[__name__]

import sys
import os

root_dir = os.path.abspath(os.path.dirname(__file__))
clothing_app_dir = os.path.join(root_dir, "clothing_app")

# Remove any existing references to clothing_app or root from sys.path
sys.path = [p for p in sys.path if p not in (root_dir, clothing_app_dir)]

# Insert clothing_app at index 0 so 'import app' always resolves to clothing_app/app
sys.path.insert(0, clothing_app_dir)
sys.path.insert(1, root_dir)

# Ensure sys.modules['app'] points to clothing_app/app
if "app" in sys.modules and getattr(sys.modules["app"], "__file__", "").find("clothing_agent") != -1:
    for key in list(sys.modules.keys()):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]

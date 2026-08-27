import sys
import os

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
clothing_app_dir = os.path.join(root_dir, "clothing_app")

if clothing_app_dir in sys.path:
    sys.path.remove(clothing_app_dir)
sys.path.insert(0, clothing_app_dir)

if root_dir not in sys.path:
    sys.path.insert(1, root_dir)

if "app" in sys.modules:
    app_mod = sys.modules["app"]
    if getattr(app_mod, "__file__", "").find("clothing_agent") != -1 or getattr(app_mod, "__path__", [""])[0].find("clothing_agent") != -1:
        for key in list(sys.modules.keys()):
            if key == "app" or key.startswith("app."):
                del sys.modules[key]

from pathlib import Path
import sys

# Ensure the hfoGUI package is importable when launched from the repo root
ROOT = Path(__file__).resolve().parent
HFO_GUI_DIR = ROOT / "hfoGUI"
if str(HFO_GUI_DIR) not in sys.path:
    sys.path.insert(0, str(HFO_GUI_DIR))

from hfoGUI.main import run

if __name__ == "__main__":
    run()

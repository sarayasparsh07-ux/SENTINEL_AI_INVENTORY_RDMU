"""Convenience launcher for the SENTINEL AI dashboard.

    python SENTINEL_AI_Inventory_Dashboard.py

is equivalent to running, from the project root:

    streamlit run dashboard/app.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")],
                             cwd=str(ROOT)))

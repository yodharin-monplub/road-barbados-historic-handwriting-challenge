#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_htr.model36_export import main

if __name__ == "__main__":
    main()

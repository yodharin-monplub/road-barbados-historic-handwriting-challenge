from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from road_htr.vlm_cli import infer_main
if __name__=="__main__":
    infer_main(ROOT/"outputs/model107/config_fold0.yaml")

# Reuse the fixed held-out evaluator with the conservative v2 checkpoint.

from pathlib import Path

import eval_v2


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
eval_v2.V2_CHECKPOINT = ROOT / "checkpoints" / "v2" / "best_conservative.pt"
eval_v2.OUTPUT_FILE = HERE / "eval_outputs_conservative.txt"


if __name__ == "__main__":
    eval_v2.main()

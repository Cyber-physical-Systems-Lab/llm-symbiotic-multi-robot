import argparse
import json
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # TODO: 这里初始化 TA-RWARE 环境、LLM agent、跑 episodes
    # 你先用假数据验证 pipeline 通
    results = {
        "seed": args.seed,
        "episodes": args.episodes,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "pick_rate": 0.0,
            "completion_time": 0.0,
            "deadlocks": 0
        }
    }

    outfile = outdir / f"result_seed{args.seed}.json"
    outfile.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    main()

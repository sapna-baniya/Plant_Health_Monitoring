import argparse
import csv
import itertools
import os
import subprocess
import sys
import tempfile
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maskrcnn_config.yaml")
    parser.add_argument("--max_trials", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5, help="Short search epochs per trial")
    args = parser.parse_args()

    base = yaml.safe_load(open(args.config))
    search_space = {
        "learning_rate": [0.001, 0.0005, 0.0001],
        "batch_size": [1, 2],
        "weight_decay": [0.0001, 0.0005],
    }
    combos = list(itertools.product(search_space["learning_rate"], search_space["batch_size"], search_space["weight_decay"]))[:args.max_trials]

    os.makedirs("outputs/hparam", exist_ok=True)
    results_path = "outputs/hparam/random_search_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "learning_rate", "batch_size", "weight_decay", "best_val_loss", "log_path"])
        writer.writeheader()

        for trial, (lr, bs, wd) in enumerate(combos, 1):
            cfg = dict(base)
            cfg["training"] = dict(base["training"])
            cfg["training"]["learning_rate"] = lr
            cfg["training"]["batch_size"] = bs
            cfg["training"]["weight_decay"] = wd
            cfg["training"]["epochs"] = args.epochs
            cfg["training"]["output_dir"] = f"outputs/hparam/trial_{trial}"
            os.makedirs(cfg["training"]["output_dir"], exist_ok=True)

            tmp_config = f"outputs/hparam/trial_{trial}_config.yaml"
            with open(tmp_config, "w") as cf:
                yaml.safe_dump(cfg, cf)

            print(f"\n=== Trial {trial}: lr={lr}, batch={bs}, wd={wd} ===")
            subprocess.run([sys.executable, "src/train_maskrcnn.py", "--config", tmp_config], check=True)

            log_path = os.path.join(cfg["training"]["output_dir"], "training_log.csv")
            best_val = None
            if os.path.exists(log_path):
                import pandas as pd
                log = pd.read_csv(log_path)
                best_val = float(log["val_loss"].min())
            writer.writerow({
                "trial": trial,
                "learning_rate": lr,
                "batch_size": bs,
                "weight_decay": wd,
                "best_val_loss": best_val,
                "log_path": log_path,
            })
            f.flush()
    print(f"Saved random search results to {results_path}")


if __name__ == "__main__":
    main()

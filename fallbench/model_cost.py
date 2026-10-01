"""One-off: ST-GCN cost per forward pass (FLOPs/MACs, parameters).

    python -m fallbench.model_cost [--weights weights/tsstg-model.pth]

MediaPipe is a TFLite graph and cannot be profiled this way; its cost is
reported in time and CPU-ms instead.
"""
import argparse
import json
import os

import torch

from .classifier import STGCNClassifier
from .config import DEFAULT_WEIGHTS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", default="results/model_cost.json")
    a = ap.parse_args(argv)

    torch.set_num_threads(1)
    clf = STGCNClassifier(a.weights)
    pts, mot = clf.dummy_inputs()
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        with torch.inference_mode():
            clf.model((pts, mot))
    flops = fc.get_total_flops()
    cost = {
        "stgcn_flops_per_pass": flops,
        "stgcn_gflops_per_pass": flops / 1e9,
        "stgcn_gmacs_per_pass": flops / 2e9,
        "stgcn_params": sum(p.numel() for p in clf.model.parameters()),
        "note": "FLOPs counted by torch FlopCounterMode (conv/matmul); MACs = FLOPs/2",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(cost, f, indent=2)
    print(json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()

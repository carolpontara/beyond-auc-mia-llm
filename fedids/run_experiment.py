"""Run one federated training configuration with the Flower simulation engine."""
import argparse
import json
import os

import numpy as np
from flwr.simulation import run_simulation

from common import dirichlet_partition, load_pool, member_indices
from fl_app import client_app, make_server_app

PARTITION_SEED = 42


def partition_path(work, dataset, alpha, clients):
    path = os.path.join(work, "partitions", f"{dataset}_a{alpha}_k{clients}.npz")
    if not os.path.exists(path):
        _, y, split = load_pool(os.path.join(work, "data", f"{dataset}.npz"))
        client = dirichlet_partition(y[member_indices(split)], clients, alpha, PARTITION_SEED)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, client=client)
    return path


def run_dir(work, dataset, alpha, agg, seed):
    return os.path.join(work, "runs", dataset, f"a{alpha}", agg, f"s{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.expanduser("~/fedids_work"))
    ap.add_argument("--dataset", default="cicids2017")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--agg", default="fedavg", choices=["fedavg", "fedprox", "scaffold", "fedadam"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--server-lr", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=1e-3)
    a = ap.parse_args()
    pool = os.path.join(a.work, "data", f"{a.dataset}.npz")
    cfg = {"dataset": a.dataset, "alpha": a.alpha, "agg": a.agg, "seed": a.seed, "clients": a.clients,
           "rounds": a.rounds, "lr": a.lr, "batch_size": a.batch_size, "local_epochs": a.local_epochs,
           "mu": a.mu, "server_lr": a.server_lr, "tau": a.tau, "pool": pool,
           "part": partition_path(a.work, a.dataset, a.alpha, a.clients),
           "outdir": run_dir(a.work, a.dataset, a.alpha, a.agg, a.seed)}
    os.makedirs(cfg["outdir"], exist_ok=True)
    with open(os.path.join(cfg["outdir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    run_simulation(server_app=make_server_app(cfg), client_app=client_app, num_supernodes=a.clients,
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0},
                                   "init_args": {"num_cpus": 2, "include_dashboard": False}})


if __name__ == "__main__":
    main()

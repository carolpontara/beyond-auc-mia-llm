"""Flower ClientApp and ServerApp for the federated IDS audit.

Aggregators: FedAvg, FedProx and FedAdam use Flower's message-based strategies;
SCAFFOLD (option II control variates) is implemented on top of Flower's FedAvg.
Every strategy stores the global model every TRAJECTORY_EVERY rounds and, at the
CHECKPOINT_ROUNDS, the broadcast model together with each client's local model.
"""
import json
import os
import time

import numpy as np
import torch
from torch import nn
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAdam, FedAvg, FedProx
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords
from sklearn.metrics import accuracy_score, f1_score

from common import CHECKPOINT_ROUNDS, TRAJECTORY_EVERY, build_model, client_data, load_pool

client_app = ClientApp()


@client_app.train()
def train(msg: Message, context: Context) -> Message:
    torch.set_num_threads(1)
    cfg = msg.content["config"]
    pid = int(context.node_config["partition-id"])
    pool, part, agg = str(cfg["pool"]), str(cfg["part"]), str(cfg["agg"])
    lr, bs, epochs = float(cfg["lr"]), int(cfg["batch-size"]), int(cfg["local-epochs"])
    rnd, seed = int(cfg["server-round"]), int(cfg["seed"])
    X, y = client_data(pool, part, pid)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    model = build_model(pool)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gen = torch.Generator().manual_seed(seed * 1_000_003 + rnd * 101 + pid)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    mu = float(cfg["proximal-mu"]) if agg == "fedprox" else 0.0
    warm = 0
    if agg == "scaffold":
        warm = int("c_local" in context.state)
        c_glob = msg.content["c_global"].to_torch_state_dict()
        c_loc = (context.state["c_local"].to_torch_state_dict() if "c_local" in context.state
                 else {k: torch.zeros_like(v) for k, v in w0.items()})
    steps, total = 0, 0.0
    params = dict(model.named_parameters())
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y), bs):
            b = perm[i:i + bs]
            loss = ce(model(X[b]), y[b])
            if mu > 0:
                loss = loss + mu / 2 * sum(((p - w0[k]) ** 2).sum() for k, p in params.items())
            opt.zero_grad()
            loss.backward()
            if agg == "scaffold":
                for k, p in params.items():
                    p.grad += c_glob[k] - c_loc[k]
            opt.step()
            steps += 1
            total += float(loss.detach()) * len(b)
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"num-examples": int(len(y)), "train-loss": total / (epochs * len(y)),
                                 "pid": pid, "state-warm": warm}),
    })
    if agg == "scaffold":
        wl = model.state_dict()
        c_new = {k: c_loc[k] - c_glob[k] + (w0[k] - wl[k]) / (steps * lr) for k in w0}
        content["c_delta"] = ArrayRecord({k: c_new[k] - c_loc[k] for k in w0})
        context.state["c_local"] = ArrayRecord(c_new)
    return Message(content=content, reply_to=msg)


def _pid(m):
    return -1 if m.has_error() else int(m.content["metrics"]["pid"])


class AuditMixin:
    """Stores broadcast models and per-client local models at checkpoint rounds."""

    outdir = None

    def configure_train(self, server_round, arrays, config, grid):
        self._sent = {k: a.numpy().copy() for k, a in arrays.items()}
        return super().configure_train(server_round, arrays, config, grid)

    def _store(self, server_round, replies):
        if server_round not in CHECKPOINT_ROUNDS:
            return
        out = {f"sent/{k}": v for k, v in self._sent.items()}
        for m in replies:
            if m.has_error():
                continue
            p = _pid(m)
            for k, a in m.content["arrays"].items():
                out[f"client{p}/{k}"] = a.numpy()
        np.savez_compressed(os.path.join(self.outdir, f"updates_r{server_round}.npz"), **out)

    def aggregate_train(self, server_round, replies):
        replies = sorted(list(replies), key=_pid)
        self._store(server_round, replies)
        return super().aggregate_train(server_round, replies)


class AuditFedAvg(AuditMixin, FedAvg):
    pass


class AuditFedProx(AuditMixin, FedProx):
    pass


class AuditFedAdam(AuditMixin, FedAdam):
    pass


class AuditScaffold(AuditMixin, FedAvg):
    """SCAFFOLD (Karimireddy et al., 2020), option II, full participation, server lr 1."""

    c_global = None

    def configure_train(self, server_round, arrays, config, grid):
        if self.c_global is None:
            self.c_global = {k: np.zeros_like(a.numpy()) for k, a in arrays.items()}
        msgs = list(super().configure_train(server_round, arrays, config, grid))
        cg = ArrayRecord({k: torch.from_numpy(v.copy()) for k, v in self.c_global.items()})
        for m in msgs:
            m.content["c_global"] = cg
        return msgs

    def aggregate_train(self, server_round, replies):
        replies = sorted(list(replies), key=_pid)
        self._store(server_round, replies)
        valid, _ = self._check_and_log_replies(replies, is_train=True, validate=False)
        if not valid:
            return None, None
        n = np.array([float(m.content["metrics"]["num-examples"]) for m in valid])
        wts = n / n.sum()
        new = {k: sum(w * m.content["arrays"][k].numpy() for w, m in zip(wts, valid)) for k in self._sent}
        for k in self.c_global:
            self.c_global[k] = self.c_global[k] + np.mean(
                [m.content["c_delta"][k].numpy() for m in valid], axis=0)
        arrays = ArrayRecord({k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in new.items()})
        metrics = aggregate_metricrecords([m.content for m in valid], self.weighted_by_key)
        return arrays, metrics


STRATEGIES = {"fedavg": AuditFedAvg, "fedprox": AuditFedProx, "fedadam": AuditFedAdam,
              "scaffold": AuditScaffold}


def make_server_app(cfg):
    server_app = ServerApp()

    @server_app.main()
    def main(grid: Grid, context: Context) -> None:
        outdir = cfg["outdir"]
        os.makedirs(outdir, exist_ok=True)
        torch.manual_seed(cfg["seed"])
        model = build_model(cfg["pool"])
        common = dict(fraction_train=1.0, fraction_evaluate=0.0, min_train_nodes=cfg["clients"],
                      min_available_nodes=cfg["clients"])
        agg = cfg["agg"]
        if agg == "fedprox":
            strategy = AuditFedProx(proximal_mu=cfg["mu"], **common)
        elif agg == "fedadam":
            strategy = AuditFedAdam(eta=cfg["server_lr"], eta_l=cfg["lr"], tau=cfg["tau"], **common)
        else:
            strategy = STRATEGIES[agg](**common)
        strategy.outdir = outdir
        X, y, split = load_pool(cfg["pool"])
        Xt, yt = torch.from_numpy(X[split == "test"]), y[split == "test"]
        history = []

        def evaluate_fn(rnd, arrays):
            sd = arrays.to_torch_state_dict()
            if rnd % TRAJECTORY_EVERY == 0:
                torch.save(sd, os.path.join(outdir, f"global_r{rnd}.pt"))
            if rnd % TRAJECTORY_EVERY != 0 and rnd != 1:
                return None
            model.load_state_dict(sd)
            with torch.no_grad():
                pred = model(Xt).argmax(1).numpy()
            rec = {"round": rnd, "acc": float(accuracy_score(yt, pred)),
                   "macro_f1": float(f1_score(yt, pred, average="macro")),
                   "per_class_f1": f1_score(yt, pred, average=None).tolist(), "time": time.time()}
            history.append(rec)
            return MetricRecord({"acc": rec["acc"], "macro-f1": rec["macro_f1"]})

        train_cfg = ConfigRecord({"pool": cfg["pool"], "part": cfg["part"], "agg": agg, "lr": cfg["lr"],
                                  "batch-size": cfg["batch_size"], "local-epochs": cfg["local_epochs"],
                                  "seed": cfg["seed"]})
        strategy.start(grid=grid, initial_arrays=ArrayRecord(model.state_dict()),
                       num_rounds=cfg["rounds"], train_config=train_cfg, evaluate_fn=evaluate_fn)
        with open(os.path.join(outdir, "utility.json"), "w") as f:
            json.dump({"config": cfg, "history": history}, f, indent=1)

    return server_app

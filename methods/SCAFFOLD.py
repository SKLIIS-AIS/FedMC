import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entities.SCAFFOLD import SCAFFOLDClient, SCAFFOLDServer
from utils.gen_dataset import distribute_dataset
from utils.metric import get_accuracy


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_round_lr(args, round_index):
    base_lr = float(args["lr"])
    min_lr = float(args.get("min_lr", 0.0))
    schedule = str(args.get("lr_schedule", "constant")).lower()

    if schedule != "cosine":
        return base_lr

    decay_rounds = max(
        int(args.get("lr_decay_rounds", args["rounds"])),
        1,
    )
    progress = min(float(round_index) / decay_rounds, 1.0)

    return min_lr + 0.5 * (base_lr - min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


if __name__ == "__main__":
    config_path = PROJECT_ROOT / "configs" / "SCAFFOLD.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        args = yaml.safe_load(file)

    set_seed(int(args.get("seed", 0)))

    print(f"Config: {config_path}")
    print(json.dumps(args, indent=4))

    client_train_set, client_test_set, global_test_set = distribute_dataset(
        args["dataset"],
        args["client_num"],
        args["partition"],
        args["alpha"],
        args["seed"],
    )

    clients = [
        SCAFFOLDClient(
            client_id,
            args,
            client_train_set[client_id],
            client_test_set[client_id],
        )
        for client_id in range(args["client_num"])
    ]

    for client in clients:
        client.update_class_mask()

    server = SCAFFOLDServer(args)
    server.get_client_data_size(clients)

    missing_counts = [
        int((client.class_mask == 0).sum())
        for client in clients
    ]
    print("Missing classes per client (all clients):")
    print(missing_counts)
    print(f"Average missing: {np.mean(missing_counts):.2f}")

    rounds = int(args["rounds"])
    eval_interval = max(int(args.get("eval_interval", 10)), 1)

    best_global_acc = float("-inf")
    best_round = 0

    for round_index in range(rounds):
        args["current_lr"] = get_round_lr(
            args,
            round_index,
        )

        selected_clients = server.select_clients(clients)
        if not selected_clients:
            print(f"Round {round_index + 1}: No clients selected, skip.")
            continue

        server.send_model(selected_clients)

        for client in selected_clients:
            client.train(server.global_c)

        server.aggregate(selected_clients)

        for client in selected_clients:
            client.clear_round_state()

        current_round = round_index + 1
        should_evaluate = (
            current_round % eval_interval == 0
            or current_round == rounds
        )
        if not should_evaluate:
            continue

        global_acc = get_accuracy(
            server.model,
            global_test_set,
            args["device"],
        )

        # SCAFFOLD also evaluates one shared global model at every client.
        server.send_model(clients)
        local_accs = [
            client.local_test()
            for client in clients
        ]
        missing_accs = [
            client.missing_class_test(global_test_set)
            for client in clients
        ]

        mean_local = float(np.mean(local_accs))
        mean_missing = float(np.mean(missing_accs))

        if global_acc > best_global_acc:
            best_global_acc = global_acc
            best_round = current_round

        print(
            f"Round {current_round:4d} | "
            f"LR: {args['current_lr']:.6f} | "
            f"Local: {mean_local:.4f} | "
            f"Global: {global_acc:.4f} | "
            f"Missing: {mean_missing:.4f} | "
            f"Best: {best_global_acc:.4f} @ {best_round}",
            flush=True,
        )

    if best_round == 0:
        raise RuntimeError(
            "No evaluation was performed. Check rounds/eval_interval."
        )

    server.send_model(clients)
    print("\n" + "=" * 60)
    print(
        "Final Evaluation... "
        f"Best Global Accuracy: {best_global_acc:.4f} "
        f"at Round {best_round}"
    )
    server.last_round_evaluate(
        clients,
        global_test_set,
    )
    print("SCAFFOLD Training Complete!")

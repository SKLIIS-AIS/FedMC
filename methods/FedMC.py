import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entities.FedMC import FedMCClient, FedMCServer
from utils.gen_dataset import distribute_dataset
from utils.metric import get_accuracy


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_config_path(config_value):
    path = Path(config_value)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return PROJECT_ROOT / path


def update_round_learning_rates(args, round_index):
    schedule = str(args.get("lr_schedule", "cosine")).lower()
    rep_base = float(args["rep_lr_base"])
    clf_base = float(args["clf_lr_base"])

    if schedule == "cosine":
        progress = round_index / max(int(args["rounds"]), 1)
        factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        rep_lr = rep_base * factor
        clf_lr = clf_base * factor
    elif schedule == "multistep":
        decay_count = sum(
            round_index >= int(milestone)
            for milestone in args.get("lr_decay_milestones", [])
        )
        rep_lr = rep_base * float(args.get("rep_lr_decay", 1.0)) ** decay_count
        clf_lr = clf_base * float(args.get("clf_lr_decay", 1.0)) ** decay_count
    elif schedule == "constant":
        rep_lr = rep_base
        clf_lr = clf_base
    else:
        raise ValueError(
            f"Unknown lr_schedule={schedule!r}; expected cosine, "
            "multistep or constant."
        )

    args["rep_lr"] = float(rep_lr)
    args["clf_lr"] = float(clf_lr)
    args["balanced_clf_lr"] = 0.5 * float(clf_lr)


def get_cli_args():
    parser = argparse.ArgumentParser(
        description="FedMC full model and component ablations"
    )
    parser.add_argument(
        "--config",
        default="configs/FedMC.yaml",
        help="YAML path relative to the FedMC project",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for repeated experiments",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = get_cli_args()
    config_path = resolve_config_path(cli_args.config)
    with config_path.open("r", encoding="utf-8") as file:
        args = yaml.safe_load(file)
    if cli_args.seed is not None:
        args["seed"] = int(cli_args.seed)

    print(f"Config: {config_path}")
    print(json.dumps(args, indent=4))
    set_random_seed(int(args["seed"]))

    client_train_set, client_test_set, global_test_set = distribute_dataset(
        args["dataset"],
        args["client_num"],
        args["partition"],
        args["alpha"],
        args["seed"],
    )

    clients = [
        FedMCClient(
            client_id,
            args,
            client_train_set[client_id],
            client_test_set[client_id],
        )
        for client_id in range(args["client_num"])
    ]
    server = FedMCServer(args)
    server.get_client_data_size(clients)

    clf_keys = list(server.model.state_dict().keys())[-2:]
    server.clf_keys = clf_keys

    server.model.eval()
    with torch.no_grad():
        _, dummy_feature = server.model(
            torch.randn(2, 3, 32, 32).to(args["device"]),
            return_feat=True,
        )
    feature_dim = dummy_feature.shape[1]
    server.global_protos = [
        torch.zeros(feature_dim, device=args["device"])
        for _ in range(args["num_classes"])
    ]

    for client in clients:
        client.clf_keys = clf_keys
        client.p_clf_params = [
            parameter.data.clone()
            for parameter in client.get_clf_parameters()
        ]
        client.update_label_distribution()
        client.global_protos = [
            proto.clone() for proto in server.global_protos
        ]

    missing_counts = [
        int((client.class_mask == 0).sum()) for client in clients
    ]
    print("Missing classes per client (all clients):")
    print(missing_counts)
    print(f"Average missing: {np.mean(missing_counts):.2f}")

    print(
        "Ablation switches:",
        {
            "proto_loss": args.get("use_proto_loss", True),
            "classifier_clustering": args.get(
                "use_classifier_clustering", True
            ),
            "missing_completion": args.get(
                "use_missing_class_completion", True
            ),
            "non_selected_update": args.get(
                "update_non_selected_clients", True
            ),
            "non_selected_proto_sync": args.get(
                "sync_nonselected_prototypes", True
            ),
            "missing_gradient_mask": args.get(
                "mask_missing_class_gradients", True
            ),
        },
    )

    if args.get("use_differential_privacy", False):
        print(
            f"DP enabled (epsilon={args.get('dp_epsilon', 8.0)})"
        )

    # All FedMC variants use exactly the same evaluation cadence.
    # With eval_interval=10, checkpoints are evaluated at rounds
    # 10, 20, 30, ..., and at the final round if it is not a multiple of 10.
    eval_interval = max(int(args.get("eval_interval", 10)), 1)
    evaluation_history = []

    for round_index in range(int(args["rounds"])):
        update_round_learning_rates(args, round_index)

        selected_clients = server.select_clients(clients)
        if not selected_clients:
            print(f"Round {round_index}: No clients selected, skip.")
            continue

        server.send_rep_params(selected_clients)

        classifier_params_for_clustering = {}
        for client in selected_clients:
            client.update_label_distribution()

            # Keep the inherited classifier-only balanced stage fixed in every
            # ablation. It is not treated as a FedMC contribution.
            if int(args["balanced_epochs"]) > 0:
                client.balance_train()
                classifier_params_for_clustering[client.id] = [
                    parameter.data.clone()
                    for parameter in client.get_clf_parameters()
                ]

            if not args["clustered_protos"]:
                client.global_protos = [
                    proto.clone() for proto in server.global_protos
                ]

            client.train_with_protos(round_index)

            if client.id not in classifier_params_for_clustering:
                classifier_params_for_clustering[client.id] = [
                    parameter.data.clone()
                    for parameter in client.get_clf_parameters()
                ]

        server.aggregate_rep(selected_clients)

        for client in selected_clients:
            client.local_protos = client.get_local_protos(client.model)
        server.aggregate_protos(selected_clients)

        # Full FedMC and the already-running w/o ProtoAlign keep the original
        # all-client prototype synchronization.  The w/o NonSelected and
        # w/o MissingHandling controls restrict this update to selected clients.
        prototype_recipients = (
            clients
            if args.get("sync_nonselected_prototypes", True)
            else selected_clients
        )
        for client in prototype_recipients:
            if args.get("use_missing_class_completion", True):
                client.global_protos = [
                    proto.clone() for proto in server.global_protos
                ]
            else:
                # w/o MissingHandling: synchronize prototypes only for labels
                # that actually exist on this client.
                for label, proto in enumerate(server.global_protos):
                    if client.class_mask[label] == 1:
                        client.global_protos[label] = proto.clone()
        server.send_rep_params(selected_clients)

        selected_class_masks = {
            client.id: client.class_mask for client in selected_clients
        }
        if args.get("use_classifier_clustering", True):
            if server.args["oracle"]:
                label_merged_dict = server.oracle_merging(
                    round_index,
                    [client.id for client in selected_clients],
                )
            else:
                label_merged_dict = server.merge_classifiers(
                    classifier_params_for_clustering,
                    selected_class_masks,
                )
        else:
            # Default FedMC path: no DBSCAN. For every class, all selected
            # clients containing that class form one class-wise aggregation
            # group. DBSCAN is enabled only by FedMC_with_DBSCAN.yaml.
            label_merged_dict = {}
            for label in range(args["num_classes"]):
                valid_ids = [
                    client.id
                    for client in selected_clients
                    if selected_class_masks[client.id][label] == 1
                ]
                label_merged_dict[label] = (
                    [valid_ids] if valid_ids else []
                )

        for label, merged_id_groups in label_merged_dict.items():
            for client_ids in merged_id_groups:
                group = [
                    client
                    for client in selected_clients
                    if client.id in client_ids
                ]
                aggregated_params = server.aggregate_label_params(
                    label, group
                )
                aggregated_proto = server.aggregate_label_protos(
                    label, group
                )
                if aggregated_params is None or aggregated_proto is None:
                    continue
                for client in group:
                    client.set_label_params(label, aggregated_params)
                    client.global_protos[label] = aggregated_proto.clone()

        global_label_params = {}
        global_label_protos = {}
        for label in range(args["num_classes"]):
            valid_clients = [
                client
                for client in selected_clients
                if selected_class_masks[client.id][label] == 1
            ]
            if not valid_clients:
                continue
            aggregated_params = server.aggregate_label_params(
                label, valid_clients
            )
            aggregated_proto = server.aggregate_label_protos(
                label, valid_clients
            )
            if aggregated_params is not None:
                global_label_params[label] = aggregated_params
            if aggregated_proto is not None:
                global_label_protos[label] = aggregated_proto

        if args.get("use_missing_class_completion", True):
            for client in selected_clients:
                for label in range(args["num_classes"]):
                    if (
                        selected_class_masks[client.id][label] == 0
                        and label in global_label_params
                        and label in global_label_protos
                    ):
                        client.set_label_params(
                            label, global_label_params[label]
                        )
                        client.global_protos[label] = (
                            global_label_protos[label].clone()
                        )

            if args.get("update_non_selected_clients", True):
                selected_ids = {
                    client.id for client in selected_clients
                }
                non_selected_clients = [
                    client
                    for client in clients
                    if client.id not in selected_ids
                ]
                for client in non_selected_clients:
                    for label in range(args["num_classes"]):
                        if (
                            client.class_mask[label] == 0
                            and label in global_label_params
                            and label in global_label_protos
                        ):
                            client.set_label_params(
                                label, global_label_params[label]
                            )
                            client.global_protos[label] = (
                                global_label_protos[label].clone()
                            )

        for client in selected_clients:
            client.p_clf_params = [
                parameter.data.clone()
                for parameter in client.get_clf_parameters()
            ]

        # Evaluate all variants with the same state: the latest global
        # representation plus each client's personalized classifier.
        current_round = round_index + 1
        should_evaluate = (
            current_round % eval_interval == 0
            or current_round == int(args["rounds"])
        )
        if should_evaluate:
            server.send_rep_params(clients)

            local_accuracies = [
                client.local_test() for client in clients
            ]
            global_accuracies = [
                get_accuracy(
                    client.model, global_test_set, args["device"]
                )
                for client in clients
            ]
            missing_accuracies = [
                client.missing_class_test(global_test_set)
                for client in clients
            ]

            mean_local = float(np.mean(local_accuracies))
            mean_global = float(np.mean(global_accuracies))
            mean_missing = float(np.mean(missing_accuracies))
            evaluation_history.append(
                (
                    current_round,
                    mean_local,
                    mean_global,
                    mean_missing,
                )
            )

            print(
                f"Round {current_round:4d} | "
                f"Local: {mean_local:.4f} | "
                f"Global: {mean_global:.4f} | "
                f"Missing: {mean_missing:.4f}",
                flush=True,
            )

        if (
            round_index % 100 == 0
            and args.get("use_differential_privacy", False)
        ):
            for client in clients[:3]:
                report = client.get_privacy_report()
                if isinstance(report, dict):
                    print(
                        f"Client {client.id}: steps={report['steps']}, "
                        f"epsilon≈{report['epsilon']:.2f}, "
                        f"rough_spent={report['rough_spent']:.4f}"
                    )

    if not evaluation_history:
        raise RuntimeError(
            "No evaluation was performed. Check rounds/eval_interval."
        )

    final_round, final_local, final_global, final_missing = (
        evaluation_history[-1]
    )
    window_size = min(10, len(evaluation_history))
    final_window = np.asarray(
        [metrics[1:] for metrics in evaluation_history[-window_size:]],
        dtype=np.float64,
    )
    window_mean = final_window.mean(axis=0)
    window_std = final_window.std(axis=0)

    print("\n" + "=" * 60)
    print(
        f"Final Evaluation @ Round {final_round}: "
        f"Local={final_local:.4f}, "
        f"Global={final_global:.4f}, "
        f"Missing={final_missing:.4f}"
    )
    print(
        f"Last-{window_size}-Checkpoint Mean: "
        f"Local={window_mean[0]:.4f}, "
        f"Global={window_mean[1]:.4f}, "
        f"Missing={window_mean[2]:.4f}"
    )
    print(
        f"Last-{window_size}-Checkpoint Temporal Std: "
        f"Local={window_std[0]:.4f}, "
        f"Global={window_std[1]:.4f}, "
        f"Missing={window_std[2]:.4f}"
    )

    server.last_round_evaluate(clients, global_test_set)
    print("FedMC Experiment Complete!")

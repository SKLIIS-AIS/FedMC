import yaml
import json
import time
import random
import numpy as np
import torch

from copy import deepcopy
from torch.nn.utils import vector_to_parameters
from entities.FedCCFA import FedCCFAClient, FedCCFAServer
from utils.gen_dataset import distribute_dataset
from utils.metric import get_accuracy

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # read training parameters
    with open("../configs/FedCCFA.yaml", 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print(json.dumps(args, indent=4))

    # initialize clients, server and global model
    client_train_set, client_test_set, global_test_set = distribute_dataset(
        args["dataset"], args["client_num"], args["partition"], args["alpha"], args["seed"]
    )

    clients = []
    for client_id in range(args["client_num"]):
        client = FedCCFAClient(client_id, args, client_train_set[client_id], client_test_set[client_id])
        clients.append(client)

    server = FedCCFAServer(args)
    server.get_client_data_size(clients)

    clf_keys = list(server.model.state_dict().keys())[-2:]

    for client in clients:
        client.clf_keys = clf_keys
        client.p_clf_params = deepcopy(client.get_clf_parameters())
        client.update_label_distribution()
    server.clf_keys = clf_keys

    server.model.eval()
    with torch.no_grad():
        _, features = server.model(
            torch.randn(2, 3, 32, 32).to(args["device"]), True
        )
    feature_dim = features.shape[1]
    server.global_protos = [
        torch.zeros(feature_dim, device=args["device"])
        for _ in range(args["num_classes"])
    ]
    for client in clients:
        client.global_protos = deepcopy(server.global_protos)

    print("Missing classes per client:", [
        int((client.class_mask == 0).sum()) for client in clients
    ])
    empty_client_ids = [client.id for client in clients if len(client.train_set) == 0]
    print("Empty training clients:", empty_client_ids)

    best_global_acc = 0.0
    best_round = 0

    for _round in range(args["rounds"]):
        total_time = 0
        selected_clients = server.select_clients(clients)
        # Under severe Dirichlet skew, some clients can have zero training
        # samples. Keep the original sampling result, but exclude those
        # clients from training and aggregation because they provide no update.
        active_clients = [
            client for client in selected_clients if len(client.train_set) > 0
        ]
        if not active_clients:
            print(
                f"Round {_round + 1:4d} | no non-empty clients selected; skipped",
                flush=True,
            )
            continue

        server.send_rep_params(active_clients)

        balanced_clf_params_dict = {}

        for client in active_clients:
            client.update_label_distribution()

            if args["balanced_epochs"] > 0:
                start_time = time.time()
                client.balance_train()
                balanced_clf_params_dict[client.id] = deepcopy(client.get_clf_parameters())
                end_time = time.time()
                total_time += end_time - start_time

            if not args["clustered_protos"]:
                client.global_protos = deepcopy(server.global_protos)

            client.train_with_protos(_round)
            if args["balanced_epochs"] == 0:
                balanced_clf_params_dict[client.id] = deepcopy(client.get_clf_parameters())

        server.aggregate_rep(active_clients)
        server.aggregate_protos(active_clients)
        server.send_rep_params(active_clients)

        start_time = time.time()

        if server.args["oracle"]:
            label_merged_dict = server.oracle_merging(_round, [c.id for c in active_clients])
        else:
            label_merged_dict = server.merge_classifiers(balanced_clf_params_dict)

        for label, merged_identities in label_merged_dict.items():
            print(label, merged_identities)
            for indices in merged_identities:
                # aggregate personalized classifier parameters according to label distribution
                clients_group = [client for client in active_clients if client.id in indices]
                aggregated_label_params = server.aggregate_label_params(label, clients_group)
                aggregated_label_proto = server.aggregate_label_protos(label, clients_group)
                if aggregated_label_proto is None:
                    continue
                for client in clients_group:
                    client_label_params = [param[label] for name, param in client.model.named_parameters()
                                           if name in clf_keys]
                    vector_to_parameters(aggregated_label_params, client_label_params)
                    client.set_label_params(label, client_label_params)
                    client.global_protos[label] = aggregated_label_proto.clone()
        for client in active_clients:
            client.p_clf_params = deepcopy(client.get_clf_parameters())

        end_time = time.time()
        total_time += end_time - start_time
        # print(total_time)

        # Evaluate every 10 communication rounds (and always on the final round).
        if (_round + 1) % 10 == 0 or (_round + 1) == args["rounds"]:
            server.send_rep_params(clients)
            local_accs = [client.local_test() for client in clients]
            global_accs = [
                get_accuracy(client.model, global_test_set, args["device"])
                for client in clients
            ]
            missing_accs = [
                client.missing_class_test(global_test_set) for client in clients
            ]
            mean_local = float(np.mean(local_accs))
            mean_global = float(np.mean(global_accs))
            mean_missing = float(np.mean(missing_accs))

            if mean_global > best_global_acc:
                best_global_acc = mean_global
                best_round = _round + 1

            print(
                f"Round {_round + 1:4d} | "
                f"Local: {mean_local:.4f} | "
                f"Global: {mean_global:.4f} | "
                f"Missing: {mean_missing:.4f} | "
                f"Best: {best_global_acc:.4f} @ {best_round}",
                flush=True,
            )

    # fine-tune all clients' local models
    # for client in clients:
    #     client.fine_tune()

    server.send_rep_params(clients)
    print("\n" + "=" * 60)
    print(
        f"Final Evaluation... Best Global Accuracy: "
        f"{best_global_acc:.4f} at Round {best_round}"
    )
    server.last_round_evaluate(clients, global_test_set)
    print("FedCCFA Training Complete!")

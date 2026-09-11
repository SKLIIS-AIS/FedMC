import torch
import random
import numpy as np
from typing import Iterator
from torch.optim import SGD
from torch.nn import Parameter, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils.gen_dataset import ClientDataset
from utils.models import get_model
from utils.metric import get_accuracy, save_results

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None


class Client:
    def __init__(self, client_id, args, train_set, test_set):
        self.id = client_id
        self.args = args
        self.train_set = train_set
        self.test_set = test_set
        self.model = get_model(args["dataset"], args["model"]).to(args["device"])
        self.criterion = CrossEntropyLoss().to(args["device"])
        self.local_update = None

    def set_params(self, new_params: Iterator[Parameter]):
        for new_param, local_param in zip(new_params, self.model.parameters()):
            local_param.data = new_param.data.clone()

    def train(self):
        if len(self.train_set) == 0:
            self.local_update = torch.zeros_like(parameters_to_vector(self.model.parameters()))
            return
        self.model.train()
        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)
        init_params = parameters_to_vector(self.model.parameters())
        optimizer = SGD(self.model.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"],
                        momentum=self.args["momentum"])
        for epoch in range(self.args["epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.local_update = parameters_to_vector(self.model.parameters()) - init_params

    def local_test(self):
        return get_accuracy(self.model, self.test_set, self.args["device"])

    def global_test(self, global_test_set):
        return get_accuracy(self.model, global_test_set, self.args["device"])

    def missing_class_test(self, global_test_set):
        if not hasattr(self, 'class_mask') or self.class_mask is None:
            return 0.0
        missing_classes = [i for i, have in enumerate(self.class_mask) if not have]
        if not missing_classes:
            return 0.0
        targets = np.array(global_test_set.targets)
        missing_indices = [i for i, t in enumerate(targets) if t in missing_classes]
        if not missing_indices:
            return 0.0
        subset = ClientDataset(global_test_set, missing_indices, transform=global_test_set.transform)
        return get_accuracy(self.model, subset, self.args["device"])


class Server:
    def __init__(self, args):
        self.args = args
        self.model = get_model(args["dataset"], args["model"]).to(args["device"])
        self.client_data_size = []
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(
                f"../runs/{args['client_num']}/{args['dataset']}/{args['partition']}/{args['model']}/{args['rounds']}"
            )
        else:
            self.writer = None

    def get_client_data_size(self, clients):
        self.client_data_size = [len(client.train_set) for client in clients]

    def select_clients(self, clients):
        return random.sample(clients, int(self.args["client_num"] * self.args["sample_ratio"]))

    def send_params(self, clients):
        # 只同步可学习参数，不同步BN统计量
        for client in clients:
            client.set_params(self.model.parameters())

    def aggregate_by_params(self, clients):
        new_params = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        total_size = 0
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            client_params = parameters_to_vector(client.model.parameters())
            new_params += client_size * client_params
        new_params /= total_size
        new_params.to(self.args["device"])
        vector_to_parameters(new_params, self.model.parameters())

    def aggregate_by_updates(self, clients):
        total_size = 0
        update_sum = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            update_sum += client_size * client.local_update
        aggregated_update = update_sum / total_size
        new_params = parameters_to_vector(self.model.parameters()) + aggregated_update
        vector_to_parameters(new_params, self.model.parameters())

    def local_evaluate(self, clients, cur_round):
        accuracy_list = [client.local_test() for client in clients]
        if self.writer is not None:
            self.writer.add_scalars("local_accuracy_mean", {self.args["algorithm"]: np.array(accuracy_list).mean()}, cur_round)
        return np.array(accuracy_list).mean()

    def global_evaluate(self, clients, global_test_set, cur_round):
        accuracy_list = [client.global_test(global_test_set) for client in clients]
        if self.writer is not None:
            self.writer.add_scalars("global_accuracy_mean", {self.args["algorithm"]: np.array(accuracy_list).mean()}, cur_round)
        return np.array(accuracy_list).mean()

    def last_round_evaluate(self, clients, global_test_set):
        info = []
        local_accuracy_list, global_accuracy_list = [], []
        for client in clients:
            local_accuracy_list.append(client.local_test())
            global_accuracy_list.append(client.global_test(global_test_set))
        for index, client in enumerate(clients):
            self.args.update({
                "client_id": client.id,
                "local_accuracy": local_accuracy_list[index],
                "global_accuracy": global_accuracy_list[index]
            })
            info.append(self.args.copy())
        save_results(info, self.args)

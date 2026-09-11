import torch

from entities.base import Client, Server


class FedAvgClient(Client):
    """Standard FedAvg client using the shared local SGD implementation."""

    def __init__(self, client_id, args, train_set, test_set):
        super().__init__(client_id, args, train_set, test_set)
        self.class_mask = torch.zeros(args["num_classes"], dtype=torch.long)

    def update_class_mask(self):
        if len(self.train_set) == 0:
            self.class_mask.zero_()
            return
        labels = torch.as_tensor(self.train_set.targets, dtype=torch.long)
        counts = torch.bincount(labels, minlength=self.args["num_classes"])
        self.class_mask = (counts > 0).long()


class FedAvgServer(Server):
    """FedAvg server with full-model broadcast and weighted BN aggregation."""

    def send_model(self, clients):
        state = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }
        for client in clients:
            client.model.load_state_dict(state)

    @torch.no_grad()
    def aggregate_bn_buffers(self, clients):
        if not clients:
            return

        state = self.model.state_dict()
        total = float(sum(self.client_data_size[c.id] for c in clients))
        total = max(total, 1.0)

        # 与 FedProx/FedMC 一致：对本轮客户端 BN 浮点统计量直接加权平均。
        for key in state:
            if "running_mean" in key or "running_var" in key:
                aggregated = torch.zeros_like(state[key])
                for client in clients:
                    weight = float(self.client_data_size[client.id]) / total
                    aggregated.add_(
                        client.model.state_dict()[key], alpha=weight
                    )
                state[key].copy_(aggregated)

        self.model.load_state_dict(state)

    def aggregate(self, clients):
        self.aggregate_by_params(clients)
        self.aggregate_bn_buffers(clients)

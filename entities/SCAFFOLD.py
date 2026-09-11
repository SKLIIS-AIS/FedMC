from typing import Iterable, Sequence

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from entities.base import Client, Server


class SCAFFOLDOptimizer(Optimizer):
    """Local SGD corrected by the server and client control variates."""

    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        global_c: Sequence[torch.Tensor],
        client_c: Sequence[torch.Tensor],
    ):
        """Apply grad - c_i + c to every trainable model parameter."""
        control_index = 0

        for group in self.param_groups:
            for parameter in group["params"]:
                if control_index >= len(global_c):
                    raise RuntimeError(
                        "Control variate length is smaller than model parameters."
                    )

                server_control = global_c[control_index]
                client_control = client_c[control_index]
                control_index += 1

                if parameter.grad is None:
                    continue

                gradient = parameter.grad.detach()

                if group["weight_decay"] != 0.0:
                    gradient = gradient.add(
                        parameter.detach(),
                        alpha=group["weight_decay"],
                    )

                # Canonical SCAFFOLD local correction: grad - c_i + c.
                corrected_gradient = (
                    gradient - client_control + server_control
                )

                if group["momentum"] != 0.0:
                    state = self.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = (
                            corrected_gradient.clone()
                        )
                    else:
                        state["momentum_buffer"].mul_(
                            group["momentum"]
                        ).add_(corrected_gradient)

                    corrected_gradient = state["momentum_buffer"]

                parameter.add_(
                    corrected_gradient,
                    alpha=-group["lr"],
                )

        if (
            control_index != len(global_c)
            or control_index != len(client_c)
        ):
            raise RuntimeError(
                "Control variate length does not match model parameters."
            )


class SCAFFOLDClient(Client):
    """SCAFFOLD client compatible with the shared Client implementation."""

    def __init__(self, client_id, args, train_set, test_set):
        super().__init__(client_id, args, train_set, test_set)

        self.class_mask = torch.zeros(
            args["num_classes"],
            dtype=torch.long,
        )

        control_device = args.get("control_device", "cpu")
        self.local_c = [
            torch.zeros_like(parameter, device=control_device)
            for parameter in self.model.parameters()
        ]

        self.local_c_update = []

    def update_class_mask(self):
        """Record which classes are present in this client's train set."""
        if len(self.train_set) == 0:
            self.class_mask.zero_()
            return

        labels = torch.as_tensor(
            self.train_set.targets,
            dtype=torch.long,
        )
        counts = torch.bincount(
            labels,
            minlength=self.args["num_classes"],
        )
        self.class_mask = (counts > 0).long()

    def train(self, global_c: Sequence[torch.Tensor]):
        """Run corrected local SGD and update the client control variate."""
        parameters = list(self.model.parameters())

        if len(global_c) != len(parameters):
            raise RuntimeError(
                "Global control variate does not match the client model."
            )

        if len(self.train_set) == 0:
            self.local_update = torch.zeros_like(
                parameters_to_vector(parameters)
            )
            self.local_c_update = [
                torch.zeros_like(control)
                for control in self.local_c
            ]
            return

        device = self.args["device"]
        current_lr = float(self.args["current_lr"])

        if current_lr <= 0.0:
            raise ValueError(
                "SCAFFOLD requires a positive current_lr."
            )

        # Persistent client controls remain on CPU. Only controls belonging
        # to the selected client are transferred to the training device.
        client_c_device = [
            control.to(device)
            for control in self.local_c
        ]

        initial_parameters = [
            parameter.detach().clone()
            for parameter in parameters
        ]
        initial_vector = (
            parameters_to_vector(parameters)
            .detach()
            .clone()
        )

        train_loader = DataLoader(
            self.train_set,
            batch_size=self.args["batch_size"],
            shuffle=True,
        )

        optimizer = SCAFFOLDOptimizer(
            parameters,
            lr=current_lr,
            momentum=self.args.get("momentum", 0.0),
            weight_decay=self.args.get("weight_decay", 0.0),
        )

        self.model.train()
        local_steps = 0

        for _ in range(int(self.args["epochs"])):
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step(global_c, client_c_device)

                local_steps += 1

        if local_steps == 0:
            raise RuntimeError(
                f"Client {self.id} produced no local optimization steps."
            )

        with torch.no_grad():
            # c_i^new = c_i - c + (x_global - x_local)/(K * eta)
            scale = 1.0 / (local_steps * current_lr)
            new_local_c = []
            local_c_update = []

            for (
                server_control,
                old_client_control,
                initial,
                final,
            ) in zip(
                global_c,
                client_c_device,
                initial_parameters,
                parameters,
            ):
                updated_control = (
                    old_client_control
                    - server_control
                    + (initial - final.detach()) * scale
                )
                control_delta = (
                    updated_control - old_client_control
                )

                new_local_c.append(
                    updated_control.detach().cpu().clone()
                )
                local_c_update.append(
                    control_delta.detach().cpu().clone()
                )

            self.local_c = new_local_c
            self.local_c_update = local_c_update

            self.local_update = (
                parameters_to_vector(parameters).detach()
                - initial_vector
            ).clone()

    def clear_round_state(self):
        """Release temporary tensors after server aggregation."""
        self.local_update = None
        self.local_c_update = []


class SCAFFOLDServer(Server):
    """SCAFFOLD server with damped updates and EMA BN aggregation."""

    def __init__(self, args):
        super().__init__(args)

        # The server model lives on args['device'], so its control variate
        # is kept on the same device for local correction and aggregation.
        self.global_c = [
            torch.zeros_like(parameter)
            for parameter in self.model.parameters()
        ]

    def send_model(self, clients: Iterable[SCAFFOLDClient]):
        """Broadcast the full global model, including BN buffers."""
        state = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

        for client in clients:
            client.model.load_state_dict(state)

    @torch.no_grad()
    def aggregate_model_updates(self, clients):
        """Uniformly average model deltas, as in canonical SCAFFOLD."""
        if not clients:
            return

        update_sum = torch.zeros_like(
            parameters_to_vector(self.model.parameters())
        )

        for client in clients:
            if client.local_update is None:
                raise RuntimeError(
                    f"Client {client.id} has no local model update."
                )
            update_sum.add_(client.local_update)

        average_update = update_sum / float(len(clients))
        global_lr = float(self.args.get("global_lr", 1.0))

        new_parameters = (
            parameters_to_vector(self.model.parameters())
            + global_lr * average_update
        )

        vector_to_parameters(
            new_parameters,
            self.model.parameters(),
        )

    @torch.no_grad()
    def aggregate_bn_buffers(self, clients):
        """Aggregate client BN statistics and update the server using EMA."""
        if not clients:
            return

        server_state = self.model.state_dict()

        total_size = float(
            sum(
                self.client_data_size[client.id]
                for client in clients
            )
        )
        total_size = max(total_size, 1.0)

        beta = float(self.args.get("bn_ema", 0.9))
        beta = min(max(beta, 0.0), 0.9999)

        for key in server_state:
            if "running_mean" in key or "running_var" in key:
                round_bn = torch.zeros_like(server_state[key])

                for client in clients:
                    client_state = client.model.state_dict()
                    weight = (
                        float(self.client_data_size[client.id])
                        / total_size
                    )
                    round_bn.add_(
                        client_state[key],
                        alpha=weight,
                    )

                # server_bn <- beta * server_bn
                #              + (1-beta) * current_round_bn
                server_state[key].mul_(beta).add_(
                    round_bn,
                    alpha=1.0 - beta,
                )

            elif "num_batches_tracked" in key:
                # This buffer is integer-valued and must not be averaged.
                client_values = [
                    client.model.state_dict()[key]
                    for client in clients
                ]
                server_state[key].copy_(
                    torch.stack(client_values).max()
                )

        self.model.load_state_dict(server_state)

    @torch.no_grad()
    def update_global_control(self, clients):
        """Apply the standard partial-participation control update."""
        if not clients:
            return

        # c <- c + (1/N) * sum_i(c_i^new - c_i^old)
        client_num = float(self.args["client_num"])

        for parameter_index, server_control in enumerate(
            self.global_c
        ):
            delta_sum = torch.zeros_like(server_control)

            for client in clients:
                if not client.local_c_update:
                    raise RuntimeError(
                        f"Client {client.id} has no control-variate update."
                    )

                delta_sum.add_(
                    client.local_c_update[parameter_index].to(
                        server_control.device
                    )
                )

            server_control.add_(
                delta_sum,
                alpha=1.0 / client_num,
            )

    def aggregate(self, clients):
        """Aggregate model, BN statistics, and control variates."""
        self.aggregate_model_updates(clients)
        self.aggregate_bn_buffers(clients)
        self.update_global_control(clients)

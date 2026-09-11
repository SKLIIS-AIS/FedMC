import torch
import numpy as np
from torch.optim import SGD
from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader
from entities.base import Client, Server
from utils.differential_privacy import DifferentialPrivacy
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


class FedMCClient(Client):
    def __init__(self, client_id, args, train_set, test_set):
        super().__init__(client_id, args, train_set, test_set)
        self.clf_keys = []
        self.proto_sim = CosineSimilarity(dim=0).to(args["device"])
        self.local_protos = {}
        self.global_protos = None
        self.feat_dim = None
        self.label_distribution = torch.zeros(args["num_classes"], dtype=torch.long)
        self.class_mask = torch.zeros(args["num_classes"], dtype=torch.long)
        self.proto_weight = 0.0
        self.p_clf_params = None

        self.use_dp = args.get("use_differential_privacy", False)
        if self.use_dp:
            self.dp = DifferentialPrivacy(
                epsilon=args.get("dp_epsilon", 8.0),
                delta=args.get("dp_delta", 1e-5),
                max_grad_norm=args.get("dp_max_grad_norm", 1.0),
                noise_multiplier=args.get("dp_noise_multiplier", 0.5)
            )
            self.privacy_accountant = {"steps": 0, "batch_count": 0}
        else:
            self.dp = None

    def _get_feature_dim(self):
        if self.feat_dim is not None:
            return self.feat_dim
        self.model.eval()
        dummy = torch.randn(2, 3, 32, 32).to(self.args["device"])
        with torch.no_grad():
            _, feat = self.model(dummy, return_feat=True)
            self.feat_dim = feat.shape[1]
        return self.feat_dim

    def get_clf_parameters(self):
        return [p for n, p in self.model.named_parameters() if n in self.clf_keys]

    def set_rep_params(self, rep_params_dict):
        state = self.model.state_dict()
        rep_keys = [k for k in state.keys() if k not in self.clf_keys]
        if set(rep_keys) != set(rep_params_dict.keys()):
            missing = set(rep_keys) - set(rep_params_dict.keys())
            extra = set(rep_params_dict.keys()) - set(rep_keys)
            raise RuntimeError(f"表征参数键不匹配！缺失: {missing}, 多余: {extra}")
        for k in rep_keys:
            state[k] = rep_params_dict[k].data.clone()
        self.model.load_state_dict(state)

    def set_label_params(self, label, params):
        clf_params = self.get_clf_parameters()
        if len(clf_params) != len(params):
            raise ValueError(f"分类器参数数量不匹配: {len(clf_params)} vs {len(params)}")
        for p, new_val in zip(clf_params, params):
            p.data[label] = new_val.data.clone()

    def _clip_and_noise(self):
        # 缺失类梯度保护是 FedMC 缺失类处理的一部分。
        # 默认开启以保持原 FedMC / w/o ProtoAlign 的既有训练语义；
        # 仅在 w/o MissingHandling 中显式关闭。
        if self.args.get("mask_missing_class_gradients", True):
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if name in self.clf_keys and p.grad.shape[0] == self.args["num_classes"]:
                        inactive_mask = (self.class_mask == 0).to(p.device)
                        p.grad.data[inactive_mask] = 0.0

        params_to_clip = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if not params_to_clip:
            return
        torch.nn.utils.clip_grad_norm_(params_to_clip, self.dp.max_grad_norm if self.use_dp else 10.0)

        if self.use_dp and self.dp is not None:
            sigma = self.dp.noise_multiplier * self.dp.max_grad_norm / self.args["batch_size"]
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    noise = torch.normal(0.0, sigma, size=p.grad.shape, device=p.device)
                    if (
                        self.args.get("mask_missing_class_gradients", True)
                        and name in self.clf_keys
                        and p.grad.shape[0] == self.args["num_classes"]
                        and len(p.grad.shape) > 1
                    ):
                        inactive_mask = (self.class_mask == 0).to(p.device)
                        noise[inactive_mask] = 0.0
                    p.grad.add_(noise)
            self.privacy_accountant["steps"] += 1
            self.privacy_accountant["batch_count"] += 1

    def balance_train(self):
        if len(self.train_set) == 0:
            return
        self.model.train()
        loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)
        for name, p in self.model.named_parameters():
            p.requires_grad = (name in self.clf_keys)
        opt = SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args["balanced_clf_lr"],
            momentum=self.args["momentum"],
            weight_decay=self.args["weight_decay"]
        )
        for _ in range(self.args["balanced_epochs"]):
            for x, y in loader:
                x, y = x.to(self.args["device"]), y.to(self.args["device"])
                out = self.model(x)
                loss = self.criterion(out, y)
                opt.zero_grad()
                loss.backward()
                self._clip_and_noise()
                opt.step()

    def train_with_protos(self, _round):
        if len(self.train_set) == 0:
            return
        dim = self._get_feature_dim()
        if self.global_protos is None or self.global_protos[0].shape[0] != dim:
            self.global_protos = [torch.zeros(dim, device=self.args["device"]) for _ in range(self.args["num_classes"])]

        loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        # 阶段1：表征训练
        if self.args["rep_epochs"] > 0:
            for name, p in self.model.named_parameters():
                p.requires_grad = (name not in self.clf_keys)
            opt_rep = SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args["rep_lr"],
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"]
            )
            for _ in range(self.args["rep_epochs"]):
                for x, y in loader:
                    x, y = x.to(self.args["device"]), y.to(self.args["device"])
                    out, feat = self.model(x, return_feat=True)
                    proto_loss = torch.tensor(
                        0.0, device=self.args["device"]
                    )
                    if self.args.get("use_proto_loss", True):
                        valid_cnt = 0
                        for i, label in enumerate(y):
                            label_id = int(label.item())
                            proto = self.global_protos[label_id]
                            if proto.abs().sum() > 0:
                                proto_loss += 1 - self.proto_sim(
                                    feat[i], proto
                                )
                                valid_cnt += 1
                        if valid_cnt > 0:
                            proto_loss = proto_loss / valid_cnt

                    loss = self.criterion(out, y)
                    if self.args.get("use_proto_loss", True):
                        loss = loss + self.proto_weight * proto_loss
                    opt_rep.zero_grad()
                    loss.backward()
                    self._clip_and_noise()
                    opt_rep.step()

        # 阶段2：分类器训练
        if self.args["clf_epochs"] > 0:
            for name, p in self.model.named_parameters():
                p.requires_grad = (name in self.clf_keys)
            opt_clf = SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args["clf_lr"],
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"]
            )
            for _ in range(self.args["clf_epochs"]):
                for x, y in loader:
                    x, y = x.to(self.args["device"]), y.to(self.args["device"])
                    out = self.model(x)
                    loss = self.criterion(out, y)
                    opt_clf.zero_grad()
                    loss.backward()
                    self._clip_and_noise()
                    opt_clf.step()

    def update_label_distribution(self):
        if len(self.train_set) == 0:
            return
        labels = torch.tensor(self.train_set.targets)
        dist = torch.bincount(labels, minlength=self.args["num_classes"])
        self.label_distribution = dist
        self.class_mask = (dist > 0).long()
        prob = dist.float() / max(dist.sum(), 1)
        valid = prob[prob > 0]
        entropy = -(valid * torch.log(valid + 1e-10)).sum().item()
        self.proto_weight = max(self.args["lambda"], entropy / self.args["gamma"])

    def get_local_protos(self, model):
        if len(self.train_set) == 0:
            return {}
        loader = DataLoader(self.train_set, batch_size=256, shuffle=False)
        proto_dict = {}
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.args["device"]), y.to(self.args["device"])
                _, feat = model(x, return_feat=True)
                for i, label in enumerate(y):
                    l = int(label.item())
                    proto_dict.setdefault(l, []).append(feat[i])
        return {k: torch.stack(v).mean(0) for k, v in proto_dict.items()}

    def get_privacy_report(self):
        if not self.use_dp:
            return "DP off"
        return {
            "steps": self.privacy_accountant["steps"],
            "batch_count": self.privacy_accountant["batch_count"],
            "epsilon": self.dp.epsilon,
            "delta": self.dp.delta,
            "rough_spent": self.dp.compute_privacy_spent(self.privacy_accountant["steps"])
        }


class FedMCServer(Server):
    def __init__(self, args):
        super().__init__(args)
        self.global_protos = []
        self.clf_keys = []
        self.use_secure_aggregation = args.get("use_secure_aggregation", False)
        self.privacy_aggregator = None

    def send_rep_params(self, clients):
        rep_keys = [k for k in self.model.state_dict().keys() if k not in self.clf_keys]
        rep_params_dict = {k: self.model.state_dict()[k].clone() for k in rep_keys}
        for client in clients:
            client.set_rep_params(rep_params_dict)

    def aggregate_rep(self, clients):
        # 修复：同步所有表征层 state_dict 的浮点 buffer（包括 BN running_mean/var）
        rep_keys = [k for k in self.model.state_dict().keys() if k not in self.clf_keys]
        float_keys = [k for k in rep_keys if self.model.state_dict()[k].dtype in (torch.float32, torch.float64)]
        if not float_keys:
            return

        new_params = {k: torch.zeros_like(self.model.state_dict()[k]) for k in float_keys}
        total = 0
        for c in clients:
            w = self.client_data_size[c.id]
            total += w
            client_state = c.model.state_dict()
            for k in float_keys:
                new_params[k] += w * client_state[k]

        for k in float_keys:
            new_params[k] /= max(total, 1)

        state = self.model.state_dict()
        for k in float_keys:
            state[k] = new_params[k]
        self.model.load_state_dict(state)

    def aggregate_protos(self, clients):
        aggregate_proto_dict = {}
        if self.args["weights"] == "uniform":
            label_counts = {}
            for client in clients:
                for label, proto in client.local_protos.items():
                    if label in aggregate_proto_dict:
                        aggregate_proto_dict[label] += proto
                        label_counts[label] += 1
                    else:
                        aggregate_proto_dict[label] = proto.clone()
                        label_counts[label] = 1
            for label in aggregate_proto_dict:
                aggregate_proto_dict[label] /= label_counts[label]
        else:
            label_size_dict = {}
            for client in clients:
                for label, proto in client.local_protos.items():
                    w = client.label_distribution[label]
                    if label in aggregate_proto_dict:
                        aggregate_proto_dict[label] += proto * w
                        label_size_dict[label] += w
                    else:
                        aggregate_proto_dict[label] = proto.clone() * w
                        label_size_dict[label] = w
            for label in aggregate_proto_dict:
                aggregate_proto_dict[label] /= max(label_size_dict[label], 1)

        if aggregate_proto_dict:
            feat_dim = next(iter(aggregate_proto_dict.values())).shape[0]
            zero_proto = torch.zeros(feat_dim, device=self.args["device"])
        else:
            self.model.eval()
            with torch.no_grad():
                _, dummy_feat = self.model(torch.randn(2, 3, 32, 32).to(self.args["device"]), return_feat=True)
                feat_dim = dummy_feat.shape[1]
            zero_proto = torch.zeros(feat_dim, device=self.args["device"])

        self.global_protos = [aggregate_proto_dict.get(label, zero_proto.clone())
                              for label in range(self.args["num_classes"])]

    def aggregate_label_params(self, label, clients):
        valid_clients = [c for c in clients if c.class_mask[label] == 1]
        if not valid_clients:
            return None
        ref_params = [p[label] for n, p in self.model.named_parameters() if n in self.clf_keys]
        agg = [torch.zeros_like(r) for r in ref_params]

        if self.args["weights"] == "uniform":
            for c in valid_clients:
                c_params = [p[label] for n, p in c.model.named_parameters() if n in self.clf_keys]
                for i in range(len(agg)):
                    agg[i] += c_params[i]
            factor = float(len(valid_clients))
        else:
            total_w = 0.0
            for c in valid_clients:
                w = float(c.label_distribution[label])
                total_w += w
                c_params = [p[label] for n, p in c.model.named_parameters() if n in self.clf_keys]
                for i in range(len(agg)):
                    agg[i] += c_params[i] * w
            factor = max(total_w, 1.0)
        for i in range(len(agg)):
            agg[i] /= factor
        return agg

    def aggregate_label_protos(self, label, clients):
        valid_clients = [c for c in clients if label in c.local_protos]
        if not valid_clients:
            return None
        if self.args["weights"] == "uniform":
            proto = None
            for c in valid_clients:
                if proto is None:
                    proto = c.local_protos[label].clone()
                else:
                    proto += c.local_protos[label]
            proto /= float(len(valid_clients))
        else:
            proto = None
            total_w = 0.0
            for c in valid_clients:
                w = float(c.label_distribution[label])
                total_w += w
                if proto is None:
                    proto = c.local_protos[label].clone() * w
                else:
                    proto += c.local_protos[label] * w
            proto /= max(total_w, 1.0)
        return proto

    @staticmethod
    def _compute_distance_matrix(vecs):
        return cosine_distances(vecs)

    def merge_classifiers(self, clf_params_dict, class_mask_dict):
        client_ids = np.array(list(clf_params_dict.keys()))
        client_clf_params = list(clf_params_dict.values())
        label_merged_dict = {}
        for label in range(self.args["num_classes"]):
            valid_indices, valid_params = [], []
            for idx, cid in enumerate(client_ids):
                if class_mask_dict[cid][label] == 1:
                    params = []
                    has_nan = False
                    for tensor_param in client_clf_params[idx]:
                        param_slice = tensor_param[label].detach().cpu().numpy().ravel()
                        if np.any(np.isnan(param_slice)) or np.any(np.isinf(param_slice)):
                            has_nan = True
                            break
                        params.append(param_slice)
                    if has_nan:
                        print(f"Warning: Client {cid} has NaN/inf in classifier params for label {label}, skipping.")
                        continue
                    valid_indices.append(idx)
                    valid_params.append(np.hstack(params))
            if len(valid_params) < 2:
                label_merged_dict[label] = []
                continue
            valid_params = np.array(valid_params)
            if np.any(np.isnan(valid_params)) or np.any(np.isinf(valid_params)):
                print(f"Warning: valid_params contains NaN/inf for label {label}, skipping clustering.")
                label_merged_dict[label] = []
                continue
            dist_matrix = FedMCServer._compute_distance_matrix(valid_params)
            clustering = DBSCAN(eps=self.args["eps"], min_samples=1, metric="precomputed")
            clustering.fit(dist_matrix)
            merged_ids = []
            for cluster_label in set(clustering.labels_):
                if cluster_label == -1:
                    continue
                idx_in_valid = np.where(clustering.labels_ == cluster_label)[0]
                original_ids = [client_ids[valid_indices[i]] for i in idx_in_valid]
                merged_ids.append(original_ids)
            label_merged_dict[label] = merged_ids
        return label_merged_dict

    def oracle_merging(self, _round, ids):
        if _round < 100:
            return {i: [ids] for i in range(10)}
        else:
            return {
                0: [ids],
                1: [[_id for _id in ids if 0 <= _id % 10 < 3], [_id for _id in ids if _id % 10 >= 3]],
                2: [[_id for _id in ids if 0 <= _id % 10 < 3], [_id for _id in ids if _id % 10 >= 3]],
                3: [[_id for _id in ids if 3 <= _id % 10 < 6], [_id for _id in ids if not 3 <= _id % 10 < 6]],
                4: [[_id for _id in ids if 3 <= _id % 10 < 6], [_id for _id in ids if not 3 <= _id % 10 < 6]],
                5: [[_id for _id in ids if _id % 10 >= 6], [_id for _id in ids if _id % 10 < 6]],
                6: [[_id for _id in ids if _id % 10 >= 6], [_id for _id in ids if _id % 10 < 6]],
                7: [ids],
                8: [ids],
                9: [ids]
            }

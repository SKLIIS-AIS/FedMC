# FedMC: Class-Wise Dual Knowledge Completion for Federated Learning under Extreme Missing Classes

This paper has been accepted by ICA3PP 2026, and the link to the original text is https://github.com/dawangLi050/FedMC

This repository provides the anonymous implementation of FedMC,
including the comparison baselines and ablation configurations used
in the paper.

FedMC addresses federated learning under extreme label skew, where most clients
observe only a small subset of the global classes. The method jointly completes
two forms of class-specific knowledge:

- **semantic knowledge**, represented by class prototypes; and
- **discriminative knowledge**, represented by class-specific classifier
  parameters, including both classifier weights and biases.

The final FedMC model uses class-presence masks to construct one aggregation
pool for each class and does **not** use DBSCAN. Aggregated
classifier--prototype pairs are used to complete missing knowledge at both
selected and non-selected clients.

## Repository Structure

```text
.
├── configs/
│   ├── FedAvg.yaml
│   ├── SCAFFOLD.yaml
│   ├── FedCCFA.yaml
│   ├── FedMC.yaml
│   └── ablations/
│       ├── FedMC_with_DBSCAN.yaml
│       ├── FedMC_wo_completion.yaml
│       ├── FedMC_wo_missing.yaml
│       └── FedMC_wo_proto_align.yaml
├── entities/
│   ├── init.py
│   ├── base.py
│   ├── FedAvg.py
│   ├── SCAFFOLD.py
│   ├── FedCCFA.py
│   └── FedMC.py
├── methods/
│   ├── init.py
│   ├── FedAvg.py
│   ├── SCAFFOLD.py
│   ├── FedCCFA.py
│   └── FedMC.py
├── utils/
│   ├── differential_privacy.py
│   ├── gen_dataset.py
│   ├── metric.py
│   └── models.py
├── requirements.txt
├── LICENSE
└── README.md
```

The baseline implementations, FedMC, and all ablations reuse the same data
partitioning, model definitions, evaluation functions, and shared base classes
in `entities/base.py` and `utils/`.

## Environment

The code is implemented in Python with PyTorch. A CUDA-capable GPU is
recommended for the 1,500-round experiments.

Recommended environment:

```text
Python >= 3.10
PyTorch >= 2.1
torchvision >= 0.16
NumPy >= 1.24
pandas >= 2.0
PyYAML >= 6.0
scikit-learn >= 1.3
tensorboard >= 2.14
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset and Federated Setting

The reported experiments use CIFAR-10. The dataset is downloaded
automatically by `torchvision` when it is not already available.

The common experimental setting is:

| Setting | Value |
|---|---:|
| Dataset | CIFAR-10 |
| Number of clients | 100 |
| Clients selected per round | 10 |
| Participation ratio | 0.1 |
| Dirichlet concentration | 0.05 |
| Backbone | CIFAR-style ResNet-18 |
| Communication rounds | 1,500 |
| Evaluation interval | 10 rounds |
| Reported random seed | 0 |

The dataset will be automatically downloaded by torchvision
when it is not available locally.

The partition is generated deterministically from the configured seed. Use the
same seed and configuration across all methods to reproduce an identical
federated partition.

## Compared Methods

The repository contains the following comparison baselines:

| Method | Description |
|---|---|
| FedAvg | Standard weighted full-model aggregation |
| SCAFFOLD | Federated optimization with server and client control variates |
| FedCCFA | Classifier clustering and feature alignment with its original DBSCAN stage |

FedCCFA retains DBSCAN because classifier clustering is part of the original
baseline. Removing DBSCAN from the final FedMC model does not imply removing it
from FedCCFA.

## FedMC and Reported Ablations

All variants are defined relative to the final no-DBSCAN FedMC.

| Configuration | DBSCAN | Gradient mask | Dual completion | Non-selected propagation | Prototype alignment |
|---|:---:|:---:|:---:|:---:|:---:|
| `FedMC.yaml` |  | ✓ | ✓ | ✓ | ✓ |
| `FedMC_with_DBSCAN.yaml` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FedMC_wo_completion.yaml` |  | ✓ |  |  | ✓ |
| `FedMC_wo_missing.yaml` |  |  |  |  | ✓ |
| `FedMC_wo_proto_align.yaml` |  | ✓ | ✓ | ✓ |  |

The variants have the following meanings:

- **FedMC**: the complete method. All valid selected clients that observe the
  same class form one class-wise aggregation pool.
- **FedMC+DBSCAN**: enables classifier clustering while retaining all other
  FedMC components.
- **FedMC w/o Completion**: disables missing-class classifier--prototype
  completion for selected clients and the corresponding classifier/prototype
  propagation to non-selected clients, while retaining missing-class gradient
  masking.
- **FedMC w/o Missing Handling**: additionally disables missing-class gradient
  masking. This is a nested ablation relative to FedMC w/o Completion.
- **FedMC w/o Prototype Alignment**: removes the prototype-alignment loss from
  local representation learning while retaining dual completion and gradient
  protection.

Only the four ablation variants reported in the paper are listed above.

## Running the Baselines

Run all commands from the repository root.

Create a directory for console logs:

```bash
mkdir -p logs
```

### FedAvg

```bash
python methods/FedAvg.py 2>&1 | tee logs/FedAvg_seed0.log
```

### SCAFFOLD

```bash
python methods/SCAFFOLD.py 2>&1 | tee logs/SCAFFOLD_seed0.log
```

### FedCCFA

```bash
python methods/FedCCFA.py 2>&1 | tee logs/FedCCFA_seed0.log
```

The baseline scripts read `configs/FedAvg.yaml`,
`configs/SCAFFOLD.yaml`, and `configs/FedCCFA.yaml`, respectively.
To run another random seed, change the `seed` field in the corresponding YAML
file.

## Running FedMC

### Main FedMC

```bash
python methods/FedMC.py \
  --config configs/FedMC.yaml \
  --seed 0 2>&1 | tee logs/FedMC_seed0.log
```

### FedMC+DBSCAN

```bash
python methods/FedMC.py \
  --config configs/ablations/FedMC_with_DBSCAN.yaml \
  --seed 0 2>&1 | tee logs/FedMC_with_DBSCAN_seed0.log
```

### FedMC w/o Completion

```bash
python methods/FedMC.py \
  --config configs/ablations/FedMC_wo_completion.yaml \
  --seed 0 2>&1 | tee logs/FedMC_wo_completion_seed0.log
```

### FedMC w/o Missing Handling

```bash
python methods/FedMC.py \
  --config configs/ablations/FedMC_wo_missing.yaml \
  --seed 0 2>&1 | tee logs/FedMC_wo_missing_seed0.log
```

### FedMC w/o Prototype Alignment

```bash
python methods/FedMC.py \
  --config configs/ablations/FedMC_wo_proto_align.yaml \
  --seed 0 2>&1 | tee logs/FedMC_wo_proto_align_seed0.log
```

The `--seed` argument overrides the seed stored in the selected FedMC YAML
file.

## Evaluation Metrics

The implementation reports three client-averaged metrics:

- **Local accuracy** (`A_local`): each client model is evaluated on its
  client-specific test set.
- **Global accuracy** (`A_global`): each client model is evaluated on the
  complete CIFAR-10 test set, and the accuracies are averaged across clients.
- **Missing-class accuracy** (`A_miss`): each client model is evaluated only on
  classes that are absent from that client's local training set.

FedMC and FedCCFA are evaluated using the latest shared representation and
each client's personalized classifier. FedAvg and SCAFFOLD broadcast the same
global model to all clients before client-level evaluation.

All methods are evaluated every 10 communication rounds. The paper reports the best-observed checkpoint according to
A_global on this common evaluation grid;
`A_local` and `A_miss` are taken from the same checkpoint.

## Outputs

During training, the scripts print round-level results in the following form:

```text
Round 1500 | Local: ... | Global: ... | Missing: ... | Best: ... @ ...
```

Console output can be retained with the `tee` commands shown above. Final
per-client CSV results are written by `utils/metric.py` to `../log`.

The temporal standard deviation printed by the training script measures
variation across late-stage checkpoints within one run. It is **not** a
standard deviation across independent random seeds.

## Reproducibility Notes

- The paper currently reports seed `0`.
- Use the same dataset partition seed, client count, sampling ratio, model, and
  evaluation interval for every method.
- Do not reuse an ablation result generated with DBSCAN for a no-DBSCAN
  ablation.
- Do not compare checkpoints selected using different evaluation intervals.

## Privacy Scope

The code exchanges model parameters and class prototypes but does not share raw
training examples. This property alone is not a formal privacy guarantee.
Differential privacy and secure aggregation are disabled in the reported
experiments.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Anonymous Review

This repository is intended for anonymous peer review. It deliberately omits
author names, affiliations, personal email addresses, and non-anonymous project
links. Author and citation information can be added after the review process.

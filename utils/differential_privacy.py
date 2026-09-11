import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_


class DifferentialPrivacy:
    """
    Stable DP module for Federated Learning
    """

    def __init__(self, epsilon=8.0, delta=1e-5,
                 max_grad_norm=1.0, noise_multiplier=0.5):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    # -------------------------
    # Gradient clipping
    # -------------------------
    def clip_gradients(self, model):
        clip_grad_norm_(model.parameters(), self.max_grad_norm)

    # -------------------------
    # Gaussian noise (DP-SGD)
    # -------------------------
    def add_gaussian_noise_to_gradients(self, model):
        sigma = self.noise_multiplier * self.max_grad_norm

        for p in model.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    0.0,
                    sigma,
                    size=p.grad.shape,
                    device=p.device
                )
                p.grad.add_(noise)

    # -------------------------
    # Vector DP
    # -------------------------
    def add_gaussian_noise_to_vector(self, vector, sensitivity=1.0):
        sigma = sensitivity * self.noise_multiplier
        noise = torch.normal(
            0.0,
            sigma,
            size=vector.shape,
            device=vector.device
        )
        return vector + noise

    # -------------------------
    # Laplace noise
    # -------------------------
    def add_laplace_noise_to_scalar(self, x, sensitivity=1.0):
        scale = sensitivity / max(self.noise_multiplier, 1e-8)
        noise = np.random.laplace(0, scale)
        return x + noise

    # -------------------------
    # sigma helper
    # -------------------------
    def compute_sigma(self, step=None):
        return self.noise_multiplier * self.max_grad_norm

    # -------------------------
    # privacy estimate (rough logging only)
    # -------------------------
    def compute_privacy_spent(self, steps):
        return self.noise_multiplier * np.sqrt(max(steps, 1)) / max(self.epsilon, 1e-6)
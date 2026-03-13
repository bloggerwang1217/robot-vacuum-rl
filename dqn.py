"""
DQN Network Architecture for Multi-Robot Environment
Supports: vanilla DQN, Dueling, NoisyNet, C51 (Distributional), and combinations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """
    Initialize network weights using Kaiming initialization

    Args:
        m: Network module
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# NoisyLinear (Factorized Gaussian — Fortunato et al. 2018)
# ---------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


def _make_linear(in_f: int, out_f: int, noisy: bool) -> nn.Module:
    """Helper: return NoisyLinear or nn.Linear."""
    return NoisyLinear(in_f, out_f) if noisy else nn.Linear(in_f, out_f)


# ---------------------------------------------------------------------------
# DQN — vanilla MLP (backward compatible)
# ---------------------------------------------------------------------------
class DQN(nn.Module):
    """
    Deep Q-Network with MLP architecture

    Network structure:
    - Input layer: obs_dim = 3 + (N-1)*3 + C*2 + n²
                   e.g. 2 robots + 1 charger, 5×5 → 3+3+2+25 = 33
    - Hidden layers: 128 -> 256 -> 256 -> 128
    - Output layer: num_actions (5: up/down/left/right/stay)
    """
    def __init__(self, num_actions, input_dim):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# DuelingDQN — V/A split (Wang et al. 2016)
# ---------------------------------------------------------------------------
class DuelingDQN(nn.Module):
    """
    Dueling DQN: shared feature trunk → value stream + advantage stream.
    Q(s,a) = V(s) + A(s,a) - mean(A)
    """
    def __init__(self, num_actions: int, input_dim: int, noisy: bool = False):
        super().__init__()
        self.num_actions = num_actions

        self.feature = nn.Sequential(
            _make_linear(input_dim, 128, noisy),
            nn.ReLU(),
            _make_linear(128, 256, noisy),
            nn.ReLU(),
            _make_linear(256, 256, noisy),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            _make_linear(256, 128, noisy),
            nn.ReLU(),
            _make_linear(128, 1, noisy),
        )
        self.advantage_stream = nn.Sequential(
            _make_linear(256, 128, noisy),
            nn.ReLU(),
            _make_linear(128, num_actions, noisy),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_stream(feat)            # (B, 1)
        advantage = self.advantage_stream(feat)    # (B, A)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ---------------------------------------------------------------------------
# C51DQN — Categorical / Distributional DQN (Bellemare et al. 2017)
# ---------------------------------------------------------------------------
class C51DQN(nn.Module):
    """
    Categorical DQN (C51).
    Outputs a probability distribution over `num_atoms` atoms for each action.
    Supports optional dueling and noisy layers.
    """
    def __init__(self, num_actions: int, input_dim: int,
                 num_atoms: int = 51, v_min: float = -100.0, v_max: float = 100.0,
                 dueling: bool = False, noisy: bool = False):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.register_buffer('atoms', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        self.dueling = dueling

        self.feature = nn.Sequential(
            _make_linear(input_dim, 128, noisy),
            nn.ReLU(),
            _make_linear(128, 256, noisy),
            nn.ReLU(),
            _make_linear(256, 256, noisy),
            nn.ReLU(),
        )

        if dueling:
            self.value_stream = nn.Sequential(
                _make_linear(256, 128, noisy),
                nn.ReLU(),
                _make_linear(128, num_atoms, noisy),
            )
            self.advantage_stream = nn.Sequential(
                _make_linear(256, 128, noisy),
                nn.ReLU(),
                _make_linear(128, num_actions * num_atoms, noisy),
            )
        else:
            self.head = nn.Sequential(
                _make_linear(256, 128, noisy),
                nn.ReLU(),
                _make_linear(128, num_actions * num_atoms, noisy),
            )

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Return action-atom log-probabilities: (B, A, num_atoms)."""
        feat = self.feature(x)

        if self.dueling:
            val = self.value_stream(feat).unsqueeze(1)                   # (B, 1, N_atoms)
            adv = self.advantage_stream(feat).view(-1, self.num_actions, self.num_atoms)  # (B, A, N_atoms)
            logits = val + adv - adv.mean(dim=1, keepdim=True)
        else:
            logits = self.head(feat).view(-1, self.num_actions, self.num_atoms)

        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values (expected value under distribution): (B, A)."""
        log_probs = self.dist(x)
        probs = log_probs.exp()
        q = (probs * self.atoms.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


def build_network(num_actions: int, input_dim: int,
                  dueling: bool = False, noisy: bool = False,
                  c51: bool = False, num_atoms: int = 51,
                  v_min: float = -100.0, v_max: float = 100.0) -> nn.Module:
    """
    Factory function: build the appropriate DQN variant.
    Returns a standard DQN-compatible module (forward(x) → Q-values).
    """
    if c51:
        return C51DQN(num_actions, input_dim, num_atoms=num_atoms,
                       v_min=v_min, v_max=v_max,
                       dueling=dueling, noisy=noisy)
    elif dueling:
        return DuelingDQN(num_actions, input_dim, noisy=noisy)
    elif noisy:
        # NoisyNet-only: same architecture as DQN but with NoisyLinear
        return DuelingDQN(num_actions, input_dim, noisy=True)  # reuse Dueling arch
    else:
        return DQN(num_actions, input_dim)

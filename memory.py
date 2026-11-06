"""Directly writable, initially inactive FFN memory (experimental).

This implements storage and insertion, not a semantic encoder or automatic
attention-state writer. Program weights explicitly between forward passes.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class WeightMemory(nn.Module):
    """Residual SwiGLU memory: x + strength * down(silu(gate(x)) * up(x)).

    All three effective matrices start at zero. Raw weights can start at either
    zero or one; the serialized baseline is subtracted before computation.
    Defaults match the Qwen3.5-0.8B FFN dimensions (11,010,048 weights).
    Keep this module in FP32, especially with an all-ones baseline. Input/output
    activations may be lower precision. No additional normalization is applied:
    the wrapper receives the existing normalized pre-FFN activation.
    """
    def __init__(self, hidden_size=1024, intermediate_size=3584,
                 initial_value=0.0, strength=1.0):
        super().__init__()
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
               for n in (hidden_size, intermediate_size)):
            raise ValueError('Dimensions must be positive integers')
        if initial_value not in (0, 1) or not math.isfinite(strength):
            raise ValueError('Initial value must be 0 or 1; strength must be finite')
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.register_buffer('_baseline', torch.tensor(float(initial_value)))
        self.register_buffer('_strength', torch.tensor(float(strength)))
        self.gate_weight = nn.Parameter(
            torch.full((intermediate_size, hidden_size), float(initial_value)),
            requires_grad=False)
        self.up_weight = nn.Parameter(
            torch.full((intermediate_size, hidden_size), float(initial_value)),
            requires_grad=False)
        self.down_weight = nn.Parameter(
            torch.full((hidden_size, intermediate_size), float(initial_value)),
            requires_grad=False)

    def forward(self, x):
        if x.shape[-1] != self.hidden_size:
            raise ValueError('Input width does not match hidden_size')
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError('Keep memory weights in FP32; cast the base model before wrapping')
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = x.float()
            gate = F.silu(F.linear(z, self.gate_weight - self._baseline))
            up = F.linear(z, self.up_weight - self._baseline)
            read = F.linear(gate * up, self.down_weight - self._baseline)
            correction = self._strength * read
        return x + correction.to(x.dtype)

    def _check_index(self, index):
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError('Memory unit index must be an integer')
        if not 0 <= index < self.intermediate_size:
            raise IndexError('Memory unit index out of range')

    @torch.no_grad()
    def read_unit(self, index):
        """Return independent copies of one unit's effective weights."""
        self._check_index(index)
        return {
            'gate': (self.gate_weight[index] - self._baseline).clone(),
            'up': (self.up_weight[index] - self._baseline).clone(),
            'down': (self.down_weight[:, index] - self._baseline).clone(),
        }

    @torch.no_grad()
    def program_unit(self, index, *, gate, up, down):
        """Replace two input rows and one output column; no backpropagation.

        Inputs are EFFECTIVE weights, independent of the initial baseline.
        Validate the complete write before mutation. Returns the prior effective
        weights, usable with program_unit(index, **previous) for rollback.
        A default-sized write touches at most 3 * 1024 = 3072 parameters.
        """
        self._check_index(index)
        values = []
        for name, vector in (('gate', gate), ('up', up), ('down', down)):
            vector = torch.as_tensor(vector, device=self.gate_weight.device,
                                     dtype=torch.float32).detach().clone()
            if vector.shape != (self.hidden_size,) or not torch.isfinite(vector).all():
                raise ValueError(f'{name} must be a finite vector of length {self.hidden_size}')
            values.append(vector)
        previous = self.read_unit(index)
        self.gate_weight[index].copy_(values[0] + self._baseline)
        self.up_weight[index].copy_(values[1] + self._baseline)
        self.down_weight[:, index].copy_(values[2] + self._baseline)
        return previous

    @torch.no_grad()
    def clear_unit(self, index):
        self._check_index(index)
        baseline = self._baseline.item()
        self.gate_weight[index].fill_(baseline)
        self.up_weight[index].fill_(baseline)
        self.down_weight[:, index].fill_(baseline)

    @torch.no_grad()
    def reset(self):
        for parameter in self.parameters():
            parameter.fill_(self._baseline.item())


class MemoryBeforeFFN(nn.Module):
    """Insert memory after existing pre-FFN normalization, before the old MLP.

    Load/cast/place the base model first, then wrap one layer's mlp. Base model
    weights are retained. This changes state_dict key names under that mlp;
    restore the same wrapper architecture before loading a combined checkpoint.
    """
    def __init__(self, mlp, hidden_size=1024, intermediate_size=3584, **options):
        super().__init__()
        self.mlp = mlp
        self.memory = WeightMemory(hidden_size, intermediate_size, **options)
        parameter = next(mlp.parameters(), None)
        if parameter is not None:
            self.memory.to(device=parameter.device)

    def forward(self, x):
        return self.mlp(self.memory(x))

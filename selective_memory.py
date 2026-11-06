"""A small, explicitly programmed associative residual memory.

No text or token IDs are stored in this module. Key and delta matrices are
persistent parameters, initialized to zero and written without gradients.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SelectiveMemory(nn.Module):
    def __init__(self, width, slots=16):
        super().__init__()
        self.width = width
        self.slots = slots
        self.keys = nn.Parameter(torch.zeros(slots, width), requires_grad=False)
        self.deltas = nn.Parameter(torch.zeros(slots, width), requires_grad=False)
        self.register_buffer('occupied', torch.zeros(slots, dtype=torch.bool))
        self.register_buffer('thresholds', torch.full((slots,), 0.95))

    def _slot(self, slot):
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < self.slots:
            raise ValueError('Invalid memory slot')

    @torch.no_grad()
    def program(self, slot, source, target, threshold=0.95):
        """Write a recognition key and target-minus-source activation delta."""
        self._slot(slot)
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('Threshold must lie strictly between 0 and 1')
        vectors = []
        for vector in (source, target):
            vector = torch.as_tensor(vector, dtype=self.keys.dtype, device=self.keys.device).clone()
            if vector.shape != (self.width,) or not torch.isfinite(vector).all():
                raise ValueError('Expected a finite hidden vector with the configured width')
            vectors.append(vector)
        source, target = vectors
        if source.norm() < 1e-8:
            raise ValueError('A zero source cannot be used as a recognition key')
        delta = target - source
        if not torch.isfinite(delta).all():
            raise ValueError('Activation difference is not finite')
        self.keys[slot].copy_(F.normalize(source, dim=0))
        self.deltas[slot].copy_(delta)
        self.thresholds[slot] = threshold
        self.occupied[slot] = True

    @torch.no_grad()
    def edit_row(self, matrix, slot, values):
        """Replace raw memory weights exactly, without normalization or enabling.

        Values must be finite and representable in the matrix's storage dtype.
        All validation happens before any weight changes.
        """
        self._slot(slot)
        matrices = {'keys': self.keys, 'deltas': self.deltas}
        if matrix not in matrices:
            raise ValueError('Choose the keys or deltas matrix')
        weight = matrices[matrix]
        candidate = torch.as_tensor(values, dtype=weight.dtype, device=weight.device).detach().clone()
        if candidate.shape != (self.width,) or not torch.isfinite(candidate).all():
            raise ValueError(f'Enter {self.width} finite values representable as {weight.dtype}')
        weight[slot].copy_(candidate)

    @torch.no_grad()
    def set_slot_options(self, slot, *, enabled=None, threshold=None):
        self._slot(slot)
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError('Enabled must be true or false')
        if threshold is not None:
            if not math.isfinite(threshold) or not 0 < threshold < 1:
                raise ValueError('Threshold must lie strictly between 0 and 1')
            stored = torch.as_tensor(threshold, dtype=self.thresholds.dtype)
            if not 0 < stored.item() < 1:
                raise ValueError('Threshold must remain between 0 and 1 at storage precision')
        if enabled is not None:
            self.occupied[slot] = enabled
        if threshold is not None:
            self.thresholds[slot] = threshold

    def lookup(self, x):
        if x.shape[-1] != self.width:
            raise ValueError('Wrong hidden width')
        scores = F.linear(F.normalize(x, dim=-1), self.keys)
        eligible = self.occupied & (scores >= self.thresholds)
        # Highest similarity wins among eligible slots; ties prefer lower index.
        masked = scores.masked_fill(~eligible, -torch.inf)
        _, index = masked.max(dim=-1)
        active = eligible.any(dim=-1)
        best_similarity = scores.masked_fill(~self.occupied, -torch.inf).max(dim=-1).values
        return index, active, best_similarity

    def forward(self, x):
        index, active, _ = self.lookup(x)
        delta = self.deltas[index]
        return x + torch.where(active.unsqueeze(-1), delta, torch.zeros_like(delta))

    @torch.no_grad()
    def clear(self, slot=None):
        if slot is None:
            self.keys.zero_()
            self.deltas.zero_()
            self.occupied.zero_()
            self.thresholds.fill_(0.95)
        else:
            self._slot(slot)
            self.keys[slot].zero_()
            self.deltas[slot].zero_()
            self.occupied[slot] = False
            self.thresholds[slot] = 0.95

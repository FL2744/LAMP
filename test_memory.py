import io
import unittest
import torch
from torch import nn
from memory import WeightMemory, MemoryBeforeFFN


class MemoryTests(unittest.TestCase):
    def test_default_parameter_count(self):
        memory = WeightMemory()
        self.assertEqual(sum(p.numel() for p in memory.parameters()), 11_010_048)
        self.assertTrue(all(not p.requires_grad for p in memory.parameters()))

    def test_blank_memory_and_wrapped_ffn_are_identity(self):
        for baseline in (0, 1):
            for dtype in (torch.float32, torch.bfloat16):
                mlp = nn.Sequential(nn.Linear(4, 7), nn.SiLU(), nn.Linear(7, 4)).to(dtype)
                wrapped = MemoryBeforeFFN(mlp, 4, 5, initial_value=baseline)
                x = torch.randn(2, 6, 4).to(dtype)
                self.assertTrue(all(torch.all(p == baseline) for p in wrapped.memory.parameters()))
                self.assertTrue(torch.equal(wrapped.memory(x), x))
                self.assertTrue(torch.equal(wrapped(x), mlp(x)))

    def test_complete_unit_write_changes_only_its_connections(self):
        for baseline in (0, 1):
            memory = WeightMemory(4, 5, initial_value=baseline)
            gate = torch.tensor([1., 2., 3., 4.])
            up = torch.tensor([2., 1., 4., 3.])
            down = torch.tensor([.25, .5, -.25, -.5])
            previous = memory.program_unit(2, gate=gate, up=up, down=down)
            self.assertEqual(sum(torch.count_nonzero(p - baseline).item()
                                 for p in memory.parameters()), 12)
            x = torch.tensor([[.25, -.25, .5, .25]])
            expected = x + (torch.nn.functional.silu(x @ gate) * (x @ up)).unsqueeze(-1) * down
            torch.testing.assert_close(memory(x), expected)
            self.assertFalse(torch.equal(memory(x), x))
            memory.program_unit(2, **previous)
            self.assertTrue(torch.equal(memory(x), x))

    def test_input_weights_alone_have_no_output_path(self):
        memory = WeightMemory(4, 5)
        memory.program_unit(0, gate=torch.ones(4), up=torch.ones(4), down=torch.zeros(4))
        x = torch.ones(1, 4)
        self.assertTrue(torch.equal(memory(x), x))

    def test_invalid_write_preserves_entire_state(self):
        memory = WeightMemory(4, 5)
        before = {k: v.clone() for k, v in memory.state_dict().items()}
        with self.assertRaises(ValueError):
            memory.program_unit(0, gate=torch.ones(4), up=torch.ones(4),
                                down=[1., 2., float('nan'), 3.])
        for k, v in memory.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        with self.assertRaises(IndexError):
            memory.clear_unit(-1)

    def test_checkpoint_preserves_baseline_strength_and_memory(self):
        memory = WeightMemory(4, 5, initial_value=1, strength=.5)
        memory.program_unit(0, gate=torch.ones(4), up=torch.ones(4), down=torch.ones(4))
        stream = io.BytesIO()
        torch.save(memory.state_dict(), stream)
        stream.seek(0)
        restored = WeightMemory(4, 5)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        x = torch.randn(2, 4)
        self.assertTrue(torch.equal(memory(x), restored(x)))
        restored.clear_unit(0)
        self.assertTrue(torch.equal(restored(x), x))

    def test_forward_does_not_write(self):
        memory = WeightMemory(4, 5)
        memory.program_unit(0, gate=torch.ones(4), up=torch.ones(4), down=torch.ones(4))
        before = {k: v.clone() for k, v in memory.state_dict().items()}
        memory(torch.randn(2, 4))
        for k, v in memory.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        memory.reset()
        self.assertTrue(all(torch.count_nonzero(p) == 0 for p in memory.parameters()))


if __name__ == '__main__':
    unittest.main()
